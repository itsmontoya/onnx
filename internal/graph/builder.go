package graph

import (
	"context"
	"fmt"
	"slices"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
)

func newBuilder(ctx context.Context, model *onnxpb.ModelProto, encodedBytes int64, limits Limits, destination *Graph) (out *builder, err error) {
	var b builder
	if ctx == nil || model == nil || encodedBytes < 0 {
		return nil, fmt.Errorf("%w: context, model or encoded byte count", ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("build graph: %w", err)
	}

	if b.limits, err = limits.normalized(); err != nil {
		return nil, err
	}

	if err = validateEnvelope(model); err != nil {
		return nil, err
	}

	if err = validateFeatures(ctx, model.Graph, b.limits); err != nil {
		return nil, err
	}

	if b.decoder, err = decode.NewTensorDecoder(ctx, model, encodedBytes, b.limits.Tensor); err != nil {
		return nil, fmt.Errorf("build graph payload: %w", err)
	}

	b.ctx = ctx
	b.graph = destination
	b.graph.indices = make(map[string]int)
	return &b, nil
}

type builder struct {
	ctx     context.Context
	limits  Limits
	decoder *decode.TensorDecoder
	graph   *Graph
}

func (b *builder) build(model *onnxpb.ModelProto) (err error) {
	if err = validateStrings(b.ctx, model.ProtoReflect(), "model", b.limits.MaxNameBytes); err != nil {
		return err
	}

	if b.graph.metadata, err = readMetadata(model); err != nil {
		return err
	}

	if err = b.inputs(model.Graph.Input); err != nil {
		return err
	}

	if err = b.initializers(model.Graph.Initializer); err != nil {
		return err
	}

	for index, wire := range model.Graph.Node {
		if err = b.node(index, wire); err != nil {
			return nodeError(index, wire, err)
		}
	}

	if b.graph.outputs, err = b.declarations(model.Graph.Output, true, "output"); err != nil {
		return err
	}

	if _, err = b.declarations(model.Graph.ValueInfo, false, "value_info"); err != nil {
		return err
	}

	if err = b.ctx.Err(); err != nil {
		return fmt.Errorf("build graph: %w", err)
	}

	return nil
}

func (b *builder) inputs(wires []*onnxpb.ValueInfoProto) (err error) {
	for _, wire := range wires {
		var info ValueInfo
		if info, err = b.info(wire, true, true); err != nil {
			return err
		}

		if err = b.addValue(Value{Name: info.Name, Producer: -1, Input: true, Declarations: []ValueInfo{info}}); err != nil {
			return fmt.Errorf("graph input %q: %w", info.Name, err)
		}

		b.graph.inputs = append(b.graph.inputs, info)
	}
	return nil
}

func (b *builder) initializers(wires []*onnxpb.TensorProto) (err error) {
	seen := make(map[string]bool)
	for index, wire := range wires {
		if wire == nil || wire.GetName() == "" {
			return fmt.Errorf("initializer[%d] value %q: %w: nonempty name required", index, wire.GetName(), ErrInvalidModel)
		}

		name := wire.GetName()
		if seen[name] {
			return fmt.Errorf("initializer value %q: %w: duplicate initializer", name, ErrInvalidModel)
		}

		seen[name] = true
		var constant *Constant
		if constant, err = newConstant(b.ctx, b.decoder, wire); err != nil {
			return fmt.Errorf("initializer value %q: %w", name, err)
		}

		valueIndex, exists := b.graph.indices[name]
		if !exists {
			if err = b.addValue(Value{Name: name, Producer: -1, Default: constant}); err != nil {
				return err
			}

			continue
		}

		value := &b.graph.values[valueIndex]
		if err = validateConstant(value.Declarations[0], constant); err != nil {
			return fmt.Errorf("default initializer value %q: %w", name, err)
		}

		value.Default = constant
		value.Declarations[0].Defaulted = true
		b.graph.inputs[valueIndex].Defaulted = true
	}
	return nil
}

func (b *builder) node(index int, wire *onnxpb.NodeProto) (err error) {
	if err = b.ctx.Err(); err != nil {
		return err
	}

	var n Node
	n.Index = index
	n.Name = wire.GetName()
	n.Operator = wire.GetOpType()
	n.Domain = wire.GetDomain()
	n.DocString = wire.GetDocString()
	if n.Operator == "" || len(wire.Input)+len(wire.Output) == 0 {
		return fmt.Errorf("%w: operator and at least one input/output slot required", ErrInvalidModel)
	}

	for slot, name := range wire.Input {
		if name == "" {
			continue
		}

		if _, exists := b.graph.indices[name]; !exists {
			return fmt.Errorf("%w: input[%d] value %q has no preceding producer (missing reference, forward reference or cycle)", ErrInvalidModel, slot, name)
		}
	}

	for slot, name := range wire.Output {
		if name == "" {
			continue
		}

		if err = b.addValue(Value{Name: name, Producer: index}); err != nil {
			return fmt.Errorf("output[%d] value %q: %w", slot, name, err)
		}
	}

	n.Inputs = slices.Clone(wire.Input)
	n.Outputs = slices.Clone(wire.Output)
	if n.Metadata, err = properties(wire.MetadataProps); err != nil {
		return err
	}

	for _, attribute := range wire.Attribute {
		var value Attribute
		if value, err = b.attribute(attribute); err != nil {
			return fmt.Errorf("attribute %q: %w", attribute.GetName(), err)
		}

		n.Attributes = append(n.Attributes, value)
	}
	if n.Operator == "Constant" && len(n.Attributes) == 1 {
		attribute := &n.Attributes[0]
		expected := map[string]AttributeKind{"value_float": FloatAttribute, "value_floats": FloatsAttribute, "value_int": IntAttribute, "value_ints": IntsAttribute}
		if kind, exists := expected[attribute.Name]; exists && attribute.Kind == kind {
			if attribute.Tensor, err = newAttributeConstant(b.ctx, b.decoder, *attribute); err != nil {
				return err
			}
		}
	}

	b.graph.nodes = append(b.graph.nodes, n)
	return nil
}

func (b *builder) addValue(value Value) (err error) {
	if _, exists := b.graph.indices[value.Name]; exists {
		return fmt.Errorf("%w: value %q has duplicate producers", ErrInvalidModel, value.Name)
	}

	if len(b.graph.values) >= b.limits.MaxValues {
		return fmt.Errorf("%w: value %q exceeds named value count", ErrLimit, value.Name)
	}

	b.graph.indices[value.Name] = len(b.graph.values)
	b.graph.values = append(b.graph.values, value)
	return nil
}

func (b *builder) declarations(wires []*onnxpb.ValueInfoProto, required bool, label string) (infos []ValueInfo, err error) {
	seen := make(map[string]bool)
	for _, wire := range wires {
		var info ValueInfo
		if info, err = b.info(wire, required, false); err != nil {
			return nil, fmt.Errorf("graph %s: %w", label, err)
		}

		if seen[info.Name] {
			return nil, fmt.Errorf("graph %s value %q: %w: duplicate declaration", label, info.Name, ErrInvalidModel)
		}

		seen[info.Name] = true
		valueIndex, exists := b.graph.indices[info.Name]
		if !exists {
			return nil, fmt.Errorf("graph %s value %q: %w: no producer", label, info.Name, ErrInvalidModel)
		}

		value := &b.graph.values[valueIndex]
		for _, previous := range value.Declarations {
			if err = compatibleInfo(previous, info); err != nil {
				return nil, fmt.Errorf("graph %s value %q: %w", label, info.Name, err)
			}
		}

		if value.Default != nil && !value.Input {
			if err = validateConstant(info, value.Default); err != nil {
				return nil, fmt.Errorf("graph %s value %q: %w", label, info.Name, err)
			}
		}

		value.Declarations = append(value.Declarations, info)
		infos = append(infos, info)
	}
	return infos, nil
}

func (b *builder) info(wire *onnxpb.ValueInfoProto, required, input bool) (info ValueInfo, err error) {
	if err = b.ctx.Err(); err != nil {
		return info, fmt.Errorf("value %q: %w", wire.GetName(), err)
	}

	// Dimension records duplicate logical dimension widths, not Go struct headers.
	if wire.Type != nil {
		typed, dense := wire.Type.Value.(*onnxpb.TypeProto_TensorType)
		if dense && typed != nil && typed.TensorType != nil && typed.TensorType.Shape != nil {
			rank := len(typed.TensorType.Shape.Dim)
			if rank > b.limits.Tensor.MaxRank {
				return info, fmt.Errorf("value %q: %w: metadata rank %d", wire.GetName(), ErrLimit, rank)
			}

			if err = b.decoder.ReservePayload(b.ctx, int64(rank)*8); err != nil {
				return info, fmt.Errorf("value %q: %w", wire.GetName(), err)
			}
		}
	}

	if info, err = readInfo(wire, required, input, b.limits); err != nil {
		return info, fmt.Errorf("value %q: %w", wire.GetName(), err)
	}

	return info, nil
}

func (b *builder) attribute(wire *onnxpb.AttributeProto) (attribute Attribute, err error) {
	attribute.Name = wire.GetName()
	attribute.DocString = wire.GetDocString()
	var bytes int64
	switch wire.GetType() {
	case onnxpb.AttributeProto_FLOAT:
		attribute.Kind = FloatAttribute
		attribute.Float = wire.GetF()
		bytes = 4
	case onnxpb.AttributeProto_INT:
		attribute.Kind = IntAttribute
		attribute.Int = wire.GetI()
		bytes = 8
	case onnxpb.AttributeProto_STRING:
		attribute.Kind = StringAttribute
		bytes = int64(len(wire.S))
	case onnxpb.AttributeProto_TENSOR:
		attribute.Kind = TensorAttribute
		if attribute.Tensor, err = newConstant(b.ctx, b.decoder, wire.T); err != nil {
			return attribute, err
		}
	case onnxpb.AttributeProto_FLOATS:
		attribute.Kind = FloatsAttribute
		bytes = int64(len(wire.Floats)) * 4
	case onnxpb.AttributeProto_INTS:
		attribute.Kind = IntsAttribute
		bytes = int64(len(wire.Ints)) * 8
	case onnxpb.AttributeProto_STRINGS:
		attribute.Kind = StringsAttribute
		// Reserve each string before allocating its snapshot; never sum unchecked.
		for _, value := range wire.Strings {
			if err = b.decoder.ReservePayload(b.ctx, int64(len(value))); err != nil {
				return attribute, err
			}
		}
	case onnxpb.AttributeProto_TENSORS:
		attribute.Kind = TensorsAttribute
		for _, value := range wire.Tensors {
			var constant *Constant
			if constant, err = newConstant(b.ctx, b.decoder, value); err != nil {
				return attribute, err
			}

			attribute.Tensors = append(attribute.Tensors, constant)
		}
	}

	if err = b.decoder.ReservePayload(b.ctx, bytes); err != nil {
		return attribute, err
	}

	attribute.String = slices.Clone(wire.S)
	attribute.Floats = slices.Clone(wire.Floats)
	attribute.Ints = slices.Clone(wire.Ints)
	attribute.Strings = make([][]byte, len(wire.Strings))
	for index, value := range wire.Strings {
		attribute.Strings[index] = slices.Clone(value)
	}
	return attribute, nil
}
