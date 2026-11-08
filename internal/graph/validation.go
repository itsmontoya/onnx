package graph

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func validateEnvelope(model *onnxpb.ModelProto) (err error) {
	if model.IrVersion == nil {
		return fmt.Errorf("model ir_version: %w: required field missing", ErrInvalidModel)
	}

	if model.GetIrVersion() != 8 {
		return fmt.Errorf("model ir_version %d: %w: require IR 8", model.GetIrVersion(), ErrUnsupported)
	}

	if len(model.OpsetImport) == 0 {
		return fmt.Errorf("model opset_import: %w: standard-domain import required", ErrInvalidModel)
	}

	seen := make(map[string]bool)
	for index, imported := range model.OpsetImport {
		if imported == nil || imported.Version == nil {
			return fmt.Errorf("model opset_import[%d]: %w: version required", index, ErrInvalidModel)
		}

		domain := imported.GetDomain()
		if seen[domain] {
			return fmt.Errorf("model opset_import[%d] domain %q: %w: duplicate import", index, domain, ErrInvalidModel)
		}

		seen[domain] = true
		if domain != "" || imported.GetVersion() != 13 {
			return fmt.Errorf("model opset_import[%d] domain %q version %d: %w: require empty domain opset 13", index, domain, imported.GetVersion(), ErrUnsupported)
		}
	}

	if len(model.Functions) != 0 || len(model.TrainingInfo) != 0 {
		return fmt.Errorf("model functions/training_info: %w", ErrUnsupported)
	}

	if model.Graph == nil || model.Graph.GetName() == "" {
		return fmt.Errorf("graph name %q: %w: named top-level graph required", model.Graph.GetName(), ErrInvalidModel)
	}

	return nil
}

func validateFeatures(ctx context.Context, wire *onnxpb.GraphProto, limits Limits) (err error) {
	if len(wire.Node) > limits.MaxNodes {
		return fmt.Errorf("graph %q: %w: node count", wire.GetName(), ErrLimit)
	}

	if len(wire.SparseInitializer) != 0 {
		return fmt.Errorf("graph %q sparse_initializer: %w", wire.GetName(), ErrUnsupported)
	}

	for _, declarations := range [][]*onnxpb.ValueInfoProto{wire.Input, wire.Output, wire.ValueInfo} {
		for index, info := range declarations {
			if err = ctx.Err(); err != nil {
				return fmt.Errorf("value %q: %w", info.GetName(), err)
			}

			if info == nil || info.GetName() == "" {
				return fmt.Errorf("value[%d] name %q: %w: nonempty declaration required", index, info.GetName(), ErrInvalidModel)
			}

			if info.Type != nil && info.Type.Value != nil {
				if _, dense := info.Type.Value.(*onnxpb.TypeProto_TensorType); !dense {
					return fmt.Errorf("value %q: %w: require dense tensor type", info.GetName(), ErrUnsupported)
				}
			}
		}
	}
	var attributes int
	for index, node := range wire.Node {
		if err = ctx.Err(); err != nil {
			return nodeError(index, node, err)
		}

		if node == nil {
			return nodeError(index, node, fmt.Errorf("%w: nil node", ErrInvalidModel))
		}

		if node.GetDomain() != "" || node.GetOverload() != "" {
			return nodeError(index, node, fmt.Errorf("%w: custom domain or function overload", ErrUnsupported))
		}

		if len(node.Attribute) > limits.MaxAttributes-attributes {
			return nodeError(index, node, fmt.Errorf("%w: aggregate attribute count", ErrLimit))
		}

		attributes += len(node.Attribute)
		seen := make(map[string]bool)
		for slot, attribute := range node.Attribute {
			if err = ctx.Err(); err != nil {
				return nodeError(index, node, err)
			}

			if attribute == nil || attribute.GetName() == "" {
				return nodeError(index, node, fmt.Errorf("attribute[%d] name %q: %w: nonempty attribute required", slot, attribute.GetName(), ErrInvalidModel))
			}

			name := attribute.GetName()
			if seen[name] {
				return nodeError(index, node, fmt.Errorf("attribute %q: %w: duplicate attribute", name, ErrInvalidModel))
			}

			seen[name] = true
			if err = validateAttribute(attribute); err != nil {
				return nodeError(index, node, fmt.Errorf("attribute %q: %w", name, err))
			}
		}
	}
	return nil
}

// validateValueCount rejects the whole graph before any initializer conversion.
// Defaulted inputs reuse their named source; empty optional slots are absent.
func validateValueCount(ctx context.Context, wire *onnxpb.GraphProto, limit int) (err error) {
	seen := make(map[string]bool)
	for _, input := range wire.Input {
		if err = countValue(ctx, seen, input.GetName(), limit); err != nil {
			return err
		}
	}
	for _, initializer := range wire.Initializer {
		if err = countValue(ctx, seen, initializer.GetName(), limit); err != nil {
			return err
		}
	}
	for index, node := range wire.Node {
		for _, name := range node.Output {
			if err = countValue(ctx, seen, name, limit); err != nil {
				return nodeError(index, node, err)
			}
		}
	}
	return nil
}

func countValue(ctx context.Context, seen map[string]bool, name string, limit int) (err error) {
	if err = ctx.Err(); err != nil {
		return fmt.Errorf("value %q: %w", name, err)
	}

	if name == "" || seen[name] {
		return nil
	}

	if len(seen) >= limit {
		return fmt.Errorf("%w: value %q exceeds named value count", ErrLimit, name)
	}

	seen[name] = true
	return nil
}

func validateAttribute(wire *onnxpb.AttributeProto) (err error) {
	if wire.RefAttrName != nil || wire.G != nil || len(wire.Graphs) != 0 || wire.SparseTensor != nil || len(wire.SparseTensors) != 0 || wire.Tp != nil || len(wire.TypeProtos) != 0 {
		return fmt.Errorf("%w: function reference, subgraph, sparse or type-valued attribute", ErrUnsupported)
	}

	switch wire.GetType() {
	case onnxpb.AttributeProto_FLOAT, onnxpb.AttributeProto_INT, onnxpb.AttributeProto_STRING, onnxpb.AttributeProto_FLOATS, onnxpb.AttributeProto_INTS, onnxpb.AttributeProto_STRINGS, onnxpb.AttributeProto_TENSORS:
	case onnxpb.AttributeProto_TENSOR:
		if wire.T == nil {
			return fmt.Errorf("%w: tensor attribute payload required", ErrInvalidModel)
		}

	case onnxpb.AttributeProto_UNDEFINED:
		return fmt.Errorf("%w: defined attribute type required", ErrInvalidModel)
	default:
		return fmt.Errorf("%w: attribute type %d", ErrUnsupported, wire.GetType())
	}

	// Singular zero defaults and empty repeated payloads are valid in proto3
	// exports. Every present content field must still match the discriminator.
	for _, payload := range []struct {
		present bool
		kind    onnxpb.AttributeProto_AttributeType
	}{
		{wire.F != nil, onnxpb.AttributeProto_FLOAT},
		{wire.I != nil, onnxpb.AttributeProto_INT},
		{wire.S != nil, onnxpb.AttributeProto_STRING},
		{wire.T != nil, onnxpb.AttributeProto_TENSOR},
		{len(wire.Floats) != 0, onnxpb.AttributeProto_FLOATS},
		{len(wire.Ints) != 0, onnxpb.AttributeProto_INTS},
		{len(wire.Strings) != 0, onnxpb.AttributeProto_STRINGS},
		{len(wire.Tensors) != 0, onnxpb.AttributeProto_TENSORS},
	} {
		if payload.present && wire.GetType() != payload.kind {
			return fmt.Errorf("%w: attribute type/payload mismatch", ErrInvalidModel)
		}
	}

	for index, value := range wire.Tensors {
		if value == nil {
			return fmt.Errorf("%w: nil tensor at attribute payload[%d]", ErrInvalidModel, index)
		}
	}
	return nil
}

func nodeError(index int, wire *onnxpb.NodeProto, cause error) (err error) {
	return fmt.Errorf("node[%d] name %q domain %q operator %q opset 13: %w", index, wire.GetName(), wire.GetDomain(), wire.GetOpType(), cause)
}

func validateStrings(ctx context.Context, message protoreflect.Message, path string, limit int) (err error) {
	if err = ctx.Err(); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}

	fields := message.Descriptor().Fields()
	for fieldIndex := 0; fieldIndex < fields.Len(); fieldIndex++ {
		field := fields.Get(fieldIndex)
		if !message.Has(field) {
			continue
		}

		if field.Kind() != protoreflect.MessageKind && field.Kind() != protoreflect.StringKind && (field.Kind() != protoreflect.BytesKind || field.ContainingMessage().Name() != "AttributeProto") {
			continue
		}

		value := message.Get(field)
		fieldPath := path + "." + string(field.Name())
		if field.IsList() {
			list := value.List()
			for index := 0; index < list.Len(); index++ {
				if err = validateStringField(ctx, field, list.Get(index), fmt.Sprintf("%s[%d]", fieldPath, index), limit); err != nil {
					return err
				}
			}
		} else {
			err = validateStringField(ctx, field, value, fieldPath, limit)
		}

		if err != nil {
			return err
		}
	}
	return nil
}

func validateStringField(ctx context.Context, field protoreflect.FieldDescriptor, value protoreflect.Value, path string, limit int) (err error) {
	if err = ctx.Err(); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}

	switch field.Kind() {
	case protoreflect.MessageKind:
		return validateStrings(ctx, value.Message(), path, limit)
	case protoreflect.StringKind:
		if len(value.String()) > limit {
			return fmt.Errorf("%s: %w: string exceeds %d bytes", path, ErrLimit, limit)
		}

	case protoreflect.BytesKind:
		if field.ContainingMessage().Name() == "AttributeProto" && len(value.Bytes()) > limit {
			return fmt.Errorf("%s: %w: attribute string exceeds %d bytes", path, ErrLimit, limit)
		}
	}
	return nil
}

func validateConstant(info ValueInfo, constant *Constant) (err error) {
	if info.DataType != tensor.Invalid && info.DataType != constant.DataType() {
		return fmt.Errorf("%w: declared type differs from initializer", ErrInvalidModel)
	}

	if !info.ShapeKnown {
		return nil
	}

	shape := constant.Shape()
	if len(info.Dimensions) != len(shape) {
		return fmt.Errorf("%w: declared rank differs from initializer", ErrInvalidModel)
	}

	for axis, dimension := range info.Dimensions {
		if dimension.Kind == FixedDimension && dimension.Value != shape[axis] {
			return fmt.Errorf("%w: declared fixed axis %d differs from initializer", ErrInvalidModel, axis)
		}
	}
	return nil
}

func compatibleInfo(left, right ValueInfo) (err error) {
	if left.DataType != tensor.Invalid && right.DataType != tensor.Invalid && left.DataType != right.DataType {
		return fmt.Errorf("%w: contradictory declared types", ErrInvalidModel)
	}

	if !left.ShapeKnown || !right.ShapeKnown {
		return nil
	}

	if len(left.Dimensions) != len(right.Dimensions) {
		return fmt.Errorf("%w: contradictory declared ranks", ErrInvalidModel)
	}

	for axis, dimension := range left.Dimensions {
		other := right.Dimensions[axis]
		if dimension.Kind == FixedDimension && other.Kind == FixedDimension && dimension.Value != other.Value {
			return fmt.Errorf("%w: contradictory fixed axis %d", ErrInvalidModel, axis)
		}
	}
	return nil
}
