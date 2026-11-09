package operators

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/capability"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/profile"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

func newCompiler(ctx context.Context, source *graph.Graph, registry *Registry, limits Limits, destination *Plan) (out *compiler, err error) {
	var c compiler
	if ctx == nil || source == nil || registry == nil {
		return nil, wrapError(fmt.Errorf("%w: context, graph or registry required", runtimeerror.ErrInvalidArgument))
	}

	if c.limits, err = limits.normalized(); err != nil {
		return nil, wrapError(err)
	}

	c.ctx = ctx
	c.registry = registry
	c.plan = destination
	c.loadBytes, c.plan.initializerBytes = source.Usage()
	if err = c.reserve(0); err != nil {
		return nil, wrapError(err)
	}

	c.plan.indices = make(map[string]int)
	c.constants = make(map[int]*graph.Constant)
	return &c, nil
}

type compiler struct {
	ctx       context.Context
	limits    Limits
	registry  *Registry
	plan      *Plan
	loadBytes int64
	constants map[int]*graph.Constant
}

func (c *compiler) compile(source *graph.Graph) (err error) {
	loadBytes, _ := source.Usage()
	// Each accessor may deep-copy payloads. Reserve a whole-graph upper bound
	// for each of the five snapshots before requesting any of them.
	for index := 0; index < 5; index++ {
		if err = c.reserve(loadBytes); err != nil {
			return wrapError(err)
		}
	}

	c.plan.metadata = source.Metadata()
	if !profile.Accepts(c.plan.metadata.IRVersion, "", c.plan.metadata.Opset) {
		return wrapError(fmt.Errorf("%w: graph must be a successful tested-profile build", runtimeerror.ErrInvalidArgument))
	}

	c.plan.inputs = source.Inputs()
	c.plan.outputs = source.Outputs()
	values := source.Values()
	nodes := source.Nodes()
	for _, value := range values {
		if err = c.addValue(value); err != nil {
			diagnostic := wrapError(err)
			diagnostic.Value = value.Name
			return diagnostic
		}
	}

	for _, node := range nodes {
		if err = c.node(node); err != nil {
			return c.nodeError(node, err)
		}
	}

	if err = c.ctx.Err(); err != nil {
		return wrapError(err)
	}

	return nil
}

func (c *compiler) reserve(bytes int64) (err error) {
	if err = c.ctx.Err(); err != nil {
		return fmt.Errorf("reserve compilation payload: %w", err)
	}

	if bytes < 0 || c.loadBytes > c.limits.MaxLoadBytes || bytes > c.limits.MaxLoadBytes-c.loadBytes {
		return fmt.Errorf("%w: compilation snapshot payload", graph.ErrLimit)
	}

	c.loadBytes += bytes
	return nil
}

func (c *compiler) addValue(source graph.Value) (err error) {
	var value Value
	value.Name = source.Name
	value.Producer = source.Producer
	value.Input = source.Input
	value.Default = source.Default
	value.Declarations = source.Declarations
	value.Info.Name = source.Name
	for _, declaration := range source.Declarations {
		if value.Info, err = c.merge(value.Info, declaration); err != nil {
			return err
		}
	}

	if source.Default != nil && source.Default.ByteSize() > c.limits.MaxTensorBytes {
		return fmt.Errorf("%w: default tensor bytes %d", graph.ErrLimit, source.Default.ByteSize())
	}

	if source.Default != nil {
		var info graph.ValueInfo
		if info, err = c.constantInfo(source.Name, source.Default); err != nil {
			return err
		}

		if err = c.checkInfo(info); err != nil {
			return err
		}

		if !source.Input {
			if value.Info, err = c.merge(info, value.Info); err != nil {
				return err
			}
		}
	}

	value.Deferred = !concrete(value.Info)
	if err = c.checkInfo(value.Info); err != nil {
		return err
	}

	c.plan.indices[value.Name] = len(c.plan.values)
	c.plan.values = append(c.plan.values, value)
	if source.Default != nil && !source.Input {
		c.constants[len(c.plan.values)-1] = source.Default
	}

	return nil
}

func (c *compiler) node(node graph.Node) (err error) {
	if err = c.ctx.Err(); err != nil {
		return err
	}

	var record capability.Record
	if record, err = c.registry.Resolve(node.Domain, node.Operator, c.plan.metadata.Opset); err != nil {
		return err
	}

	if record.Status != "implemented" {
		return fmt.Errorf("%w: %s-%d is planned for %s", graph.ErrUnsupported, node.Operator, record.SchemaVersion, record.Task)
	}

	if err = validateNode(node, record); err != nil {
		return err
	}

	var step Step
	step.Index, step.Name = node.Index, node.Name
	step.Domain, step.Operator = node.Domain, node.Operator
	step.Opset, step.SchemaVersion = c.plan.metadata.Opset, record.SchemaVersion
	step.DocString, step.Metadata = node.DocString, node.Metadata
	for _, name := range node.Inputs {
		index := c.plan.indices[name]
		if name == "" {
			index = -1
		} else if !acceptsType(record.DataTypes, c.plan.values[index].Info.DataType) {
			return fmt.Errorf("%w: input value %q type %v", graph.ErrUnsupported, name, c.plan.values[index].Info.DataType)
		}

		step.Inputs = append(step.Inputs, index)
	}
	for _, name := range node.Outputs {
		index := c.plan.indices[name]
		if name == "" {
			index = -1
		}

		step.Outputs = append(step.Outputs, index)
	}

	var inferred graph.ValueInfo
	if inferred, step.Constant, err = c.infer(node, &step); err != nil {
		return err
	}

	if !acceptsType(record.DataTypes, inferred.DataType) {
		return fmt.Errorf("%w: output type %v", graph.ErrUnsupported, inferred.DataType)
	}

	for slot, index := range step.Outputs {
		if index < 0 {
			continue
		}

		outputInfo := inferred
		if step.Operator == "LayerNormalization" && slot > 0 {
			if err = c.reserve(int64(len(inferred.Dimensions)) * 64); err != nil {
				return err
			}

			outputInfo = normalizationStatistics(inferred, step.Axis)
		}

		value := &c.plan.values[index]
		outputInfo.Name = value.Name
		if value.Info, err = c.merge(outputInfo, value.Info); err != nil {
			return fmt.Errorf("output value %q: %w", value.Name, err)
		}

		value.Deferred = !concrete(value.Info)
		if err = c.checkInfo(value.Info); err != nil {
			return fmt.Errorf("output value %q: %w", value.Name, err)
		}
	}
	if step.Constant != nil {
		c.constants[step.Outputs[0]] = step.Constant
	} else if step.Operator == "Identity" {
		c.constants[step.Outputs[0]] = c.constants[step.Inputs[0]]
	}

	c.plan.steps = append(c.plan.steps, step)
	return nil
}

func (c *compiler) infer(node graph.Node, step *Step) (info graph.ValueInfo, constant *graph.Constant, err error) {
	if (len(step.Outputs) != 1 && node.Operator != "MaxPool" && node.Operator != "LayerNormalization") || step.Outputs[0] < 0 {
		return info, nil, fmt.Errorf("%w: planner requires one named output", graph.ErrUnsupported)
	}

	if node.Operator == "Identity" {
		if len(step.Inputs) != 1 || step.Inputs[0] < 0 || len(node.Attributes) != 0 {
			return info, nil, fmt.Errorf("%w: Identity requires one input and no attributes", graph.ErrUnsupported)
		}

		input := c.plan.values[step.Inputs[0]].Info
		info.DataType, info.ShapeKnown, info.Dimensions = input.DataType, input.ShapeKnown, input.Dimensions
		return info, nil, nil
	}

	if node.Operator != "Constant" {
		if err = c.reserve(fixedStepBytes); err != nil {
			return info, nil, err
		}

		for _, index := range step.Inputs {
			if index < 0 {
				continue
			}

			for range c.plan.values[index].Info.Dimensions {
				if err = c.reserve(64); err != nil {
					return info, nil, err
				}
			}
		}

		inputs := make([]graph.ValueInfo, len(step.Inputs))
		for slot, index := range step.Inputs {
			if index >= 0 {
				inputs[slot] = c.plan.values[index].Info
			}
		}
		if node.Operator == "LayerNormalization" {
			info, err = inferNormalization(node, step, inputs)
		} else if IsCNNOperator(node.Operator) {
			info, err = inferCNN(node, step, inputs)
		} else if IsShapeOperator(node.Operator) {
			info, err = c.inferShape(node, step, inputs)
		} else {
			info, err = inferNumerical(node, step, inputs)
		}

		return info, nil, err
	}

	if len(step.Inputs) != 0 || len(node.Attributes) != 1 || node.Attributes[0].Tensor == nil {
		return info, nil, fmt.Errorf("%w: Constant requires exactly one supported dense tensor/scalar/list attribute", graph.ErrInvalidModel)
	}

	constant = node.Attributes[0].Tensor
	info, err = c.constantInfo(node.Outputs[0], constant)
	return info, constant, err
}

func (c *compiler) inferShape(node graph.Node, step *Step, inputs []graph.ValueInfo) (info graph.ValueInfo, err error) {
	for _, attribute := range node.Attributes {
		if attribute.Name == "perm" {
			if len(attribute.Ints) > c.limits.MaxRank {
				return info, fmt.Errorf("%w: permutation rank", graph.ErrLimit)
			}

			if err = c.reserve(int64(len(attribute.Ints)) * 8); err != nil {
				return info, err
			}
		}
	}
	if len(inputs) > 1 && step.Inputs[1] >= 0 && node.Operator != "Gather" && node.Operator != "Concat" && inputs[1].ShapeKnown && len(inputs[1].Dimensions) == 1 && inputs[1].Dimensions[0].Kind == graph.FixedDimension {
		count := inputs[1].Dimensions[0].Value
		// Shape/axis vector lengths can exceed the data rank. Account
		// for output dimension records and local metadata copies too.
		if count > c.limits.MaxLoadBytes/128 {
			return info, fmt.Errorf("%w: shape metadata payload", graph.ErrLimit)
		}

		if err = c.reserve(count * 128); err != nil {
			return info, err
		}

		if count > int64(c.limits.MaxRank) || node.Operator == "Unsqueeze" && inputs[0].ShapeKnown && count > int64(c.limits.MaxRank-len(inputs[0].Dimensions)) {
			return info, fmt.Errorf("%w: shape or axes length exceeds rank limit", graph.ErrLimit)
		}

		if node.Operator == "Squeeze" && inputs[0].ShapeKnown && count > int64(len(inputs[0].Dimensions)) {
			return info, fmt.Errorf("%w: Squeeze axes exceed input rank", graph.ErrInvalidModel)
		}
	}

	var (
		values []int64
		known  bool
	)

	if len(step.Inputs) > 1 && step.Inputs[1] >= 0 && node.Operator != "Concat" {
		constantInput := c.constants[step.Inputs[1]]
		if constantInput != nil && constantInput.DataType() == tensor.Int64 {
			if err = c.reserve(constantInput.ByteSize()); err != nil {
				return info, err
			}

			if values, err = constantInput.Int64Values(); err != nil {
				return info, err
			}

			known = true
		}
	}

	planner := newShapePlanner(c.ctx, step, inputs, values, known)
	return planner.infer(node)
}

func (c *compiler) constantInfo(name string, constant *graph.Constant) (info graph.ValueInfo, err error) {
	// Reserve copied sizes and dimension records before requesting Shape.
	if err = c.reserve(int64(constant.Rank()) * 16); err != nil {
		return info, err
	}

	shape := constant.Shape()
	info.Name, info.DataType, info.ShapeKnown = name, constant.DataType(), true
	info.Dimensions = make([]graph.Dimension, len(shape))
	for axis, size := range shape {
		info.Dimensions[axis] = graph.Dimension{Kind: graph.FixedDimension, Value: size}
	}
	return info, nil
}

func (c *compiler) merge(inferred, declared graph.ValueInfo) (info graph.ValueInfo, err error) {
	info = inferred
	if info.Name == "" {
		info.Name = declared.Name
	}

	if info.DataType == tensor.Invalid {
		info.DataType = declared.DataType
	} else if declared.DataType != tensor.Invalid && info.DataType != declared.DataType {
		return info, fmt.Errorf("%w: inferred type %v contradicts declared %v", graph.ErrInvalidModel, info.DataType, declared.DataType)
	}

	if !declared.ShapeKnown {
		return info, nil
	}

	if !info.ShapeKnown {
		info.ShapeKnown, info.Dimensions = true, declared.Dimensions
		return info, nil
	}

	if len(info.Dimensions) != len(declared.Dimensions) {
		return info, fmt.Errorf("%w: inferred rank %d contradicts declared %d", graph.ErrInvalidModel, len(info.Dimensions), len(declared.Dimensions))
	}

	if err = c.reserve(int64(len(info.Dimensions)) * 8); err != nil {
		return info, err
	}

	info.Dimensions = slices.Clone(info.Dimensions)
	for axis, dimension := range declared.Dimensions {
		current := info.Dimensions[axis]
		if current.Kind == graph.FixedDimension && dimension.Kind == graph.FixedDimension && current.Value != dimension.Value {
			return info, fmt.Errorf("%w: inferred axis %d size %d contradicts declared %d", graph.ErrInvalidModel, axis, current.Value, dimension.Value)
		}

		if dimension.Kind == graph.FixedDimension || current.Kind == graph.UnknownDimension {
			info.Dimensions[axis] = dimension
		}
	}
	return info, nil
}

func (c *compiler) nodeError(node graph.Node, cause error) (err error) {
	diagnostic := wrapError(cause)
	diagnostic.NodeIndex, diagnostic.NodeName = node.Index, node.Name
	diagnostic.Domain, diagnostic.Operator, diagnostic.Opset = node.Domain, node.Operator, c.plan.metadata.Opset
	for _, name := range node.Inputs {
		var shape []int64
		if index, exists := c.plan.indices[name]; exists {
			info := c.plan.values[index].Info
			// Error reporting must also respect the quota. Missing diagnostic
			// shapes never replace or discard the original failure cause.
			if concrete(info) && c.reserve(int64(len(info.Dimensions))*8) == nil {
				shape = make([]int64, len(info.Dimensions))
				for axis, dimension := range info.Dimensions {
					shape[axis] = dimension.Value
				}
			}
		}

		diagnostic.Shapes = append(diagnostic.Shapes, shape)
	}
	return diagnostic
}

func (c *compiler) checkInfo(info graph.ValueInfo) (err error) {
	if !info.ShapeKnown {
		return nil
	}

	if len(info.Dimensions) > c.limits.MaxRank {
		return fmt.Errorf("%w: planned rank %d", graph.ErrLimit, len(info.Dimensions))
	}

	for axis, dimension := range info.Dimensions {
		if dimension.Kind == graph.FixedDimension && dimension.Value > c.limits.MaxDimension {
			return fmt.Errorf("%w: planned axis %d size %d", graph.ErrLimit, axis, dimension.Value)
		}
	}

	if !concrete(info) || info.DataType == tensor.Invalid {
		return nil
	}

	if err = c.reserve(int64(len(info.Dimensions)) * 8); err != nil {
		return err
	}

	shape := make([]int64, len(info.Dimensions))
	for axis, dimension := range info.Dimensions {
		shape[axis] = dimension.Value
	}

	var bytes int64
	if _, bytes, err = tensor.ValidateShape(shape, info.DataType); err != nil {
		return fmt.Errorf("%w: planned shape: %w", graph.ErrLimit, err)
	}

	if bytes > c.limits.MaxTensorBytes {
		return fmt.Errorf("%w: planned tensor bytes %d", graph.ErrLimit, bytes)
	}

	return nil
}

func concrete(info graph.ValueInfo) (known bool) {
	if !info.ShapeKnown {
		return false
	}

	for _, dimension := range info.Dimensions {
		if dimension.Kind != graph.FixedDimension {
			return false
		}
	}
	return true
}

func acceptsType(names []string, dataType tensor.DataType) (accepted bool) {
	if dataType == tensor.Invalid {
		return true
	}

	name := "int64"
	if dataType == tensor.Float32 {
		name = "float32"
	} else if dataType != tensor.Int64 {
		return false
	}

	return slices.Contains(names, name)
}

func wrapError(cause error) (out *runtimeerror.Error) {
	var diagnostic runtimeerror.Error
	diagnostic.NodeIndex = -1
	diagnostic.Cause = cause
	for _, kind := range []error{runtimeerror.ErrInvalidArgument, graph.ErrInvalidModel, graph.ErrUnsupported, graph.ErrLimit} {
		if errors.Is(cause, kind) {
			diagnostic.Kind = kind
			break
		}
	}
	return &diagnostic
}
