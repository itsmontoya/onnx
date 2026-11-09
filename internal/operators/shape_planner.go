package operators

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

func newShapePlanner(ctx context.Context, step *Step, inputs []graph.ValueInfo, values []int64, known bool) (out *shapePlanner) {
	var p shapePlanner
	p.ctx, p.step, p.inputs, p.values, p.known = ctx, step, inputs, values, known
	return &p
}

// shapePlanner infers metadata only. Known constants are inspected without
// evaluating nodes or constructing numerical output tensors at load time.
type shapePlanner struct {
	ctx    context.Context
	step   *Step
	inputs []graph.ValueInfo
	values []int64
	known  bool
}

func (p *shapePlanner) infer(node graph.Node) (info graph.ValueInfo, err error) {
	if err = p.ctx.Err(); err != nil {
		return info, err
	}

	p.step.Axis = 0
	if node.Operator == "Flatten" {
		p.step.Axis = 1
	}

	for _, attribute := range node.Attributes {
		switch attribute.Name {
		case "axis":
			p.step.Axis = attribute.Int
		case "allowzero":
			if attribute.Int != 0 && attribute.Int != 1 {
				return info, fmt.Errorf("%w: allowzero must be 0 or 1", graph.ErrInvalidModel)
			}

			p.step.AllowZero = attribute.Int == 1
		case "start":
			p.step.ShapeStart = attribute.Int
		case "end":
			p.step.ShapeEnd, p.step.ShapeEndSet = attribute.Int, true
		case "perm":
			p.step.Permutation, p.step.PermutationSet = slices.Clone(attribute.Ints), true
		}
	}
	info.DataType = p.inputs[0].DataType
	if node.Operator == "Shape" {
		info.DataType, info.ShapeKnown = tensor.Int64, true
		info.Dimensions = []graph.Dimension{{}}
		if p.inputs[0].ShapeKnown {
			start, end := ShapeRange(*p.step, len(p.inputs[0].Dimensions))
			info.Dimensions[0] = fixed(int64(end - start))
		}

		return info, nil
	}

	if len(p.inputs) > 1 && node.Operator != "Concat" && p.step.Inputs[1] >= 0 {
		index := p.inputs[1]
		if index.DataType != tensor.Invalid && index.DataType != tensor.Int64 {
			return info, fmt.Errorf("%w: %s input 1 must be int64", graph.ErrUnsupported, node.Operator)
		}

		if node.Operator != "Gather" && index.ShapeKnown && len(index.Dimensions) != 1 {
			return info, fmt.Errorf("%w: %s input 1 must have rank one", graph.ErrInvalidModel, node.Operator)
		}
	}

	switch node.Operator {
	case "Flatten":
		return p.flatten(info)
	case "Transpose":
		return p.transpose(info)
	case "Reshape":
		return p.reshape(info)
	case "Squeeze", "Unsqueeze":
		return p.squeeze(info)
	case "Gather":
		return p.gather(info)
	case "Concat":
		return p.concat(info)
	}
	return info, fmt.Errorf("%w: missing shape inference", graph.ErrUnsupported)
}

func (p *shapePlanner) flatten(info graph.ValueInfo) (out graph.ValueInfo, err error) {
	info.ShapeKnown, info.Dimensions = true, make([]graph.Dimension, 2)
	if !p.inputs[0].ShapeKnown {
		return info, nil
	}

	rank, axis := int64(len(p.inputs[0].Dimensions)), p.step.Axis
	if axis < -rank || axis > rank {
		return info, fmt.Errorf("%w: Flatten split axis %d rank %d", graph.ErrInvalidModel, axis, rank)
	}

	if axis < 0 {
		axis += rank
	}

	if info.Dimensions[0], err = dimensionProduct(p.inputs[0].Dimensions[:int(axis)]); err != nil {
		return info, err
	}

	info.Dimensions[1], err = dimensionProduct(p.inputs[0].Dimensions[int(axis):])
	return info, err
}

func (p *shapePlanner) transpose(info graph.ValueInfo) (out graph.ValueInfo, err error) {
	permutation := p.step.Permutation
	// Even without a known input rank an explicit permutation must be complete
	// and unique for the rank implied by its own length.
	if p.step.PermutationSet {
		if err = ValidatePermutation(permutation, len(permutation)); err != nil {
			return info, err
		}
	}

	if !p.inputs[0].ShapeKnown {
		if p.step.PermutationSet {
			info.ShapeKnown, info.Dimensions = true, make([]graph.Dimension, len(permutation))
		}

		return info, nil
	}

	rank := len(p.inputs[0].Dimensions)
	if !p.step.PermutationSet {
		permutation = ReversePermutation(rank)
	}

	if err = ValidatePermutation(permutation, rank); err != nil {
		return info, err
	}

	info.ShapeKnown, info.Dimensions = true, make([]graph.Dimension, rank)
	for axis, source := range permutation {
		info.Dimensions[axis] = p.inputs[0].Dimensions[int(source)]
	}
	return info, nil
}

func (p *shapePlanner) reshape(info graph.ValueInfo) (out graph.ValueInfo, err error) {
	if !p.known {
		shape := p.inputs[1]
		if shape.ShapeKnown && len(shape.Dimensions) == 1 && shape.Dimensions[0].Kind == graph.FixedDimension {
			info.ShapeKnown, info.Dimensions = true, make([]graph.Dimension, int(shape.Dimensions[0].Value))
		}

		return info, nil
	}

	info.ShapeKnown, info.Dimensions = true, make([]graph.Dimension, len(p.values))
	inferred := -1
	for axis, value := range p.values {
		switch {
		case value < -1:
			return info, fmt.Errorf("%w: Reshape dimension %d", graph.ErrInvalidModel, value)
		case value == -1:
			if inferred >= 0 {
				return info, fmt.Errorf("%w: multiple inferred Reshape dimensions", graph.ErrInvalidModel)
			}

			inferred = axis
		case value == 0:
			if p.step.AllowZero {
				info.Dimensions[axis] = fixed(0)
				continue
			}

			if p.inputs[0].ShapeKnown {
				if axis >= len(p.inputs[0].Dimensions) {
					return info, fmt.Errorf("%w: Reshape zero axis exceeds input rank", graph.ErrInvalidModel)
				}

				info.Dimensions[axis] = p.inputs[0].Dimensions[axis]
			}
		default:
			info.Dimensions[axis] = fixed(value)
		}
	}
	if p.step.AllowZero && inferred >= 0 {
		for _, value := range p.values {
			if value == 0 {
				return info, fmt.Errorf("%w: allowzero=1 forbids zero with an inferred dimension", graph.ErrInvalidModel)
			}
		}
	}

	productDimensions := slices.Clone(info.Dimensions)
	if inferred >= 0 {
		productDimensions[inferred] = fixed(1)
	}

	var product graph.Dimension
	if product, err = dimensionProduct(productDimensions); err != nil {
		return info, err
	}

	if inferred >= 0 && product.Kind == graph.FixedDimension && product.Value == 0 {
		return info, fmt.Errorf("%w: ambiguous zero-product Reshape inference", graph.ErrInvalidModel)
	}

	if !p.inputs[0].ShapeKnown {
		return info, nil
	}

	var count graph.Dimension
	if count, err = dimensionProduct(p.inputs[0].Dimensions); err != nil {
		return info, err
	}

	if count.Kind != graph.FixedDimension || product.Kind != graph.FixedDimension {
		return info, nil
	}

	if inferred >= 0 {
		if count.Value%product.Value != 0 {
			return info, fmt.Errorf("%w: Reshape inferred count is not integral", graph.ErrInvalidModel)
		}

		info.Dimensions[inferred] = fixed(count.Value / product.Value)
	} else if count.Value != product.Value {
		return info, fmt.Errorf("%w: Reshape element counts differ", graph.ErrInvalidModel)
	}

	return info, nil
}

func (p *shapePlanner) squeeze(info graph.ValueInfo) (out graph.ValueInfo, err error) {
	omitted := len(p.step.Inputs) == 1 || p.step.Inputs[1] < 0
	if !p.inputs[0].ShapeKnown {
		if p.known {
			if err = validateRawAxes(p.values); err != nil {
				return info, err
			}
		}

		return info, nil
	}

	dimensions := p.inputs[0].Dimensions
	if omitted {
		for _, dimension := range dimensions {
			if dimension.Kind != graph.FixedDimension {
				return info, nil
			}

			if dimension.Value != 1 {
				info.Dimensions = append(info.Dimensions, dimension)
			}
		}
		info.ShapeKnown = true
		return info, nil
	}

	if !p.known {
		return info, nil
	}

	rank := len(dimensions)
	if p.step.Operator == "Unsqueeze" {
		rank += len(p.values)
	}

	var axes []int
	if axes, err = NormalizeAxes(p.values, rank); err != nil {
		return info, err
	}

	selected := make([]bool, rank)
	for _, axis := range axes {
		selected[axis] = true
		if p.step.Operator == "Squeeze" && dimensions[axis].Kind == graph.FixedDimension && dimensions[axis].Value != 1 {
			return info, fmt.Errorf("%w: Squeeze selected nonunit axis %d", graph.ErrInvalidModel, axis)
		}
	}
	inputAxis := 0
	for axis := 0; axis < rank; axis++ {
		if p.step.Operator == "Unsqueeze" && selected[axis] {
			info.Dimensions = append(info.Dimensions, fixed(1))
			continue
		}

		if p.step.Operator == "Unsqueeze" || !selected[axis] {
			info.Dimensions = append(info.Dimensions, dimensions[inputAxis])
		}

		inputAxis++
	}
	info.ShapeKnown = true
	return info, nil
}

func (p *shapePlanner) gather(info graph.ValueInfo) (out graph.ValueInfo, err error) {
	if !p.inputs[0].ShapeKnown {
		return info, nil
	}

	var axis int
	if axis, err = tensor.NormalizeAxis(p.step.Axis, len(p.inputs[0].Dimensions)); err != nil {
		return info, fmt.Errorf("%w: Gather axis: %w", graph.ErrInvalidModel, err)
	}

	dimension := p.inputs[0].Dimensions[axis]
	if p.known && dimension.Kind == graph.FixedDimension {
		for position, index := range p.values {
			if position%1024 == 0 {
				if err = p.ctx.Err(); err != nil {
					return info, err
				}
			}

			if index < -dimension.Value || index >= dimension.Value {
				return info, fmt.Errorf("%w: Gather index %d dimension %d", graph.ErrInvalidModel, index, dimension.Value)
			}
		}
	}

	if !p.inputs[1].ShapeKnown {
		return info, nil
	}

	info.ShapeKnown = true
	info.Dimensions = append(info.Dimensions, p.inputs[0].Dimensions[:axis]...)
	info.Dimensions = append(info.Dimensions, p.inputs[1].Dimensions...)
	info.Dimensions = append(info.Dimensions, p.inputs[0].Dimensions[axis+1:]...)
	return info, nil
}

func (p *shapePlanner) concat(info graph.ValueInfo) (out graph.ValueInfo, err error) {
	var reference *graph.ValueInfo
	for index := range p.inputs {
		if err = p.ctx.Err(); err != nil {
			return info, err
		}

		input := &p.inputs[index]
		if input.DataType != tensor.Invalid {
			if info.DataType != tensor.Invalid && info.DataType != input.DataType {
				return info, fmt.Errorf("%w: Concat input types differ", graph.ErrInvalidModel)
			}

			info.DataType = input.DataType
		}

		if input.ShapeKnown && reference == nil {
			reference = input
		}
	}
	if reference == nil {
		return info, nil
	}

	var axis int
	if axis, err = tensor.NormalizeAxis(p.step.Axis, len(reference.Dimensions)); err != nil {
		return info, fmt.Errorf("%w: Concat axis: %w", graph.ErrInvalidModel, err)
	}

	info.ShapeKnown, info.Dimensions = true, slices.Clone(reference.Dimensions)
	info.Dimensions[axis] = fixed(0)
	for _, input := range p.inputs {
		if err = p.ctx.Err(); err != nil {
			return info, err
		}

		if !input.ShapeKnown {
			info.Dimensions[axis] = graph.Dimension{}
			continue
		}

		if len(input.Dimensions) != len(info.Dimensions) {
			return info, fmt.Errorf("%w: Concat ranks differ", graph.ErrInvalidModel)
		}

		for position, dimension := range input.Dimensions {
			current := info.Dimensions[position]
			if position != axis {
				if current.Kind == graph.FixedDimension && dimension.Kind == graph.FixedDimension && current.Value != dimension.Value {
					return info, fmt.Errorf("%w: Concat non-axis dimensions differ", graph.ErrInvalidModel)
				}

				if dimension.Kind == graph.FixedDimension {
					info.Dimensions[position] = dimension
				}

				continue
			}

			if current.Kind != graph.FixedDimension || dimension.Kind != graph.FixedDimension {
				info.Dimensions[position] = graph.Dimension{}
				continue
			}

			if dimension.Value > math.MaxInt64-current.Value {
				return info, fmt.Errorf("%w: Concat axis sum overflow", graph.ErrInvalidModel)
			}

			info.Dimensions[position] = fixed(current.Value + dimension.Value)
		}
	}
	return info, nil
}
