package onnx

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newShapeCall(step operators.Step, inputs []*tensor.Tensor, limits Limits) (out *shapeCall, err error) {
	var c shapeCall
	c.step, c.inputs, c.limits = step, inputs, limits
	c.dataType = inputs[0].DataType()
	if step.Operator == "Shape" {
		c.dataType = Int64
		c.workspace = int64(inputs[0].Rank()) * 8
	}

	if len(inputs) > 1 && inputs[1] != nil && step.Operator != "Concat" {
		if inputs[1].DataType() != Int64 {
			return nil, fmt.Errorf("%w: %s input 1 must be int64", ErrInvalidModel, step.Operator)
		}

		if step.Operator != "Gather" {
			if inputs[1].Rank() != 1 {
				return nil, fmt.Errorf("%w: shape or axes input must have rank one", ErrInvalidModel)
			}

			count := inputs[1].Len()
			if count > int64(limits.MaxRank) || step.Operator == "Squeeze" && count > int64(inputs[0].Rank()) || step.Operator == "Unsqueeze" && count > int64(limits.MaxRank-inputs[0].Rank()) {
				return nil, fmt.Errorf("%w: shape or axes length exceeds applicable rank", ErrLimit)
			}

			c.workspace = inputs[1].ByteSize()
		}
	}

	return &c, nil
}

// shapeCall reserves accessor staging before planning, then the complete
// destination before allocation. Released transforms/indexing copy directly
// into destinations and allocate no additional numerical scratch.
type shapeCall struct {
	step        operators.Step
	inputs      []*tensor.Tensor
	limits      Limits
	dataType    DataType
	shape       []int64
	axes        []int64
	permutation []int64
	shapeValues []int64
	bytes       int64
	workspace   int64
}

func (c *shapeCall) plan(ctx context.Context) (err error) {
	if err = ctx.Err(); err != nil {
		return err
	}

	left := c.inputs[0].Shape()
	if c.workspace > 0 && c.step.Operator != "Shape" {
		if c.axes, err = c.inputs[1].Int64Values(); err != nil {
			return err
		}
	}

	switch c.step.Operator {
	case "Shape":
		start, end := operators.ShapeRange(c.step, len(left))
		c.shape, c.shapeValues = []int64{int64(end - start)}, left[start:end]
	case "Flatten":
		c.shape, err = tensor.FlattenShape(left, c.step.Axis)
	case "Transpose":
		c.permutation = c.step.Permutation
		if !c.step.PermutationSet {
			c.permutation = operators.ReversePermutation(len(left))
		}

		if err = operators.ValidatePermutation(c.permutation, len(left)); err != nil {
			return err
		}

		c.shape, err = tensor.TransposeShape(left, c.permutation)
	case "Reshape":
		c.shape, err = operators.ResolveReshapeMode(left, c.axes, c.step.AllowZero)
	case "Squeeze":
		if len(c.inputs) == 1 || c.inputs[1] == nil {
			for axis, dimension := range left {
				if dimension == 1 {
					c.axes = append(c.axes, int64(axis))
				}
			}
		}

		c.shape, err = tensor.SqueezeShape(left, c.axes)
	case "Unsqueeze":
		c.shape, err = tensor.UnsqueezeShape(left, c.axes)
	case "Gather":
		c.shape, err = tensor.GatherShape(left, c.inputs[1].Shape(), c.step.Axis)
	case "Concat":
		shapes := make([][]int64, len(c.inputs))
		for index, input := range c.inputs {
			if err = ctx.Err(); err != nil {
				return err
			}

			if input.DataType() != c.dataType {
				return fmt.Errorf("%w: Concat types differ", ErrInvalidModel)
			}

			shapes[index] = input.Shape()
		}
		c.shape, err = tensor.ConcatShape(shapes, c.step.Axis)
	}
	if err != nil {
		return fmt.Errorf("%w: shape: %w", ErrInvalidModel, err)
	}

	if len(c.shape) > c.limits.MaxRank {
		return fmt.Errorf("%w: shape output rank", ErrLimit)
	}

	for _, size := range c.shape {
		if size > c.limits.MaxDimension {
			return fmt.Errorf("%w: shape output dimension", ErrLimit)
		}
	}
	if _, c.bytes, err = tensor.ValidateShape(c.shape, c.dataType); err != nil {
		return fmt.Errorf("%w: shape output: %w", ErrLimit, err)
	}

	return ctx.Err()
}

func (c *shapeCall) run(ctx context.Context) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if c.step.Operator == "Shape" {
		if out, err = tensor.FromInt64(c.shape, c.shapeValues); err != nil {
			return nil, fmt.Errorf("construct Shape destination: %w", err)
		}

		return out, ctx.Err()
	}

	if out, err = tensor.New(c.dataType, c.shape); err != nil {
		return nil, fmt.Errorf("allocate shape destination: %w", err)
	}

	switch c.step.Operator {
	case "Flatten":
		err = tensor.FlattenInto(c.inputs[0], out, c.step.Axis)
	case "Transpose":
		err = tensor.TransposeInto(c.inputs[0], out, c.permutation)
	case "Reshape":
		err = tensor.ReshapeInto(c.inputs[0], out)
	case "Squeeze":
		err = tensor.SqueezeInto(c.inputs[0], out, c.axes)
	case "Unsqueeze":
		err = tensor.UnsqueezeInto(c.inputs[0], out, c.axes)
	case "Gather":
		err = tensor.GatherInto(c.inputs[0], c.inputs[1], out, c.step.Axis)
	case "Concat":
		err = tensor.ConcatInto(c.inputs, out, c.step.Axis)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: shape kernel: %w", ErrInvalidModel, err)
	}

	return out, ctx.Err()
}
