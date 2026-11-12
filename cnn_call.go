package onnx

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newCNNCall(step operators.Step, inputs []*tensor.Tensor, limits Limits) (out *cnnCall, err error) {
	var c cnnCall
	c.step, c.inputs = step, inputs
	shapes := make([][]int64, len(inputs))
	for slot, input := range inputs {
		if input == nil {
			continue
		}

		if input.DataType() != Float32 {
			return nil, fmt.Errorf("%w: CNN input must be float32", ErrInvalidModel)
		}

		shapes[slot] = append([]int64{}, input.Shape()...)
	}
	if err = c.plan(shapes, limits); err != nil {
		return nil, err
	}

	return &c, nil
}

func newPreparedCNNCall(step operators.Step, shapes [][]int64, limits Limits) (out *cnnCall) {
	var c cnnCall
	c.step = step
	if c.plan(shapes, limits) != nil {
		return nil
	}

	return &c
}

// cnnCall uses the released direct Into loops: no numerical accessor copies,
// packed weights, normalization caches, or scratch buffers are allocated.
type cnnCall struct {
	step   operators.Step
	inputs []*tensor.Tensor
	shape  []int64
	conv   tensor.Conv2DConfig
	pool   tensor.Pool2DConfig
	bytes  int64
}

func (c *cnnCall) plan(shapes [][]int64, limits Limits) (err error) {
	if c.shape, c.conv, c.pool, err = c.step.CNN.Shape(c.step.Operator, shapes); err != nil {
		return fmt.Errorf("%w: CNN shape: %w", ErrInvalidModel, err)
	}

	if len(c.shape) > limits.MaxRank {
		return fmt.Errorf("%w: CNN output rank", ErrLimit)
	}

	for _, size := range c.shape {
		if size > limits.MaxDimension {
			return fmt.Errorf("%w: CNN output dimension", ErrLimit)
		}
	}
	if _, c.bytes, err = tensor.ValidateShape(c.shape, Float32); err != nil {
		return fmt.Errorf("%w: CNN output: %w", ErrLimit, err)
	}

	return nil
}

func (c *cnnCall) run(ctx context.Context, destination *tensor.Tensor) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	out = destination
	if out == nil {
		if out, err = tensor.New(Float32, c.shape); err != nil {
			return nil, fmt.Errorf("allocate CNN destination: %w", err)
		}
	}

	switch c.step.Operator {
	case "Conv":
		var bias *tensor.Tensor
		if len(c.inputs) > 2 {
			bias = c.inputs[2]
		}

		err = tensor.Conv2DInto(c.inputs[0], c.inputs[1], bias, out, c.conv)
	case "MaxPool":
		err = tensor.MaxPool2DInto(c.inputs[0], out, c.pool)
	case "AveragePool":
		err = tensor.AveragePool2DInto(c.inputs[0], out, c.pool)
	case "GlobalAveragePool":
		err = tensor.GlobalAveragePoolInto(c.inputs[0], out)
	case "BatchNormalization":
		err = tensor.BatchNormalizationInto(c.inputs[0], c.inputs[1], c.inputs[2], c.inputs[3], c.inputs[4], out, c.step.CNN.Epsilon)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: CNN kernel: %w", ErrInvalidModel, err)
	}

	return out, ctx.Err()
}
