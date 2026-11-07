package onnx

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newNumericalCall(step operators.Step, inputs []*tensor.Tensor, limits Limits) (out *numericalCall, err error) {
	var c numericalCall
	c.step, c.inputs = step, inputs
	for _, input := range inputs {
		if input != nil && input.DataType() != Float32 {
			return nil, fmt.Errorf("%w: numerical input must be float32", ErrInvalidModel)
		}
	}
	var right []int64
	left := inputs[0].Shape()
	if len(inputs) > 1 {
		right = inputs[1].Shape()
	}

	switch step.Operator {
	case "Add", "Sub", "Mul", "Div":
		c.shape, err = tensor.BroadcastShape(left, right)
	case "MatMul":
		c.shape, err = tensor.MatMulShape(left, right)
	case "Gemm":
		c.shape, err = tensor.GemmShape(left, right, step.Gemm)
		if err == nil && len(inputs) > 2 && inputs[2] != nil {
			err = tensor.ValidateBroadcastTo(inputs[2].Shape(), c.shape)
		}
	case "Relu", "Sigmoid", "Tanh":
		c.shape = left
	case "Softmax":
		c.shape = left
		_, err = tensor.NormalizeAxis(step.Axis, len(left))
	default:
		return nil, fmt.Errorf("%w: missing numerical handler", ErrUnsupported)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrInvalidModel, err)
	}

	if len(c.shape) > limits.MaxRank {
		return nil, fmt.Errorf("%w: numerical output rank", ErrLimit)
	}

	for _, size := range c.shape {
		if size > limits.MaxDimension {
			return nil, fmt.Errorf("%w: numerical output dimension", ErrLimit)
		}
	}
	if _, c.bytes, err = tensor.ValidateShape(c.shape, Float32); err != nil {
		return nil, fmt.Errorf("%w: numerical output: %w", ErrLimit, err)
	}

	if c.workspace, err = c.scratch(left, right); err != nil {
		return nil, err
	}

	return &c, nil
}

// numericalCall audits all copying v0.7.0 kernel paths before numerical allocation.
// Workspace conservatively includes every call-local buffer across all batches.
type numericalCall struct {
	step      operators.Step
	inputs    []*tensor.Tensor
	shape     []int64
	bytes     int64
	workspace int64
}

func (c *numericalCall) scratch(left, right []int64) (bytes int64, err error) {
	if c.bytes == 0 {
		return 0, nil
	}

	switch c.step.Operator {
	case "Relu":
		return checkedScratch(c.bytes, 6)
	case "Sigmoid", "Tanh":
		return checkedScratch(c.bytes, 3)
	case "Add", "Sub", "Mul":
		if slices.Equal(left, right) {
			return checkedScratch(c.bytes, 4)
		}

		return 0, nil
	case "Gemm":
		inner := left[1]
		if c.step.Gemm.TransposeA {
			inner = left[0]
		}

		if inner == 0 {
			return c.bytes, nil
		}

		parts := []int64{c.inputs[0].ByteSize(), c.inputs[1].ByteSize(), c.bytes, c.bytes}
		if c.step.Gemm.TransposeA && c.step.Gemm.TransposeB {
			parts = append(parts, c.inputs[1].ByteSize())
		}

		return sumScratch(parts...)
	case "MatMul":
		inner := left[len(left)-1]
		if inner == 0 {
			return 0, nil
		}

		rows, cols := int64(1), int64(1)
		if len(left) > 1 {
			rows = left[len(left)-2]
		}

		if len(right) > 1 {
			cols = right[len(right)-1]
		}

		// Input shapes and output bytes have already passed checked validation.
		// Divide output elements by each free dimension to avoid unchecked products.
		elements := c.bytes / 4
		var leftCopies, rightCopies, products int64
		if leftCopies, err = checkedScratch(elements/cols*4, inner); err != nil {
			return 0, err
		}

		if rightCopies, err = checkedScratch(elements/rows*4, inner); err != nil {
			return 0, err
		}

		if products, err = checkedScratch(c.bytes, 3); err != nil {
			return 0, err
		}

		return sumScratch(leftCopies, rightCopies, products)
	}
	return 0, nil
}

func (c *numericalCall) run(ctx context.Context) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if out, err = tensor.New(Float32, c.shape); err != nil {
		return nil, fmt.Errorf("allocate numerical destination: %w", err)
	}

	left := c.inputs[0]
	switch c.step.Operator {
	case "Add":
		err = tensor.AddInto(left, c.inputs[1], out)
	case "Sub":
		err = tensor.SubInto(left, c.inputs[1], out)
	case "Mul":
		err = tensor.MulInto(left, c.inputs[1], out)
	case "Div":
		err = tensor.DivInto(left, c.inputs[1], out)
	case "MatMul":
		err = tensor.MatMulInto(left, c.inputs[1], out)
	case "Gemm":
		var bias *tensor.Tensor
		if len(c.inputs) > 2 {
			bias = c.inputs[2]
		}

		err = tensor.GemmInto(left, c.inputs[1], bias, out, c.step.Gemm)
	case "Relu":
		err = tensor.ReLUInto(left, out)
	case "Sigmoid":
		err = tensor.SigmoidInto(left, out)
	case "Tanh":
		err = tensor.TanhInto(left, out)
	case "Softmax":
		err = tensor.SoftmaxInto(left, out, c.step.Axis)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: kernel: %w", ErrInvalidModel, err)
	}

	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if c.step.Operator == "Relu" && c.bytes != 0 {
		return c.patchRelu(ctx, out)
	}

	return out, nil
}

func (c *numericalCall) patchRelu(ctx context.Context, output *tensor.Tensor) (out *tensor.Tensor, err error) {
	var input, values []float32
	if input, err = c.inputs[0].Float32Values(); err != nil {
		return nil, err
	}

	if values, err = output.Float32Values(); err != nil {
		return nil, err
	}

	for index, value := range input {
		if index%1024 == 0 {
			if err = ctx.Err(); err != nil {
				return nil, err
			}
		}

		if math.IsNaN(float64(value)) {
			values[index] = value
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	return tensor.FromFloat32(c.shape, values)
}

func checkedScratch(bytes, count int64) (total int64, err error) {
	if bytes < 0 || count < 0 || count != 0 && bytes > math.MaxInt64/count {
		return 0, fmt.Errorf("%w: numerical workspace overflow", ErrLimit)
	}

	return bytes * count, nil
}

func sumScratch(parts ...int64) (total int64, err error) {
	for _, bytes := range parts {
		if bytes < 0 || bytes > math.MaxInt64-total {
			return 0, fmt.Errorf("%w: numerical workspace overflow", ErrLimit)
		}

		total += bytes
	}
	return total, nil
}
