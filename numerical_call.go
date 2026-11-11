package onnx

import (
	"context"
	"errors"
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

// numericalCall audits the pinned kernel paths before numerical allocation.
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
	case "Gemm", "MatMul":
		return productWorkspace(c, left, right)
	}
	return 0, nil
}

func (c *numericalCall) run(ctx context.Context, destination *tensor.Tensor) (out *tensor.Tensor, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	out = destination
	if out == nil {
		if out, err = tensor.New(Float32, c.shape); err != nil {
			return nil, fmt.Errorf("allocate numerical destination: %w", err)
		}
	}

	if c.step.Operator == "Relu" && c.bytes != 0 {
		return c.runPreservingRelu(ctx, out)
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

	return out, nil
}

// runPreservingRelu retains the legacy private destination on NaNs and
// cancellation without snapshotting finite inputs. Scratch charging is unchanged.
func (c *numericalCall) runPreservingRelu(ctx context.Context, destination *tensor.Tensor) (out *tensor.Tensor, err error) {
	var (
		hasNaN     bool
		restoreErr error
	)

	if hasNaN, err = tensor.ReLUPreservingNaNInto(ctx, c.inputs[0], destination); err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			return nil, fmt.Errorf("%w: kernel: %w", ErrInvalidModel, err)
		}

		// Cancellation previously followed a complete legacy kernel. Restore
		// that state even when the context-aware traversal stopped early.
		if restoreErr = tensor.ReLUInto(c.inputs[0], destination); restoreErr != nil {
			return nil, fmt.Errorf("%w: restore ReLU destination: %w", ErrInvalidModel, restoreErr)
		}

		return nil, err
	}

	if !hasNaN {
		return destination, nil
	}

	// NaNs still return independent storage and leave the supplied destination
	// with the legacy NaN-to-zero result, including if cloning fails.
	out, err = destination.Clone()
	if restoreErr = tensor.ReLUInto(c.inputs[0], destination); restoreErr != nil {
		return nil, fmt.Errorf("%w: restore ReLU destination: %w", ErrInvalidModel, restoreErr)
	}

	return out, err
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
