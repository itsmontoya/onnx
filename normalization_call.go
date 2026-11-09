package onnx

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newNormalizationCall(step operators.Step, inputs []*tensor.Tensor, limits Limits) (out *normalizationCall, err error) {
	var c normalizationCall
	c.step, c.inputs = step, inputs
	for _, input := range inputs {
		if input != nil && input.DataType() != Float32 {
			return nil, fmt.Errorf("%w: LayerNormalization requires float32", ErrInvalidModel)
		}
	}
	c.shape = inputs[0].Shape()
	var axis int
	if axis, err = tensor.NormalizeAxis(step.Axis, len(c.shape)); err != nil {
		return nil, fmt.Errorf("%w: LayerNormalization axis: %w", ErrInvalidModel, err)
	}

	for index := axis; index < len(c.shape); index++ {
		if c.shape[index] == 0 {
			return nil, fmt.Errorf("%w: LayerNormalization empty normalized dimension", ErrUnsupported)
		}

		c.axes = append(c.axes, int64(index))
	}
	for slot := 1; slot < len(inputs); slot++ {
		if inputs[slot] != nil {
			if err = tensor.ValidateBroadcastTo(inputs[slot].Shape(), c.shape); err != nil {
				return nil, fmt.Errorf("%w: LayerNormalization input %d: %w", ErrInvalidModel, slot, err)
			}
		}
	}
	if c.statistics, err = tensor.ReductionShape(c.shape, c.axes, true); err != nil {
		return nil, fmt.Errorf("%w: normalization statistics: %w", ErrInvalidModel, err)
	}

	var dataBytes, statisticBytes, dataScratch, statisticScratch int64
	if _, dataBytes, err = tensor.ValidateShape(c.shape, Float32); err != nil {
		return nil, fmt.Errorf("%w: normalization output: %w", ErrLimit, err)
	}

	if _, statisticBytes, err = tensor.ValidateShape(c.statistics, Float32); err != nil {
		return nil, fmt.Errorf("%w: normalization statistics: %w", ErrLimit, err)
	}

	if dataBytes > limits.MaxTensorBytes || statisticBytes > limits.MaxTensorBytes || limits.MaxTensorBytes < 4 {
		return nil, fmt.Errorf("%w: normalization tensor bytes", ErrLimit)
	}

	c.bytes = []int64{dataBytes, statisticBytes, statisticBytes}
	if dataScratch, err = checkedScratch(dataBytes, 20); err != nil {
		return nil, err
	}

	if statisticScratch, err = checkedScratch(statisticBytes, 16); err != nil {
		return nil, err
	}

	if c.workspace, err = sumScratch(dataScratch, statisticScratch, 8); err != nil {
		return nil, err
	}

	return &c, nil
}

// normalizationCall composes published NN-12 kernels; no input is mutated.
// Workspace sums retained temporaries and every copying arithmetic path, even
// though their individual scratch buffers do not all coexist.
type normalizationCall struct {
	step       operators.Step
	inputs     []*tensor.Tensor
	shape      []int64
	statistics []int64
	axes       []int64
	bytes      []int64
	workspace  int64
}

func (c *normalizationCall) run(ctx context.Context) (outputs []*tensor.Tensor, err error) {
	var buffers [10]*tensor.Tensor
	// Mean, D, DD, variance, variance+epsilon, standard deviation,
	// inverse standard deviation, normalized, scaled, Y use disjoint storage.
	shapes := [][]int64{c.statistics, c.shape, c.shape, c.statistics, c.statistics, c.statistics, c.statistics, c.shape, c.shape, c.shape}
	for index, shape := range shapes {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		if buffers[index], err = tensor.New(Float32, shape); err != nil {
			return nil, fmt.Errorf("allocate normalization buffer %d: %w", index, err)
		}
	}
	var epsilon, one *tensor.Tensor
	if epsilon, err = tensor.FromFloat32(nil, []float32{c.step.Epsilon}); err != nil {
		return nil, fmt.Errorf("allocate normalization epsilon: %w", err)
	}

	if one, err = tensor.FromFloat32(nil, []float32{1}); err != nil {
		return nil, fmt.Errorf("allocate normalization reciprocal numerator: %w", err)
	}

	kernels := []func() error{
		func() (err error) { return tensor.ReduceMeanInto(c.inputs[0], buffers[0], c.axes, true) },
		func() (err error) { return tensor.SubInto(c.inputs[0], buffers[0], buffers[1]) },
		func() (err error) { return tensor.MulInto(buffers[1], buffers[1], buffers[2]) },
		func() (err error) { return tensor.ReduceMeanInto(buffers[2], buffers[3], c.axes, true) },
		func() (err error) { return tensor.AddInto(buffers[3], epsilon, buffers[4]) },
		func() (err error) { return tensor.SqrtInto(buffers[4], buffers[5]) },
		func() (err error) { return tensor.DivInto(one, buffers[5], buffers[6]) },
		func() (err error) { return tensor.MulInto(buffers[1], buffers[6], buffers[7]) },
		func() (err error) { return tensor.MulInto(buffers[7], c.inputs[1], buffers[8]) },
	}
	for _, kernel := range kernels {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		if err = kernel(); err != nil {
			return nil, fmt.Errorf("%w: normalization kernel: %w", ErrInvalidModel, err)
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if len(c.inputs) > 2 && c.inputs[2] != nil {
		err = tensor.AddInto(buffers[8], c.inputs[2], buffers[9])
	} else {
		err = tensor.MulInto(buffers[8], one, buffers[9])
	}

	if err != nil {
		return nil, fmt.Errorf("%w: normalization affine kernel: %w", ErrInvalidModel, err)
	}

	if err = ctx.Err(); err != nil {
		return nil, err
	}

	return []*tensor.Tensor{buffers[9], buffers[0], buffers[6]}, nil
}
