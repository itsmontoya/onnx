package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newMetalPlan(ctx context.Context, values []operators.Value, steps []operators.Step, options ExecutionOptions) (out *metalPlan, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	var (
		p      metalPlan
		device *gpu.Device
	)

	if device, err = gpu.OpenDefault(); err != nil {
		if errors.Is(err, gpu.ErrUnavailable) {
			return nil, fmt.Errorf("%w: open Metal device: %w", ErrUnavailable, err)
		}

		return nil, fmt.Errorf("open Metal device: %w", err)
	}

	p.device = device
	p.options = options
	p.weights = make(map[int]*gpu.Tensor)
	p.stats = GPUStats{
		Active:          true,
		WeightPrecision: options.WeightPrecision,
	}

	if err = p.uploadConstants(ctx, values, steps); err != nil {
		_ = p.close()
		return nil, err
	}

	return &p, nil
}

// metalPlan owns the immutable Metal device handle and pre-uploaded constant
// weights shared across sessions of one loaded Model.
type metalPlan struct {
	device  *gpu.Device
	options ExecutionOptions
	weights map[int]*gpu.Tensor
	stats   GPUStats
}

func (p *metalPlan) uploadConstants(ctx context.Context, values []operators.Value, steps []operators.Step) (err error) {
	var (
		constants       = make([]*graph.Constant, len(values))
		usedGeneral     = make([]bool, len(values))
		needsFloat32    = make([]bool, len(values))
		reducedEligible = make([]bool, len(values))
	)

	for index, value := range values {
		if err = ctx.Err(); err != nil {
			return err
		}

		if !value.Input && value.Producer < 0 && value.Default != nil && value.Default.DataType() == Float32 {
			constants[index] = value.Default
		}
	}

	for _, step := range steps {
		if err = ctx.Err(); err != nil {
			return err
		}

		if step.Operator == "Constant" && step.Constant != nil && step.Constant.DataType() == Float32 {
			constants[step.Outputs[0]] = step.Constant
			continue
		}

		p.classifyStepInputs(step, constants, usedGeneral, needsFloat32, reducedEligible)
	}

	for index, constant := range constants {
		if err = ctx.Err(); err != nil {
			return err
		}

		if constant == nil || !usedGeneral[index] {
			continue
		}

		shape := constant.Borrow().Shape()
		uploadShape := shape
		if len(shape) == 0 {
			uploadShape = []int64{1}
		} else if !validGPUShape(shape) {
			continue
		}

		var hostValues []float32
		if hostValues, err = constant.Float32Values(); err != nil {
			return fmt.Errorf("read constant %q for Metal upload: %w", values[index].Name, err)
		}

		dtype := gpu.Float32
		if p.options.WeightPrecision != WeightFloat32 && reducedEligible[index] && !needsFloat32[index] && len(shape) >= 1 && len(shape) <= 6 {
			dtype = p.options.WeightPrecision.gpuDType()
		}

		var weight *gpu.Tensor
		if weight, err = p.uploadTensor(uploadShape, dtype, hostValues); err != nil {
			return fmt.Errorf("upload Metal weight %q: %w", values[index].Name, err)
		}

		p.weights[index] = weight
		p.recordWeight(weight, uint64(len(hostValues))*4)
	}

	return nil
}

func (p *metalPlan) classifyStepInputs(
	step operators.Step,
	constants []*graph.Constant,
	usedGeneral []bool,
	needsFloat32 []bool,
	reducedEligible []bool,
) {
	switch step.Operator {
	case "MatMul":
		if left := step.Inputs[0]; left >= 0 {
			usedGeneral[left] = true
			needsFloat32[left] = true
		}

		if right := step.Inputs[1]; right >= 0 {
			usedGeneral[right] = true
			if constants[right] != nil && constants[right].Rank() >= 2 && constants[right].Rank() <= 6 {
				reducedEligible[right] = true
			} else {
				needsFloat32[right] = true
			}
		}
	case "Gemm":
		if left := step.Inputs[0]; left >= 0 {
			usedGeneral[left] = true
			needsFloat32[left] = true
		}

		if right := step.Inputs[1]; right >= 0 {
			usedGeneral[right] = true
			if constants[right] != nil && constants[right].Rank() == 2 {
				reducedEligible[right] = true
			} else {
				needsFloat32[right] = true
			}
		}

		if len(step.Inputs) > 2 && step.Inputs[2] >= 0 {
			bias := step.Inputs[2]
			usedGeneral[bias] = true
			needsFloat32[bias] = true
		}
	case "Add", "Sub", "Mul", "Div":
		if left := step.Inputs[0]; left >= 0 {
			usedGeneral[left] = true
			needsFloat32[left] = true
		}

		if right := step.Inputs[1]; right >= 0 {
			usedGeneral[right] = true
			needsFloat32[right] = true
		}
	case "Relu", "Sigmoid", "Tanh", "Softmax", "Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze":
		if input := step.Inputs[0]; input >= 0 {
			usedGeneral[input] = true
			needsFloat32[input] = true
		}
	case "LayerNormalization":
		if input := step.Inputs[0]; input >= 0 {
			usedGeneral[input] = true
			needsFloat32[input] = true
		}

		if scale := step.Inputs[1]; scale >= 0 {
			usedGeneral[scale] = true
			if constants[scale] != nil && constants[scale].Rank() == 1 {
				reducedEligible[scale] = true
			} else {
				needsFloat32[scale] = true
			}
		}

		if len(step.Inputs) > 2 && step.Inputs[2] >= 0 {
			bias := step.Inputs[2]
			usedGeneral[bias] = true
			if constants[bias] != nil && constants[bias].Rank() == 1 {
				reducedEligible[bias] = true
			} else {
				needsFloat32[bias] = true
			}
		}
	case "Gather":
		if step.Axis == 0 && step.Inputs[0] >= 0 {
			data := step.Inputs[0]
			usedGeneral[data] = true
			if constants[data] != nil && constants[data].Rank() == 2 {
				reducedEligible[data] = true
			} else {
				needsFloat32[data] = true
			}
		}
	}
}

func (p *metalPlan) uploadTensor(shape []int64, dtype gpu.DType, values []float32) (out *gpu.Tensor, err error) {
	if out, err = p.device.NewTensorWithDType(shape, dtype); err != nil {
		return nil, err
	}

	if err = out.UploadFloat32As(values); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func (p *metalPlan) recordWeight(weight *gpu.Tensor, fp32Bytes uint64) {
	p.stats.WeightTensors++
	p.stats.WeightBytes += weight.Bytes()
	p.stats.FP32EquivalentWeightBytes += fp32Bytes
	if weight.DType() != gpu.Float32 {
		p.stats.ReducedPrecisionTensors++
	}
}

func (p *metalPlan) close() (err error) {
	if p == nil {
		return nil
	}

	for index, weight := range p.weights {
		if weight != nil {
			err = errors.Join(err, weight.Close())
		}

		delete(p.weights, index)
	}

	if p.device != nil {
		err = errors.Join(err, p.device.Close())
		p.device = nil
	}

	return err
}

func positiveShape(shape []int64) (ok bool) {
	if len(shape) == 0 {
		return false
	}

	for _, dimension := range shape {
		if dimension <= 0 {
			return false
		}
	}

	return true
}

func validGPUShape(shape []int64) (ok bool) {
	if !positiveShape(shape) {
		return false
	}

	count := int64(1)
	for _, dimension := range shape {
		if dimension > math.MaxUint32/count {
			return false
		}

		count *= dimension
	}

	return true
}
