package onnx

import (
	"context"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newPreparedCall(ctx context.Context, step operators.Step, values []operators.Value, target *graph.Constant, limits Limits) (out *preparedCall) {
	var p preparedCall
	shapes := make([][]int64, len(step.Inputs))
	for slot, index := range step.Inputs {
		if ctx.Err() != nil {
			return nil
		}

		if index < 0 {
			continue
		}

		info := values[index].Info
		if !info.ShapeKnown || info.DataType == Invalid {
			return nil
		}

		shapes[slot] = make([]int64, len(info.Dimensions))
		for axis, dimension := range info.Dimensions {
			if ctx.Err() != nil {
				return nil
			}

			if dimension.Kind != graph.FixedDimension {
				return nil
			}

			shapes[slot][axis] = dimension.Value
		}
		if step.Operator != "Reshape" && info.DataType != Float32 {
			return nil
		}
	}

	// Failed preparation is deliberately deferred to the original Run path.
	// This must never turn an overridable default into a load-time error.
	switch {
	case operators.IsCNNOperator(step.Operator):
		if p.cnn = newPreparedCNNCall(step, shapes, limits); p.cnn == nil {
			return nil
		}
	case step.Operator == "Reshape":
		if p.shape = newPreparedReshapeCall(ctx, step, shapes[0], values[step.Inputs[0]].Info.DataType, target, limits); p.shape == nil {
			return nil
		}
	case preparedNumericalOperator(step.Operator):
		if p.numerical = newPreparedNumericalCall(step, shapes, limits); p.numerical == nil {
			return nil
		}
	default:
		return nil
	}
	return &p
}

// preparedCall contains immutable geometry/results, never invocation arguments. Every
// input's concrete Info is checked by bind/checkValue before a consumer runs.
// That existing validation is the guard; deferred shapes use the dynamic path.
type preparedCall struct {
	cnn       *cnnCall
	numerical *numericalCall
	shape     *shapeCall
	retained  *tensor.Tensor
}

func (p *preparedCall) run(e *evaluation, inputs []*tensor.Tensor) (out *tensor.Tensor, err error) {
	var destination *tensor.Tensor
	switch {
	case p.cnn != nil:
		call := *p.cnn
		call.inputs = inputs
		if destination, err = e.storage.destination(Float32, call.shape, call.bytes, 0); err != nil {
			return nil, err
		}

		return call.run(e.ctx, destination)
	case p.numerical != nil:
		call := *p.numerical
		call.inputs = inputs
		if destination, err = e.storage.destination(Float32, call.shape, call.bytes, call.workspace); err != nil {
			return nil, err
		}

		defer e.budget.releaseWorkspace()
		return call.run(e.ctx, destination)
	default:
		call := *p.shape
		call.inputs = inputs
		if p.retained != nil {
			e.storage.slot = -1
		}
		if err = e.storage.reserve(0, call.workspace); err != nil {
			return nil, err
		}

		defer e.budget.releaseWorkspace()
		// Keep both plan() context checks and their position relative to budget
		// reservations, even though the immutable target needs no replanning.
		if err = e.ctx.Err(); err != nil {
			return nil, err
		}

		if err = e.ctx.Err(); err != nil {
			return nil, err
		}

		if destination, err = e.storage.destination(call.dataType, call.shape, call.bytes, 0); err != nil {
			return nil, err
		}

		if p.retained != nil {
			// Retain the eliminated synchronous copy's two context boundaries.
			if err = e.ctx.Err(); err != nil {
				return nil, err
			}

			return p.retained, e.ctx.Err()
		}

		return call.run(e.ctx, destination)
	}
}

func preparedNumericalOperator(operator string) (supported bool) {
	switch operator {
	case "Add", "Sub", "Mul", "Div", "MatMul", "Relu", "Sigmoid", "Tanh", "Softmax":
		return true
	}
	return false
}

// preparedCallConstants computes immutable provenance once in graph order.
// The temporary reference table avoids repeatedly walking long Identity chains.
func preparedCallConstants(ctx context.Context, values []operators.Value, steps []operators.Step) (constants []*graph.Constant, err error) {
	constants = make([]*graph.Constant, len(values))
	for index, value := range values {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		if !value.Input && value.Producer < 0 {
			constants[index] = value.Default
		}
	}
	for _, step := range steps {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		if step.Operator == "Constant" {
			constants[step.Outputs[0]] = step.Constant
		}

		if step.Operator == "Identity" {
			constants[step.Outputs[0]] = constants[step.Inputs[0]]
		}
	}
	return constants, nil
}
