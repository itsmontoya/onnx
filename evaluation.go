package onnx

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newEvaluation(ctx context.Context, plan *executionPlan) (out *evaluation, err error) {
	var e evaluation
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if e.budget, err = newRunBudget(plan.limits, plan.residentBytes); err != nil {
		return nil, err
	}

	e.ctx, e.plan = ctx, plan
	e.values = make([]*tensor.Tensor, len(plan.values))
	e.symbols = make(map[string]int64)
	return &e, nil
}

// evaluation retains all table entries for one run. Even aliases are charged
// as separate live values; the plan's immutable constants are charged once.
type evaluation struct {
	ctx     context.Context
	plan    *executionPlan
	budget  *runBudget
	values  []*tensor.Tensor
	symbols map[string]int64
}

func (e *evaluation) run(inputs map[string]*Tensor, options RunOptions) (outputs []NamedTensor, err error) {
	var requested []int
	if requested, err = e.requested(options.Outputs); err != nil {
		return nil, err
	}

	if err = e.checkNames(inputs); err != nil {
		return nil, err
	}

	if err = e.bind(inputs); err != nil {
		return nil, err
	}

	for _, step := range e.plan.steps {
		if err = e.ctx.Err(); err != nil {
			return nil, e.nodeError(step, err)
		}

		if err = e.execute(step); err != nil {
			return nil, e.nodeError(step, err)
		}

		if err = e.ctx.Err(); err != nil {
			return nil, e.nodeError(step, err)
		}
	}
	return e.publish(requested)
}

func (e *evaluation) requested(names []string) (indices []int, err error) {
	if len(names) == 0 {
		return e.plan.outputIndices, nil
	}

	seen := make(map[string]bool)
	for _, name := range names {
		if err = e.ctx.Err(); err != nil {
			return nil, err
		}

		index, declared := e.plan.outputNames[name]
		if !declared || seen[name] {
			return nil, operationError("run", fmt.Errorf("%w: requested output must be declared and unique", ErrInvalidArgument), name)
		}

		seen[name] = true
		indices = append(indices, index)
	}
	return indices, nil
}

func (e *evaluation) checkNames(inputs map[string]*Tensor) (err error) {
	var unexpected string
	var found bool
	for name := range inputs {
		if err = e.ctx.Err(); err != nil {
			return err
		}

		if _, exists := e.plan.inputIndices[name]; !exists && (!found || name < unexpected) {
			unexpected, found = name, true
		}
	}

	if found {
		return operationError("run", fmt.Errorf("%w: unexpected input", ErrInvalidArgument), unexpected)
	}

	return nil
}

func (e *evaluation) bind(inputs map[string]*Tensor) (err error) {
	for index, value := range e.plan.values {
		if err = e.ctx.Err(); err != nil {
			return err
		}

		if value.Producer >= 0 {
			continue
		}

		var backend *tensor.Tensor
		if provided, exists := inputs[value.Name]; value.Input && exists {
			if err = provided.Validate(); err != nil {
				return operationError("run", err, value.Name)
			}

			backend = provided.value
			if err = e.budget.reserve(backend.ByteSize(), 0); err != nil {
				return operationError("run", err, value.Name)
			}
		} else if value.Default != nil {
			if backend, err = e.cloneConstant(value.Default); err != nil {
				return operationError("run", err, value.Name)
			}
		} else {
			return operationError("run", fmt.Errorf("%w: required input is missing", ErrInvalidArgument), value.Name)
		}

		kind := ErrInvalidModel
		if value.Input {
			kind = ErrInvalidArgument
		}

		if err = e.checkValue(value, backend, kind); err != nil {
			return operationError("run", err, value.Name)
		}

		e.values[index] = backend
	}
	return nil
}

func (e *evaluation) execute(step operators.Step) (err error) {
	var backend *tensor.Tensor
	switch step.Operator {
	case "Identity":
		backend = e.values[step.Inputs[0]]
		if err = e.budget.reserve(backend.ByteSize(), 0); err != nil {
			return err
		}
	case "Constant":
		if backend, err = e.cloneConstant(step.Constant); err != nil {
			return err
		}
	default:
		var call *numericalCall
		inputs := make([]*tensor.Tensor, len(step.Inputs))
		for slot, valueIndex := range step.Inputs {
			if valueIndex >= 0 {
				inputs[slot] = e.values[valueIndex]
			}
		}
		if call, err = newNumericalCall(step, inputs, e.plan.limits); err != nil {
			return err
		}

		if err = e.budget.reserve(call.bytes, call.workspace); err != nil {
			return err
		}

		defer e.budget.releaseWorkspace()
		if backend, err = call.run(e.ctx); err != nil {
			return err
		}
	}

	index := step.Outputs[0]
	if err = e.checkValue(e.plan.values[index], backend, ErrInvalidModel); err != nil {
		return operationError("run", err, e.plan.values[index].Name)
	}

	e.values[index] = backend
	return nil
}

func (e *evaluation) checkValue(value operators.Value, backend *tensor.Tensor, kind error) (err error) {
	if backend.Rank() > e.plan.limits.MaxRank {
		return fmt.Errorf("%w: runtime rank %d", ErrLimit, backend.Rank())
	}

	shape := backend.Shape()
	for axis, size := range shape {
		if err = e.ctx.Err(); err != nil {
			return err
		}

		if size > e.plan.limits.MaxDimension {
			return fmt.Errorf("%w: runtime axis %d size %d", ErrLimit, axis, size)
		}
	}

	// Inference can refine away a symbol or unknown axis. Original declarations
	// still constrain that concrete value and bind symbols for the entire run.
	for _, declaration := range value.Declarations {
		if err = e.checkInfo(declaration, backend.DataType(), shape, kind); err != nil {
			return err
		}
	}

	return e.checkInfo(value.Info, backend.DataType(), shape, kind)
}

func (e *evaluation) checkInfo(info graph.ValueInfo, dataType DataType, shape []int64, kind error) (err error) {
	if err = e.ctx.Err(); err != nil {
		return err
	}

	if info.DataType != Invalid && info.DataType != dataType {
		return fmt.Errorf("%w: dtype %v, declared %v", kind, dataType, info.DataType)
	}

	if !info.ShapeKnown {
		return nil
	}

	if len(info.Dimensions) != len(shape) {
		return fmt.Errorf("%w: rank %d, declared %d", kind, len(shape), len(info.Dimensions))
	}

	for axis, dimension := range info.Dimensions {
		if err = e.ctx.Err(); err != nil {
			return err
		}

		size := shape[axis]
		if dimension.Kind == graph.FixedDimension && dimension.Value != size {
			return fmt.Errorf("%w: axis %d size %d, declared %d", kind, axis, size, dimension.Value)
		}

		if dimension.Kind != graph.SymbolicDimension {
			continue
		}

		if previous, exists := e.symbols[dimension.Symbol]; exists && previous != size {
			return fmt.Errorf("%w: axis %d symbol %q size %d, bound %d", kind, axis, dimension.Symbol, size, previous)
		}

		e.symbols[dimension.Symbol] = size
	}
	return nil
}

func (e *evaluation) cloneConstant(constant *graph.Constant) (out *tensor.Tensor, err error) {
	if err = e.ctx.Err(); err != nil {
		return nil, err
	}

	// Released tensor.Clone copies directly into its owned destination.
	// There is no numerical accessor/staging buffer beyond the live copy.
	if err = e.budget.reserve(constant.ByteSize(), 0); err != nil {
		return nil, err
	}

	defer e.budget.releaseWorkspace()
	if out, err = constant.Clone(); err != nil {
		return nil, fmt.Errorf("clone model constant: %w", err)
	}

	if err = e.ctx.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

func (e *evaluation) publish(indices []int) (outputs []NamedTensor, err error) {
	outputs = make([]NamedTensor, len(indices))
	for index, valueIndex := range indices {
		if err = e.ctx.Err(); err != nil {
			return nil, err
		}

		name := e.plan.values[valueIndex].Name
		var value *Tensor
		if value, err = e.cloneOutput(e.values[valueIndex]); err != nil {
			return nil, operationError("run", err, name)
		}

		outputs[index] = NamedTensor{Name: name, Tensor: value}
	}

	if err = e.ctx.Err(); err != nil {
		return nil, err
	}

	return outputs, nil
}

func (e *evaluation) cloneOutput(source *tensor.Tensor) (out *Tensor, err error) {
	if err = e.budget.reserve(source.ByteSize(), 0); err != nil {
		return nil, err
	}

	defer e.budget.releaseWorkspace()
	var value Tensor
	if value.value, err = source.Clone(); err != nil {
		return nil, fmt.Errorf("clone requested output: %w", err)
	}

	if err = e.ctx.Err(); err != nil {
		return nil, err
	}

	return &value, nil
}

func (e *evaluation) nodeError(step operators.Step, cause error) (out *Error) {
	diagnostic := operationError("run", cause, "")
	diagnostic.NodeIndex, diagnostic.NodeName = step.Index, step.Name
	diagnostic.Domain, diagnostic.Operator, diagnostic.Opset = step.Domain, step.Operator, step.Opset
	for _, index := range step.Inputs {
		var shape []int64
		if index >= 0 && e.values[index] != nil {
			shape = e.values[index].Shape()
		}

		diagnostic.Shapes = append(diagnostic.Shapes, shape)
	}
	return diagnostic
}
