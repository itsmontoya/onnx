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
	e.storage = newRunStorage(plan.lifetimes, e.budget)
	e.symbols = make(map[string]int64)
	return &e, nil
}

// evaluation owns call-local values and symbols. Static alias/consumer groups
// allow conservative reuse without changing returned-output ownership.
type evaluation struct {
	ctx       context.Context
	plan      *executionPlan
	budget    *runBudget
	storage   runStorage
	symbols   map[string]int64
	arguments [3]*tensor.Tensor
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

	for position, step := range e.plan.steps {
		if err = e.ctx.Err(); err != nil {
			return nil, e.nodeError(step, err)
		}

		if err = e.execute(step); err != nil {
			return nil, e.nodeError(step, err)
		}

		if err = e.ctx.Err(); err != nil {
			return nil, e.nodeError(step, err)
		}

		e.storage.after(position)
	}
	e.storage.trim()
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
			if err = e.storage.reserve(backend.ByteSize(), 0); err != nil {
				return operationError("run", err, value.Name)
			}
		} else if value.Default != nil {
			if backend, err = e.constantValue(value.Default); err != nil {
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

		e.storage.values[index] = backend
	}
	return nil
}

func (e *evaluation) execute(step operators.Step) (err error) {
	if step.Operator == "LayerNormalization" {
		return e.normalize(step)
	}

	var backend *tensor.Tensor
	e.storage.slot, e.storage.fresh = -1, false

	switch step.Operator {
	case "Identity":
		backend = e.storage.values[step.Inputs[0]]
	case "Constant":
		if backend, err = e.constantValue(step.Constant); err != nil {
			return err
		}
	default:
		var call *numericalCall
		inputs := e.nodeArguments(len(step.Inputs))
		for slot, valueIndex := range step.Inputs {
			if err = e.ctx.Err(); err != nil {
				return err
			}

			if valueIndex >= 0 {
				inputs[slot] = e.storage.values[valueIndex]
			}
		}
		if step.Index >= 0 && step.Index < len(e.plan.calls) && e.plan.calls[step.Index] != nil {
			e.storage.slot = step.Index
			if backend, err = e.plan.calls[step.Index].run(e, inputs); err != nil {
				return err
			}

			break
		}

		if operators.IsCNNOperator(step.Operator) {
			var cnn *cnnCall
			if cnn, err = newCNNCall(step, inputs, e.plan.limits); err != nil {
				return err
			}

			var destination *tensor.Tensor
			if destination, err = e.storage.destination(Float32, cnn.shape, cnn.bytes, 0); err != nil {
				return err
			}

			if backend, err = cnn.run(e.ctx, destination); err != nil {
				return err
			}

			break
		}

		if operators.IsShapeOperator(step.Operator) {
			var shape *shapeCall
			if shape, err = newShapeCall(step, inputs, e.plan.limits); err != nil {
				return err
			}

			if err = e.storage.reserve(0, shape.workspace); err != nil {
				return err
			}

			defer e.budget.releaseWorkspace()
			if err = shape.plan(e.ctx); err != nil {
				return err
			}

			var destination *tensor.Tensor
			if shape.step.Operator == "Shape" {
				err = e.storage.reserve(shape.bytes, 0)
			} else {
				destination, err = e.storage.destination(shape.dataType, shape.shape, shape.bytes, 0)
			}

			if err != nil {
				return err
			}

			if backend, err = shape.run(e.ctx, destination); err != nil {
				return err
			}

			break
		}

		if call, err = newNumericalCall(step, inputs, e.plan.limits); err != nil {
			return err
		}

		var destination *tensor.Tensor
		if destination, err = e.storage.destination(Float32, call.shape, call.bytes, call.workspace); err != nil {
			return err
		}

		defer e.budget.releaseWorkspace()
		if backend, err = call.run(e.ctx, destination); err != nil {
			return err
		}
	}

	index := step.Outputs[0]
	if err = e.checkValue(e.plan.values[index], backend, ErrInvalidModel); err != nil {
		return operationError("run", err, e.plan.values[index].Name)
	}

	if e.storage.fresh {
		e.storage.destinations.keep(e.storage.slot, backend, e.budget)
	}

	e.storage.values[index] = backend
	return nil
}

func (e *evaluation) normalize(step operators.Step) (err error) {
	inputs := e.nodeArguments(len(step.Inputs))
	for slot, index := range step.Inputs {
		if index >= 0 {
			inputs[slot] = e.storage.values[index]
		}
	}
	var call *normalizationCall
	if call, err = newNormalizationCall(step, inputs, e.plan.limits); err != nil {
		return err
	}

	if err = e.storage.reserve(0, call.workspace); err != nil {
		return err
	}

	defer e.budget.releaseWorkspace()
	for _, bytes := range call.bytes {
		if err = e.storage.reserve(bytes, 0); err != nil {
			return err
		}
	}
	var outputs []*tensor.Tensor
	if outputs, err = call.run(e.ctx); err != nil {
		return err
	}

	for slot, index := range step.Outputs {
		if index < 0 {
			continue
		}

		if err = e.checkValue(e.plan.values[index], outputs[slot], ErrInvalidModel); err != nil {
			return operationError("run", err, e.plan.values[index].Name)
		}

		e.storage.values[index] = outputs[slot]
	}
	return nil
}

// nodeArguments reuses invocation-local storage while handlers run synchronously.
// Clearing preserves omitted optional inputs after a preceding wider call.
func (e *evaluation) nodeArguments(count int) (inputs []*tensor.Tensor) {
	if count > len(e.arguments) {
		return make([]*tensor.Tensor, count)
	}

	inputs = e.arguments[:count]
	clear(inputs)
	return inputs
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

func (e *evaluation) constantValue(constant *graph.Constant) (out *tensor.Tensor, err error) {
	if err = e.ctx.Err(); err != nil {
		return nil, err
	}

	// Sources and Constant outputs (including Identity aliases) are pinned by
	// the lifetime plan and cannot become writable destinations. Keep the legacy
	// logical reservation so live/workspace limit behavior remains unchanged,
	// even though the decoded resident payload now serves both roles.
	if err = e.storage.reserve(constant.ByteSize(), 0); err != nil {
		return nil, err
	}

	defer e.budget.releaseWorkspace()
	out = constant.Borrow()
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
		if value, err = e.cloneOutput(e.storage.values[valueIndex]); err != nil {
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
	if err = e.storage.reserve(source.ByteSize(), 0); err != nil {
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
		if index >= 0 && e.storage.values[index] != nil {
			shape = e.storage.values[index].Shape()
		}

		diagnostic.Shapes = append(diagnostic.Shapes, shape)
	}
	return diagnostic
}
