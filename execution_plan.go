package onnx

import (
	"context"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

func newExecutionPlan(ctx context.Context, compiled *operators.Plan, limits Limits) (out *executionPlan, err error) {
	var p executionPlan
	p.limits = limits
	p.loadBytes, p.residentBytes = compiled.Usage()
	// Count copied metadata rather than multiplying prior load reservations:
	// neither constant storage nor immutable string data is copied here.
	var snapshotBytes int64
	if snapshotBytes, err = compiled.SnapshotBytes(ctx); err != nil {
		return nil, err
	}

	if p.loadBytes > limits.MaxLoadBytes || snapshotBytes > limits.MaxLoadBytes-p.loadBytes {
		return nil, fmt.Errorf("%w: executable plan snapshots", ErrLimit)
	}

	p.loadBytes += snapshotBytes
	description := compiled.Metadata()
	p.inputs, p.outputs = compiled.Inputs(), compiled.Outputs()
	p.values, p.steps = compiled.Values(), compiled.Steps()
	p.metadata = Metadata{Name: description.Name, Producer: description.Producer, IRVersion: description.IRVersion, Opset: description.Opset}
	if p.metadata.Inputs, err = publicInfos(ctx, p.inputs); err != nil {
		return nil, err
	}

	if p.metadata.Outputs, err = publicInfos(ctx, p.outputs); err != nil {
		return nil, err
	}

	p.indices, p.inputIndices, p.outputNames = make(map[string]int), make(map[string]int), make(map[string]int)
	for index, value := range p.values {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		p.indices[value.Name] = index
		if value.Input {
			p.inputIndices[value.Name] = index
		}
	}
	for _, output := range p.outputs {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		p.outputIndices = append(p.outputIndices, p.indices[output.Name])
		p.outputNames[output.Name] = p.indices[output.Name]
	}
	// Exact-capacity storage uses at most five int arrays plus pinned flags
	// per value (41 bytes), and one temporary int counter per step (8 bytes).
	// This includes construction work; Go slice headers/objects are not payload.
	available := limits.MaxLoadBytes - p.loadBytes
	if int64(len(p.values)) > available/41 {
		return nil, fmt.Errorf("%w: value lifetime metadata", ErrLimit)
	}

	lifetimeBytes := int64(len(p.values)) * 41
	if int64(len(p.steps)) > (available-lifetimeBytes)/8 {
		return nil, fmt.Errorf("%w: lifetime retirement counters", ErrLimit)
	}

	p.loadBytes += lifetimeBytes + int64(len(p.steps))*8
	if err = p.prepareCalls(ctx); err != nil {
		return nil, err
	}

	if p.lifetimes, err = newValueLifetimes(ctx, &p); err != nil {
		return nil, err
	}

	return &p, nil
}

type executionPlan struct {
	mu            sync.Mutex
	sessions      int
	limits        Limits
	loadBytes     int64
	residentBytes int64
	metadata      Metadata
	inputs        []graph.ValueInfo
	outputs       []graph.ValueInfo
	values        []operators.Value
	steps         []operators.Step
	indices       map[string]int
	inputIndices  map[string]int
	outputIndices []int
	outputNames   map[string]int
	lifetimes     *valueLifetimes
	calls         []*preparedCall
	callBytes     int64
}

func (p *executionPlan) prepareCalls(ctx context.Context) (err error) {
	var constants []*graph.Constant
	for position, step := range p.steps {
		if err = ctx.Err(); err != nil {
			return err
		}

		if !operators.IsCNNOperator(step.Operator) && !preparedNumericalOperator(step.Operator) && step.Operator != "Reshape" {
			continue
		}

		// Bound copied scalar metadata/configs (512 bytes), all input shapes,
		// output shape and Reshape target before allocating preparation state.
		bytes, available := int64(512), p.limits.MaxLoadBytes-p.loadBytes
		if available < bytes || int64(p.limits.MaxRank) > (available-bytes)/16 {
			continue
		}

		bytes += int64(p.limits.MaxRank) * 16
		for _, index := range step.Inputs {
			if index >= 0 {
				rank := int64(len(p.values[index].Info.Dimensions))
				if rank > (available-bytes)/8 {
					bytes = available + 1
					break
				}

				bytes += rank * 8
			}
		}
		if bytes > available {
			continue
		}

		var target *graph.Constant
		if step.Operator == "Reshape" {
			if constants == nil {
				if constants, err = preparedCallConstants(ctx, p.values, p.steps); err != nil {
					return err
				}
			}
			target = constants[step.Inputs[1]]
		}
		call := newPreparedCall(ctx, step, p.values, target, p.limits)
		if call == nil {
			continue
		}

		if p.calls == nil {
			p.calls = make([]*preparedCall, len(p.steps))
		}

		p.calls[position] = call
		p.loadBytes += bytes
		p.callBytes += bytes
	}
	return p.prepareReshapes(ctx, constants)
}

func (p *executionPlan) prepareReshapes(ctx context.Context, constants []*graph.Constant) (err error) {
	if constants == nil {
		return ctx.Err()
	}

	var liveBudget int64 = -1
	for position, call := range p.calls {
		if err = ctx.Err(); err != nil {
			return err
		}

		if call == nil || call.shape == nil {
			continue
		}

		source := constants[p.steps[position].Inputs[0]]
		if source == nil {
			continue
		}

		if liveBudget < 0 {
			liveBudget = p.preparedLiveBudget(ctx)
		}

		liveBudget -= p.prepareReshape(ctx, call, source, liveBudget)
	}
	return ctx.Err()
}

// preparedLiveBudget bounds all value slots and publication copies together,
// even when their actual lifetimes do not overlap. Unknown signatures decline
// optional payload preparation. This preserves previous tight Run limits without
// adding a runtime fallback or discounting physically resident prepared payload.
func (p *executionPlan) preparedLiveBudget(ctx context.Context) (available int64) {
	for _, step := range p.steps {
		// Normalization also reserves optional, unpublished statistics tensors.
		// Their storage is not represented by the named value slots below.
		if ctx.Err() != nil || step.Operator == "LayerNormalization" {
			return 0
		}
	}
	available = p.limits.MaxLiveBytes - p.residentBytes
	if available < p.limits.MaxWorkspaceBytes {
		return 0
	}

	available -= p.limits.MaxWorkspaceBytes
	for _, value := range p.values {
		if ctx.Err() != nil || !value.Info.ShapeKnown || value.Info.DataType == Invalid {
			return 0
		}

		_, bytes, shapeErr := tensor.ValidateShape(nil, value.Info.DataType)
		if shapeErr != nil {
			return 0
		}

		for _, dimension := range value.Info.Dimensions {
			if ctx.Err() != nil || dimension.Kind != graph.FixedDimension || dimension.Value < 0 {
				return 0
			}

			if dimension.Value != 0 && bytes > available/dimension.Value {
				return 0
			}

			bytes *= dimension.Value
		}
		// Binding reserves a supplied/defaulted input before checking its
		// declaration. Cover invalid signatures too, preserving error priority.
		if value.Input {
			bytes = p.limits.MaxTensorBytes
		}

		if bytes > available {
			return 0
		}

		available -= bytes
		if _, output := p.outputNames[value.Name]; output {
			if bytes > available {
				return 0
			}

			available -= bytes
		}
	}
	return available
}

func (p *executionPlan) prepareReshape(ctx context.Context, call *preparedCall, source *graph.Constant, liveBudget int64) (retainedBytes int64) {
	shape := *call.shape
	bytes := shape.bytes
	// Charge the owned shape and the backend's temporary resolved-shape copy
	// as load work, as well as the persistent numerical payload.
	metadata := int64(len(shape.shape)) * 16
	available := p.limits.MaxLoadBytes - p.loadBytes
	if bytes > liveBudget || bytes > p.limits.MaxTensorBytes || shape.workspace > p.limits.MaxWorkspaceBytes || metadata > available || bytes > available-metadata {
		return 0
	}

	shape.inputs = []*tensor.Tensor{source.Borrow()}
	var (
		retained *tensor.Tensor
		err      error
	)

	if retained, err = shape.run(ctx, nil); err != nil {
		return 0
	}

	call.retained = retained
	p.loadBytes += bytes + metadata
	p.residentBytes += bytes
	return bytes
}

func (p *executionPlan) acquire() (err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.sessions >= p.limits.MaxSessions {
		return fmt.Errorf("%w: live sessions %d", ErrLimit, p.sessions)
	}

	p.sessions++
	return nil
}

func (p *executionPlan) release() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.sessions--
}

func publicInfos(ctx context.Context, infos []graph.ValueInfo) (out []ValueInfo, err error) {
	out = make([]ValueInfo, len(infos))
	for index, info := range infos {
		if err = ctx.Err(); err != nil {
			return nil, err
		}

		out[index] = ValueInfo{Name: info.Name, DataType: info.DataType, ShapeKnown: info.ShapeKnown, Defaulted: info.Defaulted}
		out[index].Dimensions = make([]Dimension, len(info.Dimensions))
		for axis, dimension := range info.Dimensions {
			if err = ctx.Err(); err != nil {
				return nil, err
			}

			out[index].Dimensions[axis] = Dimension{Kind: DimensionKind(dimension.Kind), Value: dimension.Value, Symbol: dimension.Symbol}
		}
	}
	return out, nil
}
