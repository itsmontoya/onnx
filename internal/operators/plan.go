package operators

import (
	"context"
	"fmt"
	"math"

	"github.com/itsmontoya/onnx/internal/graph"
)

// Compile resolves every graph node in order and returns an immutable plan.
// Deep graph snapshots are reserved before access. Constant tensors are shared
// immutably; shape inference never constructs or clones numerical output tensors.
func Compile(ctx context.Context, source *graph.Graph, registry *Registry, limits Limits) (out *Plan, err error) {
	var (
		p Plan
		c *compiler
	)

	if c, err = newCompiler(ctx, source, registry, limits, &p); err != nil {
		return nil, err
	}

	if err = c.compile(source); err != nil {
		return nil, err
	}

	p.loadBytes = c.loadBytes
	return &p, nil
}

// Plan is an immutable protobuf-free execution description. Copying accessors
// are safe concurrently; their caller-owned snapshots are outside load budgets.
type Plan struct {
	metadata         graph.Metadata
	inputs           []graph.ValueInfo
	outputs          []graph.ValueInfo
	values           []Value
	steps            []Step
	indices          map[string]int
	loadBytes        int64
	initializerBytes int64
}

// Metadata returns an independent descriptive snapshot.
func (p *Plan) Metadata() (metadata graph.Metadata) {
	metadata = p.metadata
	metadata.Properties = append([]graph.Property(nil), p.metadata.Properties...)
	metadata.GraphProperties = append([]graph.Property(nil), p.metadata.GraphProperties...)
	metadata.Annotations = make([]graph.Annotation, len(p.metadata.Annotations))
	for index, annotation := range p.metadata.Annotations {
		metadata.Annotations[index] = annotation
		metadata.Annotations[index].Parameters = append([]graph.Property(nil), annotation.Parameters...)
	}
	return metadata
}

// Inputs returns original input contracts in declaration order.
func (p *Plan) Inputs() (inputs []graph.ValueInfo) { return cloneInfos(p.inputs) }

// Outputs returns original output contracts in declaration order.
func (p *Plan) Outputs() (outputs []graph.ValueInfo) { return cloneInfos(p.outputs) }

// Values returns source/producer-ordered inference records and constraints.
func (p *Plan) Values() (values []Value) {
	values = make([]Value, len(p.values))
	for index, value := range p.values {
		values[index] = value.clone()
	}
	return values
}

// Value returns a named immutable record's independent snapshot.
func (p *Plan) Value(name string) (value Value, found bool) {
	var index int
	if index, found = p.indices[name]; !found {
		return value, false
	}

	return p.values[index].clone(), true
}

// Steps returns all resolved nodes in their original order.
func (p *Plan) Steps() (steps []Step) {
	steps = make([]Step, len(p.steps))
	for index, step := range p.steps {
		steps[index] = step.clone()
	}
	return steps
}

// Usage includes graph construction and conservative compilation reservations.
// Initializer bytes are unchanged: compilation shares rather than copies tensors.
func (p *Plan) Usage() (loadBytes, initializerBytes int64) {
	return p.loadBytes, p.initializerBytes
}

// SnapshotBytes bounds the logical payload copied by one call to each deep
// accessor plus a consumer's second input/output metadata copy. It allocates no
// snapshots. Constants and immutable string bytes are shared, not copied; Go
// headers/capacity overhead remains outside payload accounting.
func (p *Plan) SnapshotBytes(ctx context.Context) (bytes int64, err error) {
	if ctx == nil || p == nil {
		return 0, fmt.Errorf("%w: snapshot context or plan", graph.ErrInvalidArgument)
	}

	// Fixed numeric metadata fits in 32 bytes per value and 256 per step,
	// including CNN configs and version flags. Dimension records allow 16 bytes for kind/size.
	bytes = 32
	for repeat := 0; repeat < 2; repeat++ {
		for _, infos := range [][]graph.ValueInfo{p.inputs, p.outputs} {
			for _, info := range infos {
				if err = ctx.Err(); err != nil {
					return 0, err
				}

				if err = addSnapshotInfo(&bytes, info); err != nil {
					return 0, err
				}
			}
		}
	}
	for _, value := range p.values {
		if err = ctx.Err(); err != nil {
			return 0, err
		}

		if err = addSnapshotInfo(&bytes, value.Info); err != nil {
			return 0, err
		}

		for _, declaration := range value.Declarations {
			if err = addSnapshotInfo(&bytes, declaration); err != nil {
				return 0, err
			}
		}
	}
	for _, step := range p.steps {
		if err = ctx.Err(); err != nil {
			return 0, err
		}

		if bytes > math.MaxInt64-fixedStepBytes || int64(len(step.Inputs)) > (math.MaxInt64-bytes-fixedStepBytes)/8 {
			return 0, fmt.Errorf("%w: snapshot input indices", graph.ErrLimit)
		}

		bytes += fixedStepBytes + int64(len(step.Inputs))*8
		if int64(len(step.Outputs)) > (math.MaxInt64-bytes)/8 {
			return 0, fmt.Errorf("%w: snapshot output indices", graph.ErrLimit)
		}

		bytes += int64(len(step.Outputs)) * 8
		if int64(len(step.Permutation)) > (math.MaxInt64-bytes)/8 {
			return 0, fmt.Errorf("%w: snapshot permutation", graph.ErrLimit)
		}

		bytes += int64(len(step.Permutation)) * 8
	}

	if err = ctx.Err(); err != nil {
		return 0, err
	}

	return bytes, nil
}

func addSnapshotInfo(bytes *int64, info graph.ValueInfo) (err error) {
	if *bytes > math.MaxInt64-32 || int64(len(info.Dimensions)) > (math.MaxInt64-*bytes-32)/16 {
		return fmt.Errorf("%w: snapshot dimensions", graph.ErrLimit)
	}

	*bytes += 32 + int64(len(info.Dimensions))*16
	return nil
}

func cloneInfos(infos []graph.ValueInfo) (out []graph.ValueInfo) {
	out = make([]graph.ValueInfo, len(infos))
	for index, info := range infos {
		out[index] = cloneInfo(info)
	}
	return out
}
