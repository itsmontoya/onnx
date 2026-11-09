package operators

import (
	"context"

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

func cloneInfos(infos []graph.ValueInfo) (out []graph.ValueInfo) {
	out = make([]graph.ValueInfo, len(infos))
	for index, info := range infos {
		out[index] = cloneInfo(info)
	}
	return out
}
