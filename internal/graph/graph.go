package graph

import (
	"context"

	"github.com/itsmontoya/onnx/internal/onnxpb"
)

// Build validates a complete decoded model and snapshots the whole named graph.
// encodedBytes must be the actual original encoded length, never proto.Size.
// Apply bounded wire parsing first; do not mutate model during this call. No wire
// message or decoder is retained. Operator/schema validation belongs to RT-07.
func Build(ctx context.Context, model *onnxpb.ModelProto, encodedBytes int64, limits Limits) (out *Graph, err error) {
	var (
		g Graph
		b *builder
	)

	if b, err = newBuilder(ctx, model, encodedBytes, limits, &g); err != nil {
		return nil, err
	}

	if err = b.build(model); err != nil {
		return nil, err
	}

	g.loadBytes, g.initializerBytes = b.decoder.Usage()
	return &g, nil
}

// Graph owns an immutable structural snapshot. All record accessors copy slices;
// shared Constants expose no writable backend storage. Accessors may be called
// concurrently after a successful Build.
type Graph struct {
	metadata         Metadata
	inputs           []ValueInfo
	outputs          []ValueInfo
	nodes            []Node
	values           []Value
	indices          map[string]int
	loadBytes        int64
	initializerBytes int64
}

// Metadata returns an independent descriptive snapshot.
func (g *Graph) Metadata() (metadata Metadata) { return g.metadata.clone() }

// Inputs returns input contracts in declaration order, including default flags.
func (g *Graph) Inputs() (inputs []ValueInfo) {
	inputs = make([]ValueInfo, len(g.inputs))
	for index, input := range g.inputs {
		inputs[index] = input.clone()
	}
	return inputs
}

// Outputs returns output contracts in declaration order.
func (g *Graph) Outputs() (outputs []ValueInfo) {
	outputs = make([]ValueInfo, len(g.outputs))
	for index, output := range g.outputs {
		outputs[index] = output.clone()
	}
	return outputs
}

// Nodes returns every node in validated order, including unused output paths.
func (g *Graph) Nodes() (nodes []Node) {
	nodes = make([]Node, len(g.nodes))
	for index, node := range g.nodes {
		nodes[index] = node.clone()
	}
	return nodes
}

// Values returns distinct producers in source/declared node order.
func (g *Graph) Values() (values []Value) {
	values = make([]Value, len(g.values))
	for index, value := range g.values {
		values[index] = value.clone()
	}
	return values
}

// Value returns an independent named record when a producer exists.
func (g *Graph) Value(name string) (value Value, found bool) {
	var index int
	if index, found = g.indices[name]; !found {
		return value, false
	}

	return g.values[index].clone(), true
}

// Usage returns conservative load payload and owned constant numerical bytes.
// Snapshot slice payloads are included; Go object/header overhead is excluded.
func (g *Graph) Usage() (loadBytes, initializerBytes int64) {
	return g.loadBytes, g.initializerBytes
}
