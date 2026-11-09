// Package graph validates the initial ONNX envelope and whole named graph.
// It owns immutable, protobuf-independent snapshots. It does not resolve operator
// schemas, infer operator shapes, execute nodes, or establish numerical support.
package graph
