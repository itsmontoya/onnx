package operators

import (
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

// Step identifies a resolved node and value-table slots. Empty optional slots
// use -1. Gemm coefficients and Softmax axis contain resolved ONNX defaults.
// Constant shares an immutable tensor attribute without allocating numeric data.
type Step struct {
	Index          int
	Name           string
	Domain         string
	Operator       string
	Opset          int64
	SchemaVersion  int64
	DocString      string
	Metadata       []graph.Property
	Inputs         []int
	Outputs        []int
	Constant       *graph.Constant
	Gemm           tensor.GemmConfig
	Axis           int64
	Permutation    []int64
	PermutationSet bool
}

func (s Step) clone() (out Step) {
	out = s
	out.Metadata = slices.Clone(s.Metadata)
	out.Inputs = slices.Clone(s.Inputs)
	out.Outputs = slices.Clone(s.Outputs)
	out.Permutation = slices.Clone(s.Permutation)
	return out
}
