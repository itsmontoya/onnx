package operators

import (
	"slices"

	"github.com/itsmontoya/onnx/internal/graph"
)

// Step identifies a resolved node and value-table slots. Empty optional slots
// would use -1; the initial Identity/Constant subset has no optional slots.
// Constant shares an immutable tensor attribute without allocating numeric data.
type Step struct {
	Index         int
	Name          string
	Domain        string
	Operator      string
	Opset         int64
	SchemaVersion int64
	Inputs        []int
	Outputs       []int
	Constant      *graph.Constant
}

func (s Step) clone() (out Step) {
	out = s
	out.Inputs = slices.Clone(s.Inputs)
	out.Outputs = slices.Clone(s.Outputs)
	return out
}
