package operators

import (
	"slices"

	"github.com/itsmontoya/onnx/internal/graph"
)

// Value carries inferred metadata and declarations that must also hold at Run.
// Deferred is true when its shape cannot yet be expressed as concrete sizes.
// Defaults are immutable graph constants; input defaults do not fix batch sizes.
type Value struct {
	Name         string
	Producer     int
	Input        bool
	Default      *graph.Constant
	Info         graph.ValueInfo
	Declarations []graph.ValueInfo
	Deferred     bool
}

func (v Value) clone() (out Value) {
	out = v
	out.Info = cloneInfo(v.Info)
	out.Declarations = make([]graph.ValueInfo, len(v.Declarations))
	for index, declaration := range v.Declarations {
		out.Declarations[index] = cloneInfo(declaration)
	}
	return out
}

func cloneInfo(info graph.ValueInfo) (out graph.ValueInfo) {
	out = info
	out.Dimensions = slices.Clone(info.Dimensions)
	out.Metadata = slices.Clone(info.Metadata)
	return out
}
