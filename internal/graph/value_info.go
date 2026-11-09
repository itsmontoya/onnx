package graph

import (
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// ValueInfo is one declaration of tensor metadata, independently of its producer.
// DataType Invalid means absent intermediate type information. ShapeKnown
// distinguishes an absent rank from a scalar. Defaulted applies to graph inputs.
type ValueInfo struct {
	Name       string
	DataType   tensor.DataType
	ShapeKnown bool
	Dimensions []Dimension
	Defaulted  bool
	DocString  string
	Denotation string
	Metadata   []Property
}

func (v ValueInfo) clone() (out ValueInfo) {
	v.Dimensions = slices.Clone(v.Dimensions)
	v.Metadata = slices.Clone(v.Metadata)
	return v
}
