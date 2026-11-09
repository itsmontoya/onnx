package graph

import "slices"

// Attribute snapshots one named payload without retaining wire messages.
// Empty lists are distinguished by Kind, since repeated protobuf fields have no
// presence. Tensor constants expose only copied numerical access.
type Attribute struct {
	Name      string
	DocString string
	Kind      AttributeKind
	Float     float32
	Int       int64
	String    []byte
	Tensor    *Constant
	Floats    []float32
	Ints      []int64
	Strings   [][]byte
	Tensors   []*Constant
}

func (a Attribute) clone() (out Attribute) {
	out = a
	out.String = slices.Clone(a.String)
	out.Floats = slices.Clone(a.Floats)
	out.Ints = slices.Clone(a.Ints)
	out.Tensors = slices.Clone(a.Tensors)
	out.Strings = make([][]byte, len(a.Strings))
	for index, value := range a.Strings {
		out.Strings[index] = slices.Clone(value)
	}
	return out
}
