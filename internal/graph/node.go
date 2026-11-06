package graph

import "slices"

// Node preserves declared order, repeated labels and positional empty slots.
// Its presence in a structural graph does not imply operator support.
type Node struct {
	Index      int
	Name       string
	Domain     string
	Operator   string
	Inputs     []string
	Outputs    []string
	Attributes []Attribute
	DocString  string
	Metadata   []Property
}

func (n Node) clone() (out Node) {
	out = n
	out.Inputs = slices.Clone(n.Inputs)
	out.Outputs = slices.Clone(n.Outputs)
	out.Metadata = slices.Clone(n.Metadata)
	out.Attributes = make([]Attribute, len(n.Attributes))
	for index, attribute := range n.Attributes {
		out.Attributes[index] = attribute.clone()
	}
	return out
}
