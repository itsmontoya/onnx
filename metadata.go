package onnx

import "slices"

// Metadata is a model description with inputs and outputs in declaration order.
// Model accessors return independent snapshots; modifying a snapshot must not
// alter the compiled model or another session's metadata.
type Metadata struct {
	Name      string
	Producer  string
	IRVersion int64
	Opset     int64
	Inputs    []ValueInfo
	Outputs   []ValueInfo
}

func (m Metadata) clone() (out Metadata) {
	out = m
	out.Inputs, out.Outputs = slices.Clone(m.Inputs), slices.Clone(m.Outputs)
	for index, info := range out.Inputs {
		out.Inputs[index].Dimensions = slices.Clone(info.Dimensions)
	}
	for index, info := range out.Outputs {
		out.Outputs[index].Dimensions = slices.Clone(info.Dimensions)
	}
	return out
}
