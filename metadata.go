package onnx

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
