package onnx

// ValueInfo is a metadata snapshot for a named model input or output.
// ShapeKnown distinguishes an unknown rank from a known rank-zero scalar.
// Dimensions describes each axis when ShapeKnown is true. Defaulted identifies
// a graph input whose initializer supplies a value when the caller omits it.
type ValueInfo struct {
	Name       string
	DataType   DataType
	ShapeKnown bool
	Dimensions []Dimension
	Defaulted  bool
}
