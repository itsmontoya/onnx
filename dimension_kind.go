package onnx

// DimensionKind distinguishes fixed, symbolic, and unknown model metadata.
// It does not describe the concrete dimensions stored in a Tensor.
type DimensionKind uint8

const (
	// UnknownDimension has no fixed size or shared symbolic binding.
	UnknownDimension DimensionKind = iota
	// FixedDimension has a nonnegative size, including a concrete zero.
	FixedDimension
	// SymbolicDimension binds its nonempty symbol to a concrete size for each run.
	SymbolicDimension
)
