package onnx

// Dimension describes one model dimension, separately from concrete tensor data.
// Value applies only to FixedDimension; Symbol applies only to SymbolicDimension.
// An unknown dimension and a fixed zero dimension remain distinct.
type Dimension struct {
	Kind   DimensionKind
	Value  int64
	Symbol string
}
