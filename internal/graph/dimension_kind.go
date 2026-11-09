package graph

// DimensionKind distinguishes unknown, fixed and shared symbolic metadata.
type DimensionKind uint8

const (
	// UnknownDimension has no fixed size or shared binding.
	UnknownDimension DimensionKind = iota
	// FixedDimension describes a nonnegative size, including zero.
	FixedDimension
	// SymbolicDimension binds its nonempty symbol during a future run.
	SymbolicDimension
)
