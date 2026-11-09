package graph

// Dimension keeps fixed zero, a symbol and an unspecified dimension distinct.
type Dimension struct {
	Kind       DimensionKind
	Value      int64
	Symbol     string
	Denotation string
}
