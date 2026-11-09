package graph

// Value identifies one named producer. Producer is -1 for an input/initializer
// and otherwise the declared node index. Declarations retain each input, output
// or intermediate constraint without narrowing a symbolic defaulted input to
// its initializer's current batch size. Default is immutable and may be shared.
type Value struct {
	Name         string
	Producer     int
	Input        bool
	Default      *Constant
	Declarations []ValueInfo
}

func (v Value) clone() (out Value) {
	out = v
	out.Declarations = make([]ValueInfo, len(v.Declarations))
	for index, declaration := range v.Declarations {
		out.Declarations[index] = declaration.clone()
	}
	return out
}
