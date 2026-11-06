package graph

import "slices"

// Annotation preserves tensor quantization labels as metadata only. It does not
// enable quantized tensor storage or quantized operator support.
type Annotation struct {
	Tensor     string
	Parameters []Property
}

func (a Annotation) clone() (out Annotation) {
	a.Parameters = slices.Clone(a.Parameters)
	return a
}
