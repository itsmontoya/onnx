package onnx

// NamedTensor is an independently owned immutable result for a declared output.
type NamedTensor struct {
	Name   string
	Tensor *Tensor
}
