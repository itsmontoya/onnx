package onnx

// RunOptions selects declared outputs in request order. Nil or empty Outputs
// selects all declared outputs in declaration order; duplicate names fail.
type RunOptions struct {
	Outputs []string
}
