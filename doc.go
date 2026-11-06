// Package onnx is the foundation for a Go-native ONNX inference runtime.
//
// The module currently provides its runtime contract and dependency verification.
// Loading, tensors, operator adapters, and sessions are not implemented yet.
// The planned initial profile is ONNX IR 8, standard-domain opset 13, float32
// computation, and int64 constants and shape values.
package onnx
