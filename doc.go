// Package onnx is the foundation for a Go-native ONNX inference runtime.
//
// The module currently provides immutable typed tensors, model metadata records,
// bounded internal wire/tensor decoding, and named graph validation. Public loading,
// operators, and sessions are not implemented yet. Tensor values can be shared by
// concurrent readers; constructors and accessors copy, and no mutable backend
// storage is exposed.
// The planned initial profile is ONNX IR 8, standard-domain opset 13, float32
// computation, and int64 constants and shape values.
package onnx
