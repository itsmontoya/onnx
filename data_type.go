package onnx

import "github.com/itsmontoya/neuralnetwork/tensor"

// DataType identifies the concrete numeric storage of a Tensor.
type DataType = tensor.DataType

const (
	// Invalid identifies an uninitialized or unsupported tensor type.
	Invalid = tensor.Invalid
	// Float32 stores IEEE 754 binary32 values without conversion.
	Float32 = tensor.Float32
	// Int64 stores signed 64-bit integers without conversion.
	Int64 = tensor.Int64
)
