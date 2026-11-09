package decode

import "errors"

var (
	// ErrInvalidArgument identifies invalid decoder options, contexts, or readers.
	ErrInvalidArgument = errors.New("onnx decode: invalid argument")
	// ErrLimit identifies input exceeding the configured serialized byte limit.
	ErrLimit = errors.New("onnx decode: byte limit exceeded")
	// ErrInvalidWire identifies malformed protobuf, including recursion failures.
	ErrInvalidWire = errors.New("onnx decode: invalid protobuf")
)
