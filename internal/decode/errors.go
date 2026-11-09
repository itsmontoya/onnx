package decode

import "errors"

var (
	// ErrInvalidArgument identifies invalid decoder options, contexts, or readers.
	ErrInvalidArgument = errors.New("onnx decode: invalid argument")
	// ErrLimit identifies input or tensor storage exceeding a configured resource limit.
	ErrLimit = errors.New("onnx decode: resource limit exceeded")
	// ErrInvalidWire identifies malformed protobuf, including recursion failures.
	ErrInvalidWire = errors.New("onnx decode: invalid protobuf")
	// ErrInvalidTensor identifies invalid dimensions or payload encodings.
	ErrInvalidTensor = errors.New("onnx decode: invalid tensor")
	// ErrUnsupported identifies tensor types or storage features outside the profile.
	ErrUnsupported = errors.New("onnx decode: unsupported tensor")
)
