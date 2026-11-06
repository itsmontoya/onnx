package onnx

import "errors"

// ErrInvalidArgument identifies invalid tensor arguments or receivers.
var ErrInvalidArgument = errors.New("onnx: invalid argument")
