package runtimeerror

import "errors"

// ErrInvalidArgument identifies invalid public or planner arguments.
var ErrInvalidArgument = errors.New("onnx: invalid argument")
