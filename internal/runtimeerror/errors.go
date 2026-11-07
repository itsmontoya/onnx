package runtimeerror

import "errors"

var (
	// ErrInvalidArgument identifies invalid public or planner arguments.
	ErrInvalidArgument = errors.New("onnx: invalid argument")
	// ErrBusy identifies an overlapping session operation.
	ErrBusy = errors.New("onnx: session is busy")
	// ErrClosed identifies an operation on a closed model or session.
	ErrClosed = errors.New("onnx: closed")
)
