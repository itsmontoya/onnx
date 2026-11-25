package onnx

import (
	"errors"

	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

var (
	// ErrInvalidArgument identifies invalid public arguments or zero receivers.
	ErrInvalidArgument = runtimeerror.ErrInvalidArgument
	// ErrInvalidModel identifies contradictory graph or operator declarations.
	ErrInvalidModel = graph.ErrInvalidModel
	// ErrUnsupported identifies a feature outside an implemented capability.
	ErrUnsupported = graph.ErrUnsupported
	// ErrUnavailable identifies an opted-in execution device that is unavailable on this build or host.
	ErrUnavailable = errors.New("onnx: execution device unavailable")
	// ErrLimit identifies a configured load, session or run resource limit.
	ErrLimit = graph.ErrLimit
	// ErrBusy identifies an overlapping Run or Close on the same session.
	ErrBusy = runtimeerror.ErrBusy
	// ErrClosed identifies an operation requiring an open model, session or resolver.
	ErrClosed = runtimeerror.ErrClosed
)
