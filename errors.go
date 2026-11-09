package onnx

import (
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

var (
	// ErrInvalidArgument identifies invalid tensor or compilation arguments.
	ErrInvalidArgument = runtimeerror.ErrInvalidArgument
	// ErrInvalidModel identifies contradictory graph or operator declarations.
	ErrInvalidModel = graph.ErrInvalidModel
	// ErrUnsupported identifies a feature outside an implemented capability.
	ErrUnsupported = graph.ErrUnsupported
	// ErrLimit identifies a configured load or tensor resource limit.
	ErrLimit = graph.ErrLimit
)
