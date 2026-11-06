package graph

import (
	"errors"

	"github.com/itsmontoya/onnx/internal/decode"
)

var (
	// ErrInvalidModel identifies invalid graph fields, declarations or references.
	ErrInvalidModel = errors.New("onnx graph: invalid model")
	// ErrInvalidArgument identifies invalid builder arguments or limits.
	ErrInvalidArgument = decode.ErrInvalidArgument
	// ErrLimit identifies a graph or aggregate load resource limit.
	ErrLimit = decode.ErrLimit
	// ErrUnsupported identifies features outside the initial model envelope.
	ErrUnsupported = decode.ErrUnsupported
)
