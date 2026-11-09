package onnx

import (
	"errors"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

// Error describes a load, compile or run failure and preserves its cause.
type Error = runtimeerror.Error

func operationError(stage string, cause error, value string) (out *Error) {
	var diagnostic Error
	diagnostic.Stage, diagnostic.NodeIndex = stage, -1
	diagnostic.Cause, diagnostic.Value = cause, value
	var prior *Error
	if errors.As(cause, &prior) {
		diagnostic = *prior
		diagnostic.Stage = stage
		if _, direct := cause.(*Error); !direct {
			diagnostic.Cause = cause
		}

		if value != "" {
			diagnostic.Value = value
		}

		return &diagnostic
	}

	switch {
	case errors.Is(cause, ErrInvalidArgument), errors.Is(cause, decode.ErrInvalidArgument):
		diagnostic.Kind = ErrInvalidArgument
	case errors.Is(cause, ErrLimit):
		diagnostic.Kind = ErrLimit
	case errors.Is(cause, ErrUnsupported):
		diagnostic.Kind = ErrUnsupported
	case errors.Is(cause, ErrInvalidModel), errors.Is(cause, decode.ErrInvalidWire), errors.Is(cause, decode.ErrInvalidTensor):
		diagnostic.Kind = ErrInvalidModel
	}
	return &diagnostic
}
