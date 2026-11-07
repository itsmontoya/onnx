package runtimeerror

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func TestErrorClassificationAndCause(t *testing.T) {
	diagnostic := &Error{Kind: ErrInvalidArgument, NodeIndex: 2, NodeName: "copy", Operator: "Identity", Opset: 13, Value: "x", Shapes: [][]int64{nil, {}, {0, 3}}, Cause: fmt.Errorf("interrupted: %w", context.Canceled)}
	wrapped := fmt.Errorf("load model: %w", diagnostic)
	var found *Error
	if !errors.Is(wrapped, ErrInvalidArgument) || !errors.Is(wrapped, context.Canceled) || !errors.As(wrapped, &found) || found != diagnostic {
		t.Fatalf("classification or cause lost: %v", wrapped)
	}

	for _, text := range []string{`node[2] name "copy" domain "" operator "Identity" opset 13`, `value "x"`, "input shapes [unknown, [], [0 3]]", "interrupted: context canceled"} {
		if !strings.Contains(wrapped.Error(), text) {
			t.Fatalf("missing %q in %v", text, wrapped)
		}
	}

	if (&Error{Kind: ErrInvalidArgument, NodeIndex: -1}).Error() != "onnx compile: onnx: invalid argument" {
		t.Fatal("classification-only message")
	}

	for _, stage := range []string{"load", "run"} {
		if message := (&Error{Stage: stage, Kind: ErrInvalidArgument, NodeIndex: -1}).Error(); message != "onnx "+stage+": onnx: invalid argument" {
			t.Fatalf("stage %q: %s", stage, message)
		}
	}

	var absent *Error
	if absent.Unwrap() != nil || absent.Is(ErrInvalidArgument) || absent.Error() != "onnx: <nil error>" {
		t.Fatal("nil diagnostic receiver")
	}
}
