package onnx

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
)

func TestOperationErrorClassificationAndJoinedCauses(t *testing.T) {
	for _, tt := range []struct {
		cause error
		kind  error
	}{
		{decode.ErrInvalidWire, ErrInvalidModel},
		{decode.ErrInvalidTensor, ErrInvalidModel},
		{decode.ErrInvalidArgument, ErrInvalidArgument},
		{ErrUnsupported, ErrUnsupported},
		{ErrLimit, ErrLimit},
	} {
		diagnostic := operationError("load", tt.cause, "x")
		if !errors.Is(diagnostic, tt.cause) || !errors.Is(diagnostic, tt.kind) || diagnostic.Value != "x" || !strings.HasPrefix(diagnostic.Error(), "onnx load") {
			t.Fatalf("classification: %v", diagnostic)
		}
	}

	prior := &Error{Stage: "compile", NodeIndex: 3, Operator: "Identity", Cause: ErrUnsupported, Kind: ErrUnsupported}
	closeCause := errors.New("file close failed")
	joined := errors.Join(fmt.Errorf("load model: %w", prior), closeCause)
	diagnostic := operationError("load", joined, "output")
	if !errors.Is(diagnostic, ErrUnsupported) || !errors.Is(diagnostic, closeCause) || diagnostic.Stage != "load" || diagnostic.NodeIndex != 3 || diagnostic.Operator != "Identity" || diagnostic.Value != "output" {
		t.Fatalf("joined causes/context lost: %v", diagnostic)
	}

	if prior.Stage != "compile" || prior.Value != "" {
		t.Fatal("wrapping changed the source diagnostic")
	}
}
