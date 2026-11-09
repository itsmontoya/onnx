package onnx

import (
	"context"
	"errors"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func TestShapeCallExactWorkspaceAndLiveBounds(t *testing.T) {
	for _, fixture := range readM2Cases(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			data, err := fixture.Model.read()
			if err != nil {
				t.Fatal(err)
			}

			inputs, inputErr := fixture.readInputs()
			if inputErr != nil {
				t.Fatal(inputErr)
			}

			model := loadedModel(t, data, Options{})
			session := modelSession(t, model)
			step := session.plan.steps[0]
			arguments := make([]*tensor.Tensor, len(step.Inputs))
			var inputBytes int64
			for slot, index := range step.Inputs {
				arguments[slot] = inputs[session.plan.values[index].Name].value
				inputBytes += arguments[slot].ByteSize()
			}
			call, callErr := newShapeCall(step, arguments, session.plan.limits)
			if callErr != nil {
				t.Fatal(callErr)
			}

			if err = call.plan(context.Background()); err != nil {
				t.Fatal(err)
			}

			expectedScratch := int64(0)
			if step.Operator == "Shape" {
				expectedScratch = int64(arguments[0].Rank()) * 8
			} else if step.Operator == "Reshape" || step.Operator == "Unsqueeze" || step.Operator == "Squeeze" && len(arguments) > 1 {
				expectedScratch = arguments[1].ByteSize()
			}

			if call.workspace != expectedScratch {
				t.Fatalf("scratch %d, want %d", call.workspace, expectedScratch)
			}

			original := session.plan.limits
			peak := max(inputBytes+call.bytes+call.workspace, inputBytes+2*call.bytes)
			session.plan.limits.MaxWorkspaceBytes = call.workspace
			session.plan.limits.MaxLiveBytes = peak
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("exact budget: %v", err)
			}

			if call.workspace > 0 {
				session.plan.limits.MaxWorkspaceBytes--
				outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
				var diagnostic *Error
				if outputs != nil || !errors.Is(runErr, ErrLimit) || !errors.As(runErr, &diagnostic) || diagnostic.NodeIndex != 0 {
					t.Fatalf("pre-accessor scratch failure: %v, %v", outputs, runErr)
				}
			}

			session.plan.limits.MaxWorkspaceBytes = call.workspace
			if call.bytes+call.workspace > 0 {
				session.plan.limits.MaxLiveBytes = inputBytes + call.bytes + call.workspace - 1
				outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
				var diagnostic *Error
				if outputs != nil || !errors.Is(runErr, ErrLimit) || !errors.As(runErr, &diagnostic) || diagnostic.NodeIndex != 0 {
					t.Fatalf("combined destination/staging pre-allocation failure: %v, %v", outputs, runErr)
				}
			}

			session.plan.limits = original
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("recovery: %v", err)
			}
		})
	}
}
