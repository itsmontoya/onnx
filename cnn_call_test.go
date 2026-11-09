package onnx

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func TestCNNCallExactLiveBounds(t *testing.T) {
	for _, fixture := range readM3Cases(t) {
		if strings.HasPrefix(fixture.Name, "cnn_") {
			continue
		}

		t.Run(fixture.Name, func(t *testing.T) {
			data, err := fixture.Model.read()
			if err != nil {
				t.Fatal(err)
			}

			inputs, inputErr := fixture.readInputs()
			if inputErr != nil {
				t.Fatal(inputErr)
			}

			session := modelSession(t, loadedModel(t, data, Options{}))
			step := session.plan.steps[0]
			arguments := make([]*tensor.Tensor, len(step.Inputs))
			var inputBytes int64
			for slot, index := range step.Inputs {
				arguments[slot] = inputs[session.plan.values[index].Name].value
				inputBytes += arguments[slot].ByteSize()
			}
			call, callErr := newCNNCall(step, arguments, session.plan.limits)
			if callErr != nil {
				t.Fatal(callErr)
			}

			original := session.plan.limits
			session.plan.limits.MaxWorkspaceBytes = 0
			session.plan.limits.MaxLiveBytes = inputBytes + 2*call.bytes
			outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatalf("exact destination/output bounds with zero scratch: %v", runErr)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatal(err)
			}

			if call.bytes > 0 {
				session.plan.limits.MaxLiveBytes = inputBytes + call.bytes - 1
				failed, limitErr := session.Run(context.Background(), inputs, RunOptions{})
				var diagnostic *Error
				if failed != nil || !errors.Is(limitErr, ErrLimit) || !errors.As(limitErr, &diagnostic) || diagnostic.NodeIndex != 0 {
					t.Fatalf("destination rejected before allocation: %v, %v", failed, limitErr)
				}

				session.plan.limits.MaxLiveBytes = inputBytes + 2*call.bytes - 1
				if failed, limitErr = session.Run(context.Background(), inputs, RunOptions{}); failed != nil || !errors.Is(limitErr, ErrLimit) {
					t.Fatalf("publication copy rejected: %v, %v", failed, limitErr)
				}
			}

			session.plan.limits = original
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("budget failure recovery: %v", err)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatalf("later failures changed retained output: %v", err)
			}
		})
	}
}
