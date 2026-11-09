package onnx

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/operators"
)

func TestNormalizationCallReservations(t *testing.T) {
	limits, err := (Limits{}).normalized()
	if err != nil {
		t.Fatal(err)
	}

	inputs := []*tensor.Tensor{floatTensor(t, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}).value,
		floatTensor(t, []int64{3}, []float32{1, 1, 1}).value}
	call, err := newNormalizationCall(operators.Step{Axis: -1, Epsilon: 1e-5}, inputs, limits)
	if err != nil {
		t.Fatal(err)
	}

	if call.workspace != 624 || len(call.bytes) != 3 || call.bytes[0] != 24 || call.bytes[1] != 8 || call.bytes[2] != 8 {
		t.Fatalf("normalization reservations: %+v", call)
	}

	for _, tt := range []struct {
		name            string
		workspace, live int64
		want            error
	}{
		{name: "exact", workspace: 624, live: 700},
		{name: "workspace_short", workspace: 623, live: 700, want: ErrLimit},
		{name: "combined_live_short", workspace: 624, live: 699, want: ErrLimit},
	} {
		t.Run(tt.name, func(t *testing.T) {
			bounded := limits
			bounded.MaxWorkspaceBytes, bounded.MaxLiveBytes = tt.workspace, tt.live
			budget, budgetErr := newRunBudget(bounded, 36)
			if budgetErr != nil {
				t.Fatal(budgetErr)
			}

			budgetErr = budget.reserve(0, call.workspace)
			for _, bytes := range call.bytes {
				if budgetErr == nil {
					budgetErr = budget.reserve(bytes, 0)
				}
			}
			if !errors.Is(budgetErr, tt.want) {
				t.Fatalf("reservation = %v, want %v", budgetErr, tt.want)
			}
		})
	}
	empty := []*tensor.Tensor{floatTensor(t, []int64{0, 3}, nil).value, inputs[1]}
	emptyCall, emptyErr := newNormalizationCall(operators.Step{Axis: -1, Epsilon: 1e-5}, empty, limits)
	if emptyErr != nil || emptyCall.workspace != 16 {
		t.Fatalf("empty-prefix scalar staging reservation = %v, %v", emptyCall, emptyErr)
	}

	if _, err = checkedScratch(math.MaxInt64, 20); !errors.Is(err, ErrLimit) {
		t.Fatalf("scratch overflow = %v", err)
	}
}

func TestNormalizationCallCancellation(t *testing.T) {
	limits, err := (Limits{}).normalized()
	if err != nil {
		t.Fatal(err)
	}

	inputs := []*tensor.Tensor{floatTensor(t, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}).value,
		floatTensor(t, []int64{3}, []float32{1, 1, 1}).value}
	call, err := newNormalizationCall(operators.Step{Axis: -1, Epsilon: 1e-5}, inputs, limits)
	if err != nil {
		t.Fatal(err)
	}

	for _, check := range []int{1, 11, 13, 18, 20, 21} {
		ctx := newCancelContext(check)
		outputs, runErr := call.run(ctx)
		ctx.cancel()
		if outputs != nil || !errors.Is(runErr, context.Canceled) {
			t.Fatalf("cancellation check %d = %v, %v", check, outputs, runErr)
		}
	}

	if outputs, runErr := call.run(context.Background()); len(outputs) != 3 || runErr != nil {
		t.Fatalf("recovered call = %v, %v", outputs, runErr)
	}
}
