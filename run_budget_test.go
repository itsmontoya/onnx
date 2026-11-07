package onnx

import (
	"errors"
	"math"
	"testing"
)

func TestRunBudgetReservations(t *testing.T) {
	limits := Limits{MaxTensorBytes: 32, MaxLiveBytes: 64, MaxWorkspaceBytes: 16}
	tests := []struct {
		name      string
		resident  int64
		value     int64
		workspace int64
		want      error
	}{
		{"exact live and workspace", 16, 32, 16, nil},
		{"live one over", 17, 32, 16, ErrLimit},
		{"workspace one over", 0, 1, 17, ErrLimit},
		{"tensor one over", 0, 33, 0, ErrLimit},
		{"resident one over", 65, 0, 0, ErrLimit},
		{"negative resident", -1, 0, 0, ErrLimit},
		{"negative bytes", 0, -1, 0, ErrLimit},
		{"negative workspace", 0, 0, -1, ErrLimit},
		{"overflow sized reservation", 1, math.MaxInt64, math.MaxInt64, ErrLimit},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			budget, err := newRunBudget(limits, tt.resident)
			if err == nil {
				err = budget.reserve(tt.value, tt.workspace)
			}

			if tt.want != nil {
				if !errors.Is(err, tt.want) {
					t.Fatalf("reservation: %v", err)
				}

				if budget != nil && (budget.live != tt.resident || budget.workspace != 0) {
					t.Fatal("failed reservation changed accounting")
				}

				return
			}

			if err != nil || budget.live != tt.resident+tt.value || budget.workspace != tt.workspace {
				t.Fatalf("successful reservation: %+v, %v", budget, err)
			}

			budget.releaseWorkspace()
			if budget.workspace != 0 || budget.live != tt.resident+tt.value {
				t.Fatal("workspace release changed live values")
			}
		})
	}
}

func TestRunBudgetCheckedSumsAndWorkspaceCoexistence(t *testing.T) {
	limits := Limits{MaxTensorBytes: math.MaxInt64, MaxLiveBytes: math.MaxInt64, MaxWorkspaceBytes: math.MaxInt64}
	budget, err := newRunBudget(limits, math.MaxInt64-1)
	if err != nil {
		t.Fatal(err)
	}

	if err = budget.reserve(1, 1); !errors.Is(err, ErrLimit) || budget.live != math.MaxInt64-1 {
		t.Fatalf("overflowing sum accepted: %+v, %v", budget, err)
	}

	budget, err = newRunBudget(Limits{MaxTensorBytes: 16, MaxLiveBytes: 32, MaxWorkspaceBytes: 12}, 8)
	if err != nil {
		t.Fatal(err)
	}

	if err = budget.reserve(8, 8); err != nil {
		t.Fatal(err)
	}

	if err = budget.reserve(0, 5); !errors.Is(err, ErrLimit) || budget.workspace != 8 {
		t.Fatalf("overlapping scratch ignored: %+v, %v", budget, err)
	}

	budget.releaseWorkspace()
	if err = budget.reserve(8, 8); err != nil || budget.live != 24 || budget.workspace != 8 {
		t.Fatalf("exact sum after scratch release: %+v, %v", budget, err)
	}
}
