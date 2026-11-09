package operators

import (
	"errors"
	"slices"
	"testing"

	"github.com/itsmontoya/onnx/internal/graph"
)

func TestResolveReshape(t *testing.T) {
	tests := []struct {
		name    string
		input   []int64
		target  []int64
		want    []int64
		invalid bool
	}{
		{"copy and infer", []int64{2, 3, 4}, []int64{0, -1}, []int64{2, 12}, false},
		{"scalar", []int64{1}, nil, nil, false},
		{"from scalar", nil, []int64{1, -1}, []int64{1, 1}, false},
		{"inferred zero", []int64{0, 3}, []int64{-1, 3}, []int64{0, 3}, false},
		{"copied zero", []int64{0, 3}, []int64{0, 3}, []int64{0, 3}, false},
		{"ambiguous zero", []int64{0, 3}, []int64{0, -1}, nil, true},
		{"multiple inference", []int64{2, 3}, []int64{-1, -1}, nil, true},
		{"negative", []int64{2, 3}, []int64{-2, 3}, nil, true},
		{"zero outside rank", []int64{2, 3}, []int64{1, 1, 0}, nil, true},
		{"fractional", []int64{2, 3}, []int64{-1, 4}, nil, true},
		{"wrong count", []int64{2, 3}, []int64{3, 3}, nil, true},
		{"overflow", []int64{2, 3}, []int64{1 << 62, 4}, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			shape, err := ResolveReshape(tt.input, tt.target)
			if tt.invalid {
				if !errors.Is(err, graph.ErrInvalidModel) {
					t.Fatalf("expected invalid shape: %v, %v", shape, err)
				}

				return
			}

			if err != nil || !slices.Equal(shape, tt.want) {
				t.Fatalf("shape: %v, %v; want %v", shape, err, tt.want)
			}
		})
	}
}
