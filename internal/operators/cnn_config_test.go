package operators

import (
	"errors"
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

func TestCNNConfigAutomaticPadding(t *testing.T) {
	tests := []struct {
		name      string
		input     [2]int64
		kernel    [2]int64
		strides   [2]int64
		dilations [2]int64
		want      [4]int64
		invalid   bool
	}{
		{"SAME_UPPER", [2]int64{4, 5}, [2]int64{3, 2}, [2]int64{2, 2}, [2]int64{1, 1}, [4]int64{0, 0, 1, 1}, false},
		{"SAME_LOWER", [2]int64{4, 5}, [2]int64{3, 2}, [2]int64{2, 2}, [2]int64{1, 1}, [4]int64{1, 1, 0, 0}, false},
		{"SAME_UPPER", [2]int64{0, 0}, [2]int64{3, 2}, [2]int64{2, 2}, [2]int64{1, 1}, [4]int64{}, false},
		{"SAME_UPPER", [2]int64{math.MaxInt64, 1}, [2]int64{1, 1}, [2]int64{math.MaxInt64, 1}, [2]int64{1, 1}, [4]int64{}, false},
		{"SAME_UPPER", [2]int64{4, 5}, [2]int64{3, 2}, [2]int64{2, 2}, [2]int64{math.MaxInt64, 1}, [4]int64{}, true},
		{"SAME_UPPER", [2]int64{math.MaxInt64, 5}, [2]int64{3, 2}, [2]int64{2, 2}, [2]int64{1, 1}, [4]int64{}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := CNNConfig{AutoPad: tt.name}
			pads, err := config.Pads(tt.input, tt.kernel, tt.strides, tt.dilations)
			if tt.invalid {
				if !errors.Is(err, graph.ErrInvalidModel) {
					t.Fatal(err)
				}

				return
			}

			if err != nil || pads != tt.want {
				t.Fatalf("Pads = %v, %v; want %v", pads, err, tt.want)
			}
		})
	}
}

func TestCNNConfigAverageKernelAreaBeforeAllocation(t *testing.T) {
	config := CNNConfig{
		AutoPad: "NOTSET",
		Conv:    tensor.Conv2DConfig{Strides: [2]int64{math.MaxInt64, math.MaxInt64}, Dilations: [2]int64{1, 1}, Pads: [4]int64{1 << 32, 1 << 32, 0, 0}, Groups: 1},
		Pool:    tensor.Pool2DConfig{Kernel: [2]int64{1 << 32, 1 << 32}, Strides: [2]int64{math.MaxInt64, math.MaxInt64}, Dilations: [2]int64{1, 1}, Pads: [4]int64{1 << 32, 1 << 32, 0, 0}},
	}
	if _, _, _, err := config.Shape("AveragePool", [][]int64{{1, 1, 1, 1}}); !errors.Is(err, graph.ErrInvalidModel) {
		t.Fatalf("kernel area overflow: %v", err)
	}

	if _, _, _, err := config.Shape("AveragePool", [][]int64{{0, 1, 1, 1}}); err != nil {
		t.Fatalf("unused empty kernel area: %v", err)
	}
}
