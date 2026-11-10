package onnx

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"github.com/itsmontoya/onnx/internal/operators"
)

func numericalInputs(t *testing.T, wire *onnxpb.ModelProto) (inputs map[string]*Tensor) {
	t.Helper()
	inputs = make(map[string]*Tensor)
	for _, input := range wire.Graph.Input {
		shape := make([]int64, len(input.Type.GetTensorType().Shape.Dim))
		for axis, dimension := range input.Type.GetTensorType().Shape.Dim {
			shape[axis] = dimension.GetDimValue()
		}
		count, _, err := tensor.ValidateShape(shape, Float32)
		if err != nil {
			t.Fatal(err)
		}

		values := make([]float32, int(count))
		for index := range values {
			values[index] = 1
		}
		inputs[input.GetName()] = floatTensor(t, shape, values)
	}
	return inputs
}

func TestNumericalCallExactWorkspaceAndLiveBounds(t *testing.T) {
	tests := []struct {
		fixture   string
		workspace int64
	}{
		{"add_matching", 48}, {"sub_matching", 48}, {"mul_matching", 48}, {"div_matching", 0},
		{"add_broadcast", 0}, {"matmul_batch", 384}, {"matmul_dot", 28},
		{"matmul_empty_contraction", 0}, {"matmul_empty_output", 0},
		{"gemm_00_vector", 88}, {"gemm_11_full", 112}, {"gemm_empty_contraction", 24},
		{"relu_finite", 120}, {"relu_empty", 0}, {"sigmoid_finite", 60}, {"tanh_finite", 60},
		{"softmax_axis_1", 0}, {"softmax_empty_axis", 0},
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			wire := modelWire(t, "m1/"+tt.fixture)
			inputs := numericalInputs(t, wire)
			model := loadedModel(t, modelBytes(t, wire), Options{})
			session := modelSession(t, model)
			step := session.plan.steps[0]
			arguments := make([]*tensor.Tensor, len(step.Inputs))
			var inputBytes int64
			for slot, index := range step.Inputs {
				if index >= 0 {
					arguments[slot] = inputs[session.plan.values[index].Name].value
				}
			}
			for _, input := range inputs {
				inputBytes += input.ByteSize()
			}
			workspace := tt.workspace
			if !copyingProductWorkspace {
				switch tt.fixture {
				case "matmul_batch":
					workspace = 64
				case "matmul_dot":
					workspace = 4
				case "gemm_00_vector", "gemm_11_full":
					workspace = 24
				}
			}
			call, err := newNumericalCall(step, arguments, session.plan.limits)
			if err != nil || call.workspace != workspace {
				t.Fatalf("workspace = %v, want %d: %v", call, workspace, err)
			}

			original := session.plan.limits
			session.plan.limits.MaxWorkspaceBytes = workspace
			peak := max(inputBytes+call.bytes+workspace, inputBytes+2*call.bytes)
			session.plan.limits.MaxLiveBytes = peak
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("exact budget: %v", err)
			}

			if workspace > 0 {
				session.plan.limits.MaxWorkspaceBytes--
				outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
				var diagnostic *Error
				if outputs != nil || !errors.Is(runErr, ErrLimit) || !errors.As(runErr, &diagnostic) || diagnostic.NodeIndex != 0 {
					t.Fatalf("workspace bound: %v, %v", outputs, runErr)
				}
			}

			session.plan.limits.MaxWorkspaceBytes = workspace
			if peak > 0 {
				session.plan.limits.MaxLiveBytes = peak - 1
				if outputs, runErr := session.Run(context.Background(), inputs, RunOptions{}); outputs != nil || !errors.Is(runErr, ErrLimit) {
					t.Fatalf("live bound: %v, %v", outputs, runErr)
				}
			}

			session.plan.limits = original
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("recovery: %v", err)
			}
		})
	}
}

func TestNumericalScratchOverflow(t *testing.T) {
	if _, err := checkedScratch(math.MaxInt64, 2); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}

	if _, err := sumScratch(math.MaxInt64, 1); !errors.Is(err, ErrLimit) {
		t.Fatal(err)
	}
}

func TestNumericalCallReluDestination(t *testing.T) {
	tests := []struct {
		name        string
		values      []float32
		want        []float32
		destination bool
	}{
		{
			name:        "finite",
			values:      []float32{-2, 0, 3},
			want:        []float32{0, 0, 3},
			destination: true,
		},
		{
			name:        "infinities and negative zero",
			values:      []float32{float32(math.Inf(-1)), float32(math.Inf(1)), math.Float32frombits(0x80000000)},
			want:        []float32{0, float32(math.Inf(1)), 0},
			destination: true,
		},
		{
			name:   "multiple NaNs",
			values: []float32{math.Float32frombits(0x7fc00001), -2, math.Float32frombits(0xffc00002), 3},
			want:   []float32{math.Float32frombits(0x7fc00001), 0, math.Float32frombits(0xffc00002), 3},
		},
		{
			name:        "empty",
			destination: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				limits              Limits
				call                *numericalCall
				destination, output *tensor.Tensor
				err                 error
			)

			input := floatTensor(t, []int64{int64(len(tt.values))}, tt.values)
			if limits, err = (Limits{}).normalized(); err != nil {
				t.Fatal(err)
			}

			if call, err = newNumericalCall(operators.Step{Operator: "Relu"}, []*tensor.Tensor{input.value}, limits); err != nil {
				t.Fatal(err)
			}

			if destination, err = tensor.New(Float32, call.shape); err != nil {
				t.Fatal(err)
			}

			if output, err = call.run(context.Background(), destination); err != nil {
				t.Fatal(err)
			}

			if (output == destination) != tt.destination {
				t.Fatalf("returned destination = %t, want %t", output == destination, tt.destination)
			}

			var actual, unchanged []float32
			if actual, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}

			if unchanged, err = input.Float32Values(); err != nil {
				t.Fatal(err)
			}

			for index, want := range tt.want {
				if math.Float32bits(actual[index]) != math.Float32bits(want) {
					t.Fatalf("output[%d] = %08x, want %08x", index, math.Float32bits(actual[index]), math.Float32bits(want))
				}

				if math.Float32bits(unchanged[index]) != math.Float32bits(tt.values[index]) {
					t.Fatalf("input[%d] changed", index)
				}
			}
		})
	}
}
