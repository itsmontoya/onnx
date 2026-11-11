package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
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
			name:   "signaling NaNs and payloads",
			values: []float32{math.Float32frombits(0x7f801234), math.Float32frombits(0xff805678), math.Float32frombits(0x7fffffff)},
			want:   []float32{math.Float32frombits(0x7f801234), math.Float32frombits(0xff805678), math.Float32frombits(0x7fffffff)},
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

			var actual, unchanged, supplied []float32
			if actual, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}

			if unchanged, err = input.Float32Values(); err != nil {
				t.Fatal(err)
			}

			if supplied, err = destination.Float32Values(); err != nil {
				t.Fatal(err)
			}

			for index, want := range tt.want {
				if math.Float32bits(actual[index]) != math.Float32bits(want) {
					t.Fatalf("output[%d] = %08x, want %08x", index, math.Float32bits(actual[index]), math.Float32bits(want))
				}

				if math.Float32bits(unchanged[index]) != math.Float32bits(tt.values[index]) {
					t.Fatalf("input[%d] changed", index)
				}

				wantDestination := math.Float32bits(want)
				if math.IsNaN(float64(want)) {
					wantDestination = 0
				}

				if math.Float32bits(supplied[index]) != wantDestination {
					t.Fatalf("destination[%d] = %08x, want %08x", index, math.Float32bits(supplied[index]), wantDestination)
				}
			}
			if !tt.destination {
				var replacement *tensor.Tensor
				values := make([]float32, len(tt.values))
				for index := range values {
					values[index] = 91
				}
				if replacement, err = tensor.FromFloat32(call.shape, values); err != nil {
					t.Fatal(err)
				}

				if err = tensor.ReLUInto(replacement, destination); err != nil {
					t.Fatal(err)
				}

				if actual, err = output.Float32Values(); err != nil {
					t.Fatal(err)
				}

				for index, want := range tt.want {
					if math.Float32bits(actual[index]) != math.Float32bits(want) {
						t.Fatalf("output[%d] changed after destination reuse", index)
					}
				}
			}
		})
	}
}

// TestNumericalCallReluBits checks the adapter against the original float-value
// rule, independently of the new backend's integer-bit classification.
func TestNumericalCallReluBits(t *testing.T) {
	var (
		bits   = []uint32{0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678, 0x7f801234, 0xff805678, 0x7fffffff, 0xffffffff, 0x3f800000, 0xbf800000, 1, 0x80000001}
		limits Limits
		err    error
	)

	if limits, err = (Limits{}).normalized(); err != nil {
		t.Fatal(err)
	}

	for _, count := range []int{0, 1, 2, 3, 7, 14, 31, 32, 33, 42, 1023, 1024, 1025, 3136, 6272, 8193} {
		for _, pattern := range []string{"Finite", "Mixed", "RandomBits", "AllNaN", "FirstNaN", "MiddleNaN", "LastNaN"} {
			t.Run(fmt.Sprintf("%d_%s", count, pattern), func(t *testing.T) {
				var (
					values                     = make([]float32, count)
					random                     = rand.New(rand.NewSource(int64(count)))
					input, destination, output *tensor.Tensor
					call                       *numericalCall
					actual, unchanged          []float32
					probeErr                   error
				)

				for index := range values {
					valueBits := math.Float32bits(float32(index%17) - 8)
					switch pattern {
					case "Mixed":
						valueBits = bits[index%len(bits)]
					case "RandomBits":
						valueBits = random.Uint32()
					case "AllNaN":
						valueBits = bits[4+index%6]
					}
					if pattern == "FirstNaN" && index == 0 || pattern == "MiddleNaN" && index == count/2 || pattern == "LastNaN" && index == count-1 {
						valueBits = 0xff801234
					}

					values[index] = math.Float32frombits(valueBits)
				}
				shape := []int64{int64(count)}
				if count == 1 {
					shape = nil
				} else if count == 0 && pattern == "Mixed" {
					shape = []int64{2, 0, 3}
				} else if count == 42 {
					shape = []int64{2, 3, 7}
				}

				if input, probeErr = tensor.FromFloat32(shape, values); probeErr != nil {
					t.Fatal(probeErr)
				}

				if destination, probeErr = tensor.New(Float32, input.Shape()); probeErr != nil {
					t.Fatal(probeErr)
				}

				if call, probeErr = newNumericalCall(operators.Step{Operator: "Relu"}, []*tensor.Tensor{input}, limits); probeErr != nil {
					t.Fatal(probeErr)
				}

				if output, probeErr = call.run(context.Background(), destination); probeErr != nil {
					t.Fatal(probeErr)
				}

				if actual, probeErr = output.Float32Values(); probeErr != nil {
					t.Fatal(probeErr)
				}

				if unchanged, probeErr = input.Float32Values(); probeErr != nil {
					t.Fatal(probeErr)
				}

				for index, value := range values {
					inputBits := math.Float32bits(value)
					var want uint32
					if value > 0 || math.IsNaN(float64(value)) {
						want = inputBits
					}

					if math.Float32bits(actual[index]) != want || math.Float32bits(unchanged[index]) != inputBits {
						t.Fatalf("index=%d input=%08x actual=%08x want=%08x unchanged=%08x", index, inputBits, math.Float32bits(actual[index]), want, math.Float32bits(unchanged[index]))
					}
				}
			})
		}
	}
}

// legacyReluCorrection retains the old copying repair as a test/benchmark oracle.
func legacyReluCorrection(ctx context.Context, source, output *tensor.Tensor, shape []int64) (out *tensor.Tensor, err error) {
	var input, values []float32
	if input, err = source.Float32Values(); err != nil {
		return nil, err
	}

	for index, value := range input {
		if index%1024 == 0 {
			if err = ctx.Err(); err != nil {
				return nil, err
			}
		}

		if math.IsNaN(float64(value)) {
			// Inputs without NaNs already have the correct destination result.
			// Only materialize a corrected copy when NaN propagation needs it.
			if values == nil {
				if values, err = output.Float32Values(); err != nil {
					return nil, err
				}
			}

			values[index] = value
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	if values == nil {
		return output, nil
	}

	return tensor.FromFloat32(shape, values)
}

func TestNumericalCallReluCancellationDestination(t *testing.T) {
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded} {
		for _, size := range []int{0, 1, 1023, 1024, 1025, 2049, 6272, 8193} {
			for _, nan := range []bool{false, true} {
				checks := 2
				if size > 0 {
					checks = 3 + (size+1023)/1024
				}

				for boundary := 1; boundary <= checks+1; boundary++ {
					t.Run(fmt.Sprintf("%v/%d/nan%t/check%d", cause, size, nan, boundary), func(t *testing.T) {
						var (
							values                = make([]float32, size)
							before                = make([]float32, size)
							input, output, legacy *tensor.Tensor
							actual, original      []float32
							want                  []float32
							call                  *numericalCall
							limits                Limits
							err                   error
						)

						for index := range values {
							values[index] = float32(index%17) - 8
							before[index] = 91
							if nan && index%7 == 0 {
								values[index] = math.Float32frombits(0xff801234)
							}
						}

						if input, err = tensor.FromFloat32([]int64{int64(size)}, values); err != nil {
							t.Fatal(err)
						}

						if output, err = tensor.FromFloat32(input.Shape(), before); err != nil {
							t.Fatal(err)
						}

						if legacy, err = tensor.New(Float32, input.Shape()); err != nil {
							t.Fatal(err)
						}

						if err = tensor.ReLUInto(input, legacy); err != nil {
							t.Fatal(err)
						}

						if want, err = legacy.Float32Values(); err != nil {
							t.Fatal(err)
						}

						if limits, err = (Limits{}).normalized(); err != nil {
							t.Fatal(err)
						}

						if call, err = newNumericalCall(operators.Step{Operator: "Relu"}, []*tensor.Tensor{input}, limits); err != nil {
							t.Fatal(err)
						}

						ctx := newCancelContext(boundary)
						defer ctx.cancel()
						var runContext context.Context = ctx
						if cause == context.DeadlineExceeded {
							runContext = newDeadlineContext(ctx)
						}

						result, runErr := call.run(runContext, output)
						if boundary <= checks {
							if result != nil || runErr != cause || errors.Is(runErr, ErrInvalidModel) || ctx.checks != boundary {
								t.Fatalf("checks=%d result=%v err=%v", ctx.checks, result, runErr)
							}
						} else {
							if runErr != nil || ctx.checks != checks || (result == output) != (!nan || size == 0) {
								t.Fatalf("checks=%d same=%t err=%v", ctx.checks, result == output, runErr)
							}

							var reference *tensor.Tensor
							if reference, err = legacyReluCorrection(context.Background(), input, legacy, call.shape); err != nil {
								t.Fatal(err)
							}

							checkNumericalReluBits(t, result, reference)
							// Reuse the supplied destination after success. Any NaN result
							// must own its storage independently of that destination.
							if result != output {
								if err = tensor.ReLUInto(legacy, output); err != nil {
									t.Fatal(err)
								}

								checkNumericalReluBits(t, result, reference)
							}
						}

						if boundary == 1 {
							want = before
						}

						if actual, err = output.Float32Values(); err != nil {
							t.Fatal(err)
						}

						if original, err = input.Float32Values(); err != nil {
							t.Fatal(err)
						}

						for index := range values {
							if math.Float32bits(actual[index]) != math.Float32bits(want[index]) || math.Float32bits(original[index]) != math.Float32bits(values[index]) {
								t.Fatalf("index=%d destination=%08x want=%08x input=%08x original=%08x", index, math.Float32bits(actual[index]), math.Float32bits(want[index]), math.Float32bits(original[index]), math.Float32bits(values[index]))
							}
						}
					})
				}
			}
		}
	}
}

func checkNumericalReluBits(t *testing.T, actual, expected *tensor.Tensor) {
	t.Helper()
	var (
		got, want []float32
		err       error
	)

	if got, err = actual.Float32Values(); err != nil {
		t.Fatal(err)
	}

	if want, err = expected.Float32Values(); err != nil {
		t.Fatal(err)
	}

	for index, value := range want {
		if math.Float32bits(got[index]) != math.Float32bits(value) {
			t.Fatalf("index=%d got=%08x want=%08x", index, math.Float32bits(got[index]), math.Float32bits(value))
		}
	}
}

func TestNumericalCallReluValidationPrecedence(t *testing.T) {
	var (
		input, wrongShape *tensor.Tensor
		limits            Limits
		call              *numericalCall
		err               error
	)

	if input, err = tensor.FromFloat32(nil, []float32{2}); err != nil {
		t.Fatal(err)
	}

	if wrongShape, err = tensor.FromFloat32([]int64{1}, []float32{91}); err != nil {
		t.Fatal(err)
	}

	if limits, err = (Limits{}).normalized(); err != nil {
		t.Fatal(err)
	}

	if call, err = newNumericalCall(operators.Step{Operator: "Relu"}, []*tensor.Tensor{input}, limits); err != nil {
		t.Fatal(err)
	}

	for _, boundary := range []int{1, 2} {
		ctx := newCancelContext(boundary)
		defer ctx.cancel()
		out, runErr := call.run(ctx, wrongShape)
		if out != nil || ctx.checks != 1 || boundary == 1 && runErr != context.Canceled || boundary == 2 && !errors.Is(runErr, ErrInvalidModel) {
			t.Fatalf("boundary=%d checks=%d out=%v err=%v", boundary, ctx.checks, out, runErr)
		}

		var values []float32
		if values, err = wrongShape.Float32Values(); err != nil {
			t.Fatal(err)
		}

		if values[0] != 91 {
			t.Fatal("validation changed destination")
		}
	}
}

func TestNumericalCallReluValidation(t *testing.T) {
	var (
		limits                              Limits
		input, wrongShape, wrongType, empty *tensor.Tensor
		call, emptyCall                     *numericalCall
		invalid                             tensor.Tensor
		err                                 error
	)

	if limits, err = (Limits{}).normalized(); err != nil {
		t.Fatal(err)
	}

	if input, err = tensor.FromFloat32([]int64{3}, []float32{-1, 2, math.Float32frombits(0xff801234)}); err != nil {
		t.Fatal(err)
	}

	if wrongShape, err = tensor.FromFloat32([]int64{1, 3}, []float32{91, 92, 93}); err != nil {
		t.Fatal(err)
	}

	if wrongType, err = tensor.FromInt64([]int64{3}, []int64{91, 92, 93}); err != nil {
		t.Fatal(err)
	}

	if empty, err = tensor.FromFloat32([]int64{2, 0, 3}, nil); err != nil {
		t.Fatal(err)
	}

	if call, err = newNumericalCall(operators.Step{Operator: "Relu"}, []*tensor.Tensor{input}, limits); err != nil {
		t.Fatal(err)
	}

	if emptyCall, err = newNumericalCall(operators.Step{Operator: "Relu"}, []*tensor.Tensor{empty}, limits); err != nil {
		t.Fatal(err)
	}

	shallow := *input
	emptyShallow := *empty
	tests := []struct {
		name        string
		call        *numericalCall
		destination *tensor.Tensor
	}{
		{"shape", call, wrongShape},
		{"dtype", call, wrongType},
		{"invalid", call, &invalid},
		{"alias", call, input},
		{"shallow alias", call, &shallow},
		{"empty alias", emptyCall, empty},
		{"empty shallow alias", emptyCall, &emptyShallow},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				before   []float32
				integers []int64
			)

			if tt.destination.DataType() == Float32 {
				if before, err = tt.destination.Float32Values(); err != nil {
					t.Fatal(err)
				}
			} else if tt.destination.DataType() == Int64 {
				if integers, err = tt.destination.Int64Values(); err != nil {
					t.Fatal(err)
				}
			}

			output, runErr := tt.call.run(context.Background(), tt.destination)
			if output != nil || !errors.Is(runErr, ErrInvalidModel) {
				t.Fatalf("output=%v error=%v", output, runErr)
			}

			if before != nil {
				var actual []float32
				if actual, err = tt.destination.Float32Values(); err != nil {
					t.Fatal(err)
				}

				for index, value := range before {
					if math.Float32bits(actual[index]) != math.Float32bits(value) {
						t.Fatal("destination changed on validation error")
					}
				}
			} else if integers != nil {
				var actual []int64
				if actual, err = tt.destination.Int64Values(); err != nil {
					t.Fatal(err)
				}

				for index, value := range integers {
					if actual[index] != value {
						t.Fatal("destination changed on validation error")
					}
				}
			}
		})
	}
}
