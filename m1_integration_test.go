package onnx

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestM1ReferenceGraphs(t *testing.T) {
	type fixtureValue struct {
		Name     string   `json:"name"`
		Shape    []int64  `json:"shape"`
		Dtype    string   `json:"dtype"`
		Bits     []uint32 `json:"bits"`
		Integers []int64  `json:"integers"`
	}
	var manifest struct {
		Cases []struct {
			Name            string         `json:"name"`
			Model           string         `json:"model"`
			SHA256          string         `json:"sha256"`
			ReferenceStatus string         `json:"reference_status"`
			ReferenceError  string         `json:"reference_error"`
			Inputs          []fixtureValue `json:"inputs"`
			Outputs         []fixtureValue `json:"outputs"`
		} `json:"cases"`
	}
	data, err := os.ReadFile("testdata/m1/provenance.json")
	if err != nil {
		t.Fatal(err)
	}

	if err = json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}

	if len(manifest.Cases) != 81 {
		t.Fatalf("fixture count: %d", len(manifest.Cases))
	}

	for _, fixture := range manifest.Cases {
		t.Run(fixture.Name, func(t *testing.T) {
			wire, readErr := os.ReadFile("testdata/m1/" + fixture.Model)
			if readErr != nil {
				t.Fatal(readErr)
			}

			digest := sha256.Sum256(wire)
			if hex.EncodeToString(digest[:]) != fixture.SHA256 {
				t.Fatal("model hash mismatch")
			}

			if fixture.ReferenceStatus != "agreement" && (fixture.Name != "softmax_empty_axis" || fixture.ReferenceStatus != "extension_reference_rejected" || fixture.ReferenceError == "") {
				t.Fatalf("unrecognized reference result: %s", fixture.ReferenceStatus)
			}

			inputs := make(map[string]*Tensor)
			for _, input := range fixture.Inputs {
				values := make([]float32, len(input.Bits))
				for index, bits := range input.Bits {
					values[index] = math.Float32frombits(bits)
				}
				inputs[input.Name] = floatTensor(t, input.Shape, values)
			}
			model := loadedModel(t, wire, Options{})
			session := modelSession(t, model)
			outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil || len(outputs) != len(fixture.Outputs) {
				t.Fatalf("Run: %v, %v", outputs, runErr)
			}

			for outputIndex, expected := range fixture.Outputs {
				output := outputs[outputIndex]
				if output.Name != expected.Name || !slices.Equal(output.Tensor.Shape(), expected.Shape) {
					t.Fatalf("output %q shape %v, want %q %v", output.Name, output.Tensor.Shape(), expected.Name, expected.Shape)
				}

				if expected.Dtype == "int64" {
					integers, valueErr := output.Tensor.Int64Values()
					if valueErr != nil || !slices.Equal(integers, expected.Integers) {
						t.Fatalf("output %q integer mismatch: %v, %v", output.Name, integers, valueErr)
					}

					continue
				}

				actual, valueErr := output.Tensor.Float32Values()
				if valueErr != nil || len(actual) != len(expected.Bits) {
					t.Fatalf("output %q values: %v, %v", output.Name, actual, valueErr)
				}

				for index, bits := range expected.Bits {
					got, want := float64(actual[index]), float64(math.Float32frombits(bits))
					switch {
					case math.IsNaN(want):
						if !math.IsNaN(got) {
							t.Fatalf("output %q[%d] = %v, want NaN", output.Name, index, got)
						}
					case math.IsInf(want, 0):
						if got != want {
							t.Fatalf("output %q[%d] = %v, want %v", output.Name, index, got, want)
						}
					default:
						if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 1e-5+1e-5*math.Abs(want) {
							t.Fatalf("output %q[%d] = %v, want %v", output.Name, index, got, want)
						}

						if want == 0 && !strings.HasPrefix(fixture.Name, "matmul_") && !strings.HasPrefix(fixture.Name, "gemm_") && math.Signbit(got) != math.Signbit(want) {
							t.Fatalf("output %q[%d] zero sign mismatch", output.Name, index)
						}
					}
				}
			}
			// Retained outputs and all caller input bits survive another invocation.
			if _, runErr = session.Run(context.Background(), inputs, RunOptions{}); runErr != nil {
				t.Fatal(runErr)
			}

			for _, input := range fixture.Inputs {
				actual, valueErr := inputs[input.Name].Float32Values()
				if valueErr != nil {
					t.Fatal(valueErr)
				}

				for index, bits := range input.Bits {
					if math.Float32bits(actual[index]) != bits {
						t.Fatalf("input %q[%d] changed", input.Name, index)
					}
				}
			}
		})
	}
}

func TestM1RejectsStaticInvalidNodes(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		mutate  func(*onnxpb.ModelProto)
		want    error
	}{
		{"broadcast", "add_broadcast", func(wire *onnxpb.ModelProto) {
			wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 3}
		}, ErrInvalidModel},
		{"integer arithmetic", "add_scalar", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().ElemType = proto.Int32(7) }, ErrUnsupported},
		{"matmul scalar", "matmul_dot", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"matmul contraction", "matmul_dot", func(wire *onnxpb.ModelProto) {
			wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 3}
		}, ErrInvalidModel},
		{"gemm rank", "gemm_defaults", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"gemm flag", "gemm_defaults", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("transA"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(2)}}
		}, ErrInvalidModel},
		{"gemm bias", "gemm_00_vector", func(wire *onnxpb.ModelProto) {
			wire.Graph.Input[2].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 2}
		}, ErrInvalidModel},
		{"softmax scalar", "softmax_default", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"softmax axis", "softmax_default", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(2)}}
		}, ErrInvalidModel},
		{"softmax wrong attribute type", "softmax_default", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(1)}}
		}, ErrInvalidModel},
		{"extra attribute", "relu_finite", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(0)}}
		}, ErrUnsupported},
		{"empty required slot", "gemm_defaults", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Input[0] = "" }, ErrInvalidModel},
		{"extra input", "relu_finite", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Input = []string{"x", "x"} }, ErrUnsupported},
		{"multiple constant forms", "constant_float", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = append(wire.Graph.Node[0].Attribute, &onnxpb.AttributeProto{Name: proto.String("value_int"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)})
		}, ErrInvalidModel},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "m1/"+tt.fixture)
			tt.mutate(wire)
			model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{})
			var diagnostic *Error
			if model != nil || !errors.Is(err, tt.want) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 0 || diagnostic.Operator == "" {
				t.Fatalf("static rejection: %v, %v", model, err)
			}
		})
	}
}

func TestM1DynamicNodeErrorsAndRecovery(t *testing.T) {
	tests := []struct {
		fixture   string
		input     string
		badShape  []int64
		goodShape []int64
	}{
		{"add_broadcast", "a", []int64{3, 1}, []int64{2, 1}},
		{"matmul_dot", "b", []int64{3}, []int64{2}},
		{"gemm_00_column", "c", []int64{3, 1}, []int64{2, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			wire := modelWire(t, "m1/"+tt.fixture)
			// The other operand's leading axis fixes the broadcast/result size.
			if tt.fixture == "add_broadcast" {
				wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 2}
			}

			for _, input := range wire.Graph.Input {
				if input.GetName() == tt.input {
					input.Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "dynamic"}
				}
			}
			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			inputs := numericalInputs(t, wire)
			badCount := int64(1)
			for _, size := range tt.badShape {
				badCount *= size
			}
			inputs[tt.input] = floatTensor(t, tt.badShape, make([]float32, int(badCount)))
			outputs, err := session.Run(context.Background(), inputs, RunOptions{})
			var diagnostic *Error
			if outputs != nil || !errors.Is(err, ErrInvalidModel) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 0 || diagnostic.NodeName == "" || diagnostic.Operator == "" {
				t.Fatalf("dynamic node error: %v, %v", outputs, err)
			}

			goodCount := int64(1)
			for _, size := range tt.goodShape {
				goodCount *= size
			}
			inputs[tt.input] = floatTensor(t, tt.goodShape, make([]float32, int(goodCount)))
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("recovery: %v", err)
			}
		})
	}
}

func TestM1ConstantAggregateBudgets(t *testing.T) {
	for _, fixture := range []string{"constant_float", "constant_floats", "constant_int", "constant_ints", "constant_tensor", "constant_int_tensor"} {
		t.Run(fixture, func(t *testing.T) {
			data := modelFixture(t, "m1/"+fixture)
			baseline := loadedModel(t, data, Options{})
			bytes := baseline.plan.residentBytes
			if bytes == 0 {
				t.Fatal("materialized Constant is absent from initializer accounting")
			}

			exact := Options{Limits: Limits{MaxInitializerBytes: bytes, MaxLoadBytes: baseline.plan.loadBytes}}
			loadedModel(t, data, exact)
			tests := []Limits{{MaxInitializerBytes: bytes - 1}, {MaxLoadBytes: baseline.plan.loadBytes - 1}}
			for _, limits := range tests {
				model, err := LoadBytes(context.Background(), data, Options{Limits: limits})
				if model != nil || !errors.Is(err, ErrLimit) {
					t.Fatalf("Constant budget: %v, %v", model, err)
				}
			}
			wire := modelWire(t, "m1/"+fixture)
			// An initializer and scalar/list/tensor Constant share one aggregate quota.
			wire.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("unused"), DataType: proto.Int32(7), Dims: []int64{1}, Int64Data: []int64{1}}}
			model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{Limits: Limits{MaxInitializerBytes: bytes + 7}})
			if model != nil || !errors.Is(err, ErrLimit) {
				t.Fatalf("combined initializer/Constant quota: %v, %v", model, err)
			}
		})
	}
}

func TestM1CancellationAndOwnedOutputRecovery(t *testing.T) {
	for _, fixture := range []string{"relu_nonfinite", "mlp", "constant_ints"} {
		t.Run(fixture, func(t *testing.T) {
			wire := modelWire(t, "m1/"+fixture)
			inputs := numericalInputs(t, wire)
			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			baseline := newCancelContext(0)
			previous, err := session.Run(baseline, inputs, RunOptions{})
			baseline.cancel()
			if err != nil {
				t.Fatal(err)
			}

			for boundary := 1; boundary <= baseline.checks; boundary++ {
				ctx := newCancelContext(boundary)
				outputs, runErr := session.Run(ctx, inputs, RunOptions{})
				ctx.cancel()
				if outputs != nil || !errors.Is(runErr, context.Canceled) {
					t.Fatalf("boundary %d: %v, %v", boundary, outputs, runErr)
				}
			}
			recovered, err := session.Run(context.Background(), inputs, RunOptions{})
			if err != nil || len(recovered) != len(previous) {
				t.Fatalf("recovery: %v, %v", recovered, err)
			}

			for index, output := range previous {
				if output.Tensor.value == recovered[index].Tensor.value {
					t.Fatal("output storage reused")
				}
			}
		})
	}
}
