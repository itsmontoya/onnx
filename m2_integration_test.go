package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func readM2Cases(t *testing.T) (cases []conformanceCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/m2/provenance.json")
	if err != nil {
		t.Fatal(err)
	}

	var suite struct {
		FormatVersion      int               `json:"format_version"`
		ONNXSourceRevision string            `json:"onnx_source_revision"`
		ReferenceVersions  map[string]string `json:"reference_versions"`
		OriginalLicense    string            `json:"original_license"`
		Cases              []conformanceCase `json:"cases"`
	}

	if err = json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}

	versions := map[string]string{"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3"}
	if suite.FormatVersion != 1 || suite.ONNXSourceRevision != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || len(suite.Cases) != 66 || !maps.Equal(suite.ReferenceVersions, versions) || suite.OriginalLicense == "" {
		t.Fatal("M2 provenance differs")
	}

	return suite.Cases
}

func TestM2ReferenceGraphs(t *testing.T) {
	counts := make(map[string]int)
	for _, fixture := range readM2Cases(t) {
		counts[fixture.Status]++
		t.Run(fixture.Name, func(t *testing.T) {
			outputs, err := fixture.run()
			if err != nil {
				t.Fatal(err)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatal(err)
			}

			if outputs[0].Tensor.DataType() == Float32 {
				expected, readErr := fixture.Outputs[0].read()
				if readErr != nil {
					t.Fatal(readErr)
				}

				actualValues, _ := outputs[0].Tensor.Float32Values()
				expectedValues, _ := expected.Tensor.Float32Values()
				for index, value := range actualValues {
					if math.Float32bits(value) != math.Float32bits(expectedValues[index]) {
						t.Fatalf("storage transform changed float bits at %d", index)
					}
				}
			}
		})
	}
	if !maps.Equal(counts, map[string]int{"agreement": 60, "extension": 2, "difference": 4}) {
		t.Fatal(counts)
	}
}

func m2Vector(wire *onnxpb.ModelProto, values []int64) {
	name := wire.Graph.Input[1].GetName()
	wire.Graph.Initializer = append(wire.Graph.Initializer, &onnxpb.TensorProto{Name: proto.String(name), DataType: proto.Int32(7), Dims: []int64{int64(len(values))}, Int64Data: values})
	wire.Graph.Input = wire.Graph.Input[:1]
}

func TestM2StaticRejections(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		mutate  func(*onnxpb.ModelProto)
		want    error
	}{
		{"reshape below -1", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{-2, 12}) }, ErrInvalidModel},
		{"reshape repeated -1", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{-1, -1}) }, ErrInvalidModel},
		{"reshape product", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{5, 5}) }, ErrInvalidModel},
		{"reshape fractional", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{-1, 5}) }, ErrInvalidModel},
		{"reshape zero position", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{1, 1, 1, 0}) }, ErrInvalidModel},
		{"reshape ambiguous zero", "reshape_empty_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{0, -1}) }, ErrInvalidModel},
		{"reshape later attribute", "reshape_empty_float32", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("allowzero"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)}}
		}, ErrUnsupported},
		{"reshape float shape", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Input[1].Type.GetTensorType().ElemType = proto.Int32(1) }, ErrUnsupported},
		{"reshape scalar shape", "reshape_zero_infer_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Input[1].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"shape newer start", "shape_float32", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("start"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(0)}}
		}, ErrUnsupported},
		{"flatten axis", "flatten_default_float32", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(4)}}
		}, ErrInvalidModel},
		{"transpose duplicate", "transpose_explicit_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Attribute[0].Ints = []int64{0, 0, 2} }, ErrInvalidModel},
		{"transpose missing", "transpose_explicit_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Attribute[0].Ints = []int64{0, 1} }, ErrInvalidModel},
		{"transpose negative", "transpose_explicit_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Attribute[0].Ints = []int64{0, 1, -1} }, ErrInvalidModel},
		{"squeeze normalized duplicate", "squeeze_negative_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{-1, 5}) }, ErrInvalidModel},
		{"squeeze nonunit", "squeeze_negative_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{1}) }, ErrInvalidModel},
		{"squeeze old attribute", "squeeze_omitted_float32", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axes"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0}}}
		}, ErrUnsupported},
		{"unsqueeze normalized duplicate", "unsqueeze_output_rank_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{0, -5}) }, ErrInvalidModel},
		{"unsqueeze output axis", "unsqueeze_output_rank_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{5, 0}) }, ErrInvalidModel},
		{"unsqueeze missing axes", "unsqueeze_output_rank_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Input = wire.Graph.Node[0].Input[:1] }, ErrInvalidModel},
		{"gather index", "gather_scalar_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{3}) }, ErrInvalidModel},
		{"gather index minimum", "gather_scalar_float32", func(wire *onnxpb.ModelProto) { m2Vector(wire, []int64{-1 << 63}) }, ErrInvalidModel},
		{"concat type", "concat_negative_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Input[1].Type.GetTensorType().ElemType = proto.Int32(7) }, ErrInvalidModel},
		{"concat rank", "concat_negative_float32", func(wire *onnxpb.ModelProto) {
			wire.Graph.Input[1].Type.GetTensorType().Shape.Dim = wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[:2]
		}, ErrInvalidModel},
		{"concat dimensions", "concat_negative_float32", func(wire *onnxpb.ModelProto) {
			wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 3}
		}, ErrInvalidModel},
		{"concat empty list", "concat_negative_float32", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Input = nil }, ErrInvalidModel},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "m2/models/"+tt.fixture)
			tt.mutate(wire)
			model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{})
			var diagnostic *Error
			if model != nil || !errors.Is(err, tt.want) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 0 {
				t.Fatalf("static rejection: %v, %v", model, err)
			}
		})
	}
}

func TestM2DynamicFailuresRecover(t *testing.T) {
	tests := []struct {
		fixture string
		name    string
		invalid []int64
	}{
		{"reshape_zero_infer_float32", "shape", []int64{-1, -1}},
		{"reshape_empty_float32", "shape", []int64{0, -1}},
		{"squeeze_negative_float32", "axes", []int64{-1, 5}},
		{"unsqueeze_output_rank_float32", "axes", []int64{0, -5}},
		{"gather_general_int64", "indices", []int64{0, -1, 3, 0}},
	}
	for _, tt := range tests {
		t.Run(tt.fixture, func(t *testing.T) {
			var fixture conformanceCase
			for _, candidate := range readM2Cases(t) {
				if candidate.Name == tt.fixture {
					fixture = candidate
				}
			}
			inputs, err := fixture.readInputs()
			if err != nil {
				t.Fatal(err)
			}

			wire := modelWire(t, "m2/models/"+tt.fixture)
			model := loadedModel(t, modelBytes(t, wire), Options{})
			session := modelSession(t, model)
			original := inputs[tt.name]
			inputs[tt.name] = intTensor(t, original.Shape(), tt.invalid)
			outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
			var diagnostic *Error
			if outputs != nil || !errors.Is(runErr, ErrInvalidModel) || !errors.As(runErr, &diagnostic) || diagnostic.NodeIndex != 0 {
				t.Fatalf("dynamic rejection: %v, %v", outputs, runErr)
			}

			inputs[tt.name] = original
			if outputs, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatal(err)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestM2EmptyGatherValidatesIndices(t *testing.T) {
	wire := modelWire(t, "m2/models/gather_scalar_float32")
	wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 0}
	wire.Graph.Output[0].Type.GetTensorType().Shape = nil
	model := loadedModel(t, modelBytes(t, wire), Options{})
	session := modelSession(t, model)
	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{0, 3, 4}, nil), "indices": intTensor(t, nil, []int64{3})}
	if outputs, err := session.Run(context.Background(), inputs, RunOptions{}); outputs != nil || !errors.Is(err, ErrInvalidModel) {
		t.Fatalf("empty prefix bypassed invalid index: %v, %v", outputs, err)
	}

	inputs["indices"] = intTensor(t, nil, []int64{-1})
	outputs, err := session.Run(context.Background(), inputs, RunOptions{})
	if err != nil || !slices.Equal(outputs[0].Tensor.Shape(), []int64{0, 4}) {
		t.Fatalf("empty recovery: %v, %v", outputs, err)
	}
}

func TestM2ConstantVectorInputs(t *testing.T) {
	for _, fixture := range readM2Cases(t) {
		if len(fixture.Inputs) != 2 || fixture.Schemas[0].Operator == "Concat" {
			continue
		}

		t.Run(fixture.Name, func(t *testing.T) {
			wire := modelWire(t, "m2/models/"+fixture.Name)
			data, err := fixture.Inputs[1].Tensor.read()
			if err != nil {
				t.Fatal(err)
			}

			var constant onnxpb.TensorProto
			if err = proto.Unmarshal(data, &constant); err != nil {
				t.Fatal(err)
			}

			wire.Graph.Initializer = append(wire.Graph.Initializer, &constant)
			wire.Graph.Input = wire.Graph.Input[:1]
			model := loadedModel(t, modelBytes(t, wire), Options{})
			session := modelSession(t, model)
			inputs, inputErr := fixture.readInputs()
			if inputErr != nil {
				t.Fatal(inputErr)
			}

			delete(inputs, constant.GetName())
			outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatal(runErr)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestM2OmittedSqueezeTracksConcreteBatch(t *testing.T) {
	wire := modelWire(t, "m2/models/squeeze_omitted_float32")
	wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "batch"}
	wire.Graph.Output[0].Type.GetTensorType().Shape = nil
	model := loadedModel(t, modelBytes(t, wire), Options{})
	session := modelSession(t, model)
	var retained *Tensor
	for _, batch := range []int64{1, 3, 1} {
		input := floatTensor(t, []int64{batch, 2, 1, 3, 4, 1}, make([]float32, int(batch)*24))
		outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		expected := []int64{2, 3, 4}
		if batch != 1 {
			expected = []int64{batch, 2, 3, 4}
		}

		if !slices.Equal(outputs[0].Tensor.Shape(), expected) {
			t.Fatalf("batch %d: shape %v", batch, outputs[0].Tensor.Shape())
		}

		if retained == nil {
			retained = outputs[0].Tensor
		}

		if !slices.Equal(retained.Shape(), []int64{2, 3, 4}) {
			t.Fatal("earlier shape changed")
		}
	}
}

func TestM2ConcatChecksEmptyAxisSum(t *testing.T) {
	wire := modelWire(t, "m2/models/concat_negative_float32")
	for _, input := range wire.Graph.Input {
		input.Type.GetTensorType().Shape.Dim = []*onnxpb.TensorShapeProto_Dimension{
			{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 0}},
			{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1<<63 - 1}},
		}
	}
	wire.Graph.Output[0].Type.GetTensorType().Shape = nil
	model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{Limits: Limits{MaxDimension: 1<<63 - 1}})
	var diagnostic *Error
	if model != nil || !errors.Is(err, ErrInvalidModel) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 0 {
		t.Fatalf("empty Concat sum overflow: %v, %v", model, err)
	}
}

func TestM2DefaultShapeCanBeOverridden(t *testing.T) {
	wire := modelWire(t, "m2/models/reshape_zero_infer_float32")
	wire.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("shape"), DataType: proto.Int32(7), Dims: []int64{2}, Int64Data: []int64{-1, -1}}}
	model := loadedModel(t, modelBytes(t, wire), Options{})
	session := modelSession(t, model)
	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{2, 3, 4}, make([]float32, 24))}
	if outputs, err := session.Run(context.Background(), inputs, RunOptions{}); outputs != nil || !errors.Is(err, ErrInvalidModel) {
		t.Fatalf("invalid runtime default: %v, %v", outputs, err)
	}

	inputs["shape"] = intTensor(t, []int64{2}, []int64{0, -1})
	outputs, err := session.Run(context.Background(), inputs, RunOptions{})
	if err != nil || !slices.Equal(outputs[0].Tensor.Shape(), []int64{2, 12}) {
		t.Fatalf("default override: %v, %v", outputs, err)
	}
}

func TestM2CancellationAndRecovery(t *testing.T) {
	selected := map[string]bool{
		"shape_float32": true, "flatten_default_float32": true,
		"transpose_explicit_float32": true, "reshape_zero_infer_int64": true,
		"squeeze_negative_int64": true, "unsqueeze_scalar_float32": true,
		"concat_negative_float32": true, "gather_general_int64": true,
	}
	var exercised int
	for _, fixture := range readM2Cases(t) {
		if !selected[fixture.Name] {
			continue
		}

		exercised++
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
			baseline := newCancelContext(0)
			retained, runErr := session.Run(baseline, inputs, RunOptions{})
			baseline.cancel()
			if runErr != nil {
				t.Fatal(runErr)
			}

			for boundary := 1; boundary <= baseline.checks; boundary++ {
				ctx := newCancelContext(boundary)
				outputs, cancelErr := session.Run(ctx, inputs, RunOptions{})
				ctx.cancel()
				if outputs != nil || !errors.Is(cancelErr, context.Canceled) {
					t.Fatalf("boundary %d: %v, %v", boundary, outputs, cancelErr)
				}
			}
			if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
				t.Fatalf("cancellation left session state: %v", err)
			}

			if err = fixture.compareOutputs(retained); err != nil {
				t.Fatalf("cancellation changed retained outputs: %v", err)
			}
		})
	}
	if exercised != len(selected) {
		t.Fatalf("exercised %d shape handlers, want %d", exercised, len(selected))
	}
}
