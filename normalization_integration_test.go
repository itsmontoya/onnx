package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func readNormalizationCases(t *testing.T) (cases []conformanceCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/normalization/provenance.json")
	if err != nil {
		t.Fatal(err)
	}

	var suite struct {
		Source   string              `json:"onnx_source_revision"`
		Versions map[string]string   `json:"reference_versions"`
		Schema   conformanceArtifact `json:"schema_audit"`
		Cases    []conformanceCase   `json:"cases"`
	}
	if err = json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}

	versions := map[string]string{"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3", "onnxruntime": "1.19.2"}
	if suite.Source != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || !maps.Equal(suite.Versions, versions) || len(suite.Cases) != 28 {
		t.Fatal("normalization provenance changed")
	}

	if _, err = suite.Schema.read(); err != nil {
		t.Fatal(err)
	}

	return suite.Cases
}

func TestNormalizationFixtures(t *testing.T) {
	for _, fixture := range readNormalizationCases(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			outputs, err := fixture.run()
			if err != nil {
				t.Fatal(err)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestNormalizationSessions(t *testing.T) {
	fixtures := readNormalizationCases(t)
	var branches []conformanceCase
	for _, fixture := range fixtures {
		if strings.HasPrefix(fixture.Name, "residual_normalization_batch_") {
			branches = append(branches, fixture)
		}
	}
	if len(branches) != 2 {
		t.Fatal("missing normalization DAG batches")
	}

	runConcurrentInference(t, branches)
	data, err := branches[0].Model.read()
	if err != nil {
		t.Fatal(err)
	}

	model := loadedModel(t, data, Options{})
	session := modelSession(t, model)
	var retained []NamedTensor
	for _, fixture := range branches {
		inputs, inputErr := fixture.readInputs()
		if inputErr != nil {
			t.Fatal(inputErr)
		}

		before, valuesErr := inputs["x"].Float32Values()
		if valuesErr != nil {
			t.Fatal(valuesErr)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err = session.Run(ctx, inputs, RunOptions{}); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Run = %v", err)
		}

		original := session.plan.limits
		session.plan.limits.MaxWorkspaceBytes = 1
		if _, err = session.Run(context.Background(), inputs, RunOptions{}); !errors.Is(err, ErrLimit) {
			t.Fatalf("workspace limit = %v", err)
		}

		session.plan.limits = original
		outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
		if runErr != nil {
			t.Fatal(runErr)
		}

		if err = fixture.compareOutputs(outputs); err != nil {
			t.Fatal(err)
		}

		if retained == nil {
			retained = outputs
		}

		subset, subsetErr := session.Run(context.Background(), inputs, RunOptions{Outputs: []string{"inverse", "score"}})
		if subsetErr != nil || len(subset) != 2 || subset[0].Name != "inverse" || subset[1].Name != "score" {
			t.Fatalf("reordered subset = %v, %v", subset, subsetErr)
		}

		after, afterErr := inputs["x"].Float32Values()
		if afterErr != nil || !slices.Equal(before, after) {
			t.Fatalf("normalization mutated input: %v", afterErr)
		}
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	if err = branches[0].compareOutputs(retained); err != nil {
		t.Fatal(err)
	}
}

func TestNormalizationValidation(t *testing.T) {
	fixtures := readNormalizationCases(t)
	var data []byte
	var err error
	for _, fixture := range fixtures {
		if fixture.Name == "residual_normalization_batch_1" {
			if data, err = fixture.Model.read(); err != nil {
				t.Fatal(err)
			}
		}
	}
	tests := []struct {
		name   string
		change func(*onnxpb.ModelProto)
		want   error
	}{
		{name: "profile13", change: func(w *onnxpb.ModelProto) { w.OpsetImport[0].Version = proto.Int64(13) }, want: ErrUnsupported},
		{name: "stash_bfloat16", change: func(w *onnxpb.ModelProto) {
			w.Graph.Node[1].Attribute = append(w.Graph.Node[1].Attribute, m3Attribute("stash_type", 16))
		}, want: ErrUnsupported},
		{name: "invalid_axis", change: func(w *onnxpb.ModelProto) {
			w.Graph.Node[1].Attribute = append(w.Graph.Node[1].Attribute, m3Attribute("axis", 3))
		}, want: ErrInvalidModel},
		{name: "negative_epsilon", change: func(w *onnxpb.ModelProto) {
			w.Graph.Node[1].Attribute = append(w.Graph.Node[1].Attribute, normalizationFloatAttribute("epsilon", -1))
		}, want: ErrUnsupported},
		{name: "unknown_attribute", change: func(w *onnxpb.ModelProto) {
			w.Graph.Node[1].Attribute = append(w.Graph.Node[1].Attribute, m3Attribute("training_mode", 0))
		}, want: ErrUnsupported},
		{name: "scale_rank", change: func(w *onnxpb.ModelProto) {
			w.Graph.Input[2].Type.GetTensorType().Shape.Dim = append(w.Graph.Input[2].Type.GetTensorType().Shape.Dim, &onnxpb.TensorShapeProto_Dimension{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 4}})
		}, want: ErrInvalidModel},
		{name: "empty_normalized_dimension", change: func(w *onnxpb.ModelProto) {
			w.Graph.Input[0].Type.GetTensorType().Shape.Dim[2] = &onnxpb.TensorShapeProto_Dimension{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 0}}
			w.Graph.Input[1].Type.GetTensorType().Shape.Dim[2] = w.Graph.Input[0].Type.GetTensorType().Shape.Dim[2]
		}, want: ErrUnsupported},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var wire onnxpb.ModelProto
			if err = proto.Unmarshal(data, &wire); err != nil {
				t.Fatal(err)
			}

			tt.change(&wire)
			changed, marshalErr := proto.Marshal(&wire)
			if marshalErr != nil {
				t.Fatal(marshalErr)
			}

			model, loadErr := LoadBytes(context.Background(), changed, Options{})
			if model != nil || !errors.Is(loadErr, tt.want) {
				t.Fatalf("Load = %v, %v; want %v", model, loadErr, tt.want)
			}
		})
	}
}

func normalizationFloatAttribute(name string, value float32) (out *onnxpb.AttributeProto) {
	var attribute onnxpb.AttributeProto
	attribute.Name, attribute.Type, attribute.F = proto.String(name), onnxpb.AttributeProto_FLOAT.Enum(), proto.Float32(value)
	return &attribute
}

func TestNormalizationDynamicFailureRecovery(t *testing.T) {
	var fixture conformanceCase
	for _, candidate := range readNormalizationCases(t) {
		if candidate.Name == "y_only_zero_epsilon" {
			fixture = candidate
		}
	}
	data, err := fixture.Model.read()
	if err != nil {
		t.Fatal(err)
	}

	var wire onnxpb.ModelProto
	if err = proto.Unmarshal(data, &wire); err != nil {
		t.Fatal(err)
	}

	wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "batch"}}
	wire.Graph.Output[0].Type.GetTensorType().Shape.Dim[0] = wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0]
	wire.Graph.Node[0].Attribute = append(wire.Graph.Node[0].Attribute, m3Attribute("axis", 0))
	wire.Graph.Node[0].Input = append(wire.Graph.Node[0].Input, "")
	if data, err = proto.Marshal(&wire); err != nil {
		t.Fatal(err)
	}

	model := loadedModel(t, data, Options{})
	session := modelSession(t, model)
	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{0, 3}, nil), "scale": floatTensor(t, []int64{3}, []float32{1, 1, 1})}
	outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
	var diagnostic *Error
	if outputs != nil || !errors.Is(runErr, ErrUnsupported) || !errors.As(runErr, &diagnostic) || diagnostic.Operator != "LayerNormalization" || diagnostic.NodeIndex != 0 {
		t.Fatalf("dynamic empty reduction = %v, %v", outputs, runErr)
	}

	inputs["x"] = floatTensor(t, []int64{1, 3}, []float32{1, 2, 3})
	if outputs, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil || len(outputs) != 1 || !slices.Equal(outputs[0].Tensor.Shape(), []int64{1, 3}) {
		t.Fatalf("dynamic recovery = %v, %v", outputs, err)
	}
}
