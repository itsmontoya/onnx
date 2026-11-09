package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func readM3Cases(t *testing.T) (cases []conformanceCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/m3/provenance.json")
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

	versions := map[string]string{"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3", "onnxruntime": "1.19.2"}
	if suite.FormatVersion != 1 || suite.ONNXSourceRevision != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || !maps.Equal(suite.ReferenceVersions, versions) || suite.OriginalLicense == "" || len(suite.Cases) != 87 {
		t.Fatal("M3 provenance differs")
	}

	names := make(map[string]bool)
	ortCounts := make(map[string]int)
	for _, fixture := range suite.Cases {
		if fixture.Name == "" || names[fixture.Name] || fixture.Origin == "" || fixture.License == "" {
			t.Fatal("invalid/duplicate M3 provenance")
		}

		names[fixture.Name] = true
		for _, oracle := range []string{"reference", "ort"} {
			var observation struct {
				Status  string `json:"status"`
				Matches bool   `json:"matches_independent"`
			}

			if err = json.Unmarshal(fixture.Observations[oracle], &observation); err != nil {
				t.Fatal(err)
			}

			classification := "difference"
			if observation.Matches {
				classification = "agreement"
			} else if observation.Status == "rejected" {
				classification = "extension"
			}

			if oracle == "reference" && (fixture.Status != classification || (fixture.Reason == "") != observation.Matches) {
				t.Fatalf("%s falsely classifies actual reference observation", fixture.Name)
			}

			if oracle == "ort" {
				ortCounts[classification]++
			}
		}
	}
	if !maps.Equal(ortCounts, map[string]int{"agreement": 66, "difference": 6, "extension": 15}) {
		t.Fatal(ortCounts)
	}

	return suite.Cases
}

func TestM3ReferenceGraphs(t *testing.T) {
	counts := make(map[string]int)
	for _, fixture := range readM3Cases(t) {
		counts[fixture.Status]++
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
	if !maps.Equal(counts, map[string]int{"agreement": 49, "difference": 26, "extension": 12}) {
		t.Fatal(counts)
	}
}

func m3Attribute(name string, values ...int64) (out *onnxpb.AttributeProto) {
	var attribute onnxpb.AttributeProto
	attribute.Name, attribute.Type = proto.String(name), onnxpb.AttributeProto_INTS.Enum()
	attribute.Ints = values
	if len(values) == 1 {
		attribute.Type, attribute.I, attribute.Ints = onnxpb.AttributeProto_INT.Enum(), proto.Int64(values[0]), nil
	}

	return &attribute
}

func TestM3StaticRejections(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		mutate  func(*onnxpb.ModelProto)
		want    error
	}{
		{"conv group zero", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("group", 0))
		}, ErrInvalidModel},
		{"conv group mismatch", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("group", 2))
		}, ErrInvalidModel},
		{"conv kernel mismatch", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("kernel_shape", 3, 3))
		}, ErrInvalidModel},
		{"conv bad stride", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("strides", 0, 1))
		}, ErrInvalidModel},
		{"conv bad dilation", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("dilations", -1, 1))
		}, ErrInvalidModel},
		{"conv negative pad", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("pads", -1, 0, 0, 0))
		}, ErrInvalidModel},
		{"conv short pad", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("pads", 0, 0))
		}, ErrInvalidModel},
		{"conv scalar bias", "conv_default_bias", func(w *onnxpb.ModelProto) { w.Graph.Input[2].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"conv bias size", "conv_default_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Input[2].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 2}
		}, ErrInvalidModel},
		{"conv weights rank", "conv_default_bias", func(w *onnxpb.ModelProto) { w.Graph.Input[1].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"conv int64", "conv_default_bias", func(w *onnxpb.ModelProto) { w.Graph.Input[1].Type.GetTensorType().ElemType = proto.Int32(7) }, ErrUnsupported},
		{"conflicting padding", "conv_same_upper_bias", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("pads", 0, 0, 0, 0))
		}, ErrInvalidModel},
		{"unknown auto pad", "conv_same_upper_bias", func(w *onnxpb.ModelProto) { w.Graph.Node[0].Attribute[0].S = []byte("SAME") }, ErrInvalidModel},
		{"max indices", "max_default", func(w *onnxpb.ModelProto) { w.Graph.Node[0].Output = []string{"y", "indices"} }, ErrUnsupported},
		{"max storage order", "max_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("storage_order", 1))
		}, ErrUnsupported},
		{"max invalid ceil", "max_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("ceil_mode", 2))
		}, ErrInvalidModel},
		{"max missing kernel", "max_default", func(w *onnxpb.ModelProto) { w.Graph.Node[0].Attribute = nil }, ErrInvalidModel},
		{"average dilation", "average_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = append(w.Graph.Node[0].Attribute, m3Attribute("dilations", 1, 1))
		}, ErrUnsupported},
		{"average include flag", "average_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute[0].I = proto.Int64(-1)
		}, ErrInvalidModel},
		{"global rank", "global_default", func(w *onnxpb.ModelProto) { w.Graph.Input[0].Type.GetTensorType().Shape.Dim = nil }, ErrUnsupported},
		{"bn training outputs", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Output = []string{"y", "mean_out", "var_out", "saved_mean", "saved_var"}
		}, ErrUnsupported},
		{"bn empty extra output", "bn_rank2_default", func(w *onnxpb.ModelProto) { w.Graph.Node[0].Output = []string{"y", ""} }, ErrUnsupported},
		{"bn later training attr", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("training_mode", 0)}
		}, ErrUnsupported},
		{"bn rank3", "bn_rank4_default", func(w *onnxpb.ModelProto) {
			w.Graph.Input[0].Type.GetTensorType().Shape.Dim = w.Graph.Input[0].Type.GetTensorType().Shape.Dim[:3]
		}, ErrUnsupported},
		{"bn scalar scale", "bn_rank2_default", func(w *onnxpb.ModelProto) { w.Graph.Input[1].Type.GetTensorType().Shape.Dim = nil }, ErrInvalidModel},
		{"bn channels", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 3}
		}, ErrInvalidModel},
		{"bn negative epsilon", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("epsilon"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(-1)}}
		}, ErrInvalidModel},
		{"bn nan epsilon", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("epsilon"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(float32(math.NaN()))}}
		}, ErrInvalidModel},
		{"bn infinite epsilon", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("epsilon"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(float32(math.Inf(1)))}}
		}, ErrInvalidModel},
		{"bn nan momentum", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("momentum"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(float32(math.NaN()))}}
		}, ErrInvalidModel},
		{"bn invalid momentum", "bn_rank2_default", func(w *onnxpb.ModelProto) {
			w.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("momentum"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(2)}}
		}, ErrInvalidModel},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			payload, err := os.ReadFile("testdata/m3/models/" + tt.fixture + ".onnx")
			if err != nil {
				t.Fatal(err)
			}

			var wire onnxpb.ModelProto
			if err = proto.Unmarshal(payload, &wire); err != nil {
				t.Fatal(err)
			}

			tt.mutate(&wire)
			if payload, err = proto.Marshal(&wire); err != nil {
				t.Fatal(err)
			}

			model, loadErr := LoadBytes(context.Background(), payload, Options{})
			var diagnostic *Error
			if model != nil || !errors.Is(loadErr, tt.want) || !errors.As(loadErr, &diagnostic) || diagnostic.NodeIndex != 0 || !strings.Contains(loadErr.Error(), "opset 13") {
				t.Fatalf("Load = %v, %v", model, loadErr)
			}
		})
	}
}

func TestM3ComposedCNN(t *testing.T) {
	var fixtures []conformanceCase
	for _, fixture := range readM3Cases(t) {
		if strings.HasPrefix(fixture.Name, "cnn_") {
			fixtures = append(fixtures, fixture)
		}
	}
	if len(fixtures) != 2 {
		t.Fatal("composed CNN requires two reference batches")
	}

	runConcurrentInference(t, fixtures)
	data, err := fixtures[0].Model.read()
	if err != nil {
		t.Fatal(err)
	}

	model := loadedModel(t, data, Options{})
	session := modelSession(t, model)
	var retained [][]NamedTensor
	for _, fixture := range append(fixtures, fixtures[0]) {
		inputs, inputErr := fixture.readInputs()
		if inputErr != nil {
			t.Fatal(inputErr)
		}

		outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
		if runErr != nil {
			t.Fatal(runErr)
		}

		if err = fixture.compareOutputs(outputs); err != nil {
			t.Fatal(err)
		}

		checkInferenceSelection(t, session, fixture, inputs)
		retained = append(retained, outputs)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	for index, fixture := range append(fixtures, fixtures[0]) {
		if err = fixture.compareOutputs(retained[index]); err != nil {
			t.Fatal(err)
		}
	}
}

func TestM3DynamicShapesAndRecovery(t *testing.T) {
	for _, name := range []string{"conv_default_bias", "conv_same_upper_bias", "max_default", "average_default", "global_default", "bn_rank4_default"} {
		t.Run(name, func(t *testing.T) {
			var fixture conformanceCase
			for _, candidate := range readM3Cases(t) {
				if candidate.Name == name {
					fixture = candidate
				}
			}
			wire := modelWire(t, "m3/models/"+name)
			nodeIndex := len(wire.Graph.Input)
			m3DynamicInputs(wire)
			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			inputs, err := fixture.readInputs()
			if err != nil {
				t.Fatal(err)
			}

			original := inputs["x"]
			outputs, runErr := m3DynamicRun(t, session, inputs)
			if runErr != nil {
				t.Fatal(runErr)
			}

			inputs["x"] = floatTensor(t, []int64{1, 2, 3}, make([]float32, 6))
			failed, invalidErr := m3DynamicRun(t, session, inputs)
			var diagnostic *Error
			if failed != nil || !errors.Is(invalidErr, ErrInvalidModel) || !errors.As(invalidErr, &diagnostic) || diagnostic.NodeIndex != nodeIndex {
				t.Fatalf("dynamic rank rejection: %v, %v", failed, invalidErr)
			}

			inputs["x"] = original
			if strings.HasPrefix(name, "conv_") {
				weight, bias := inputs["w"], inputs["b"]
				inputs["w"] = floatTensor(t, []int64{3, 1, 2, 3}, make([]float32, 18))
				if failed, invalidErr = m3DynamicRun(t, session, inputs); failed != nil || !errors.Is(invalidErr, ErrInvalidModel) {
					t.Fatalf("dynamic weight channels: %v, %v", failed, invalidErr)
				}

				inputs["w"], inputs["b"] = weight, floatTensor(t, nil, []float32{1})
				if failed, invalidErr = m3DynamicRun(t, session, inputs); failed != nil || !errors.Is(invalidErr, ErrInvalidModel) {
					t.Fatalf("dynamic scalar bias: %v, %v", failed, invalidErr)
				}

				inputs["b"] = bias
				inputs["w"] = floatTensor(t, weight.Shape(), make([]float32, weight.Len()))
				zeroWeights, zeroErr := m3DynamicRun(t, session, inputs)
				if zeroErr != nil {
					t.Fatal(zeroErr)
				}

				values, _ := zeroWeights[0].Tensor.Float32Values()
				biasValues, _ := bias.Float32Values()
				shape := zeroWeights[0].Tensor.Shape()
				for index, value := range values {
					if value != biasValues[(int64(index)/(shape[2]*shape[3]))%shape[1]] {
						t.Fatal("dynamic weights were cached")
					}
				}
				if name == "conv_same_upper_bias" {
					inputs["x"] = floatTensor(t, []int64{1, 2, 5, 6}, make([]float32, 60))
					inputs["w"] = floatTensor(t, []int64{3, 2, 1, 1}, make([]float32, 6))
					inputs["b"] = floatTensor(t, []int64{3}, []float32{1, 2, 3})
					changed, changeErr := m3DynamicRun(t, session, inputs)
					if changeErr != nil || !slices.Equal(changed[0].Tensor.Shape(), []int64{1, 3, 3, 3}) {
						t.Fatalf("dynamic SAME dimensions/weights/bias: %v, %v", changed, changeErr)
					}

					changedValues, _ := changed[0].Tensor.Float32Values()
					for index, value := range changedValues {
						if value != float32(index/9+1) {
							t.Fatal("dynamic bias or automatic padding was cached")
						}
					}
					inputs["x"], inputs["b"] = original, bias
				}

				inputs["w"] = weight
			}

			if strings.HasPrefix(name, "bn_") {
				scale := inputs["scale"]
				inputs["scale"] = floatTensor(t, []int64{1}, []float32{1})
				if failed, invalidErr = m3DynamicRun(t, session, inputs); failed != nil || !errors.Is(invalidErr, ErrInvalidModel) {
					t.Fatalf("dynamic normalization channels: %v, %v", failed, invalidErr)
				}

				inputs["scale"] = scale
			}

			if _, err = m3DynamicRun(t, session, inputs); err != nil {
				t.Fatalf("dynamic recovery: %v", err)
			}

			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatalf("dynamic failure changed earlier output: %v", err)
			}
		})
	}
}

func TestM3CancellationAndRecovery(t *testing.T) {
	selected := map[string]bool{"conv_grouped_bias": true, "max_dilated": true, "average_ceil_partial_include": true, "global_default": true, "bn_rank4_epsilon": true, "cnn_batch_3": true}
	for _, fixture := range readM3Cases(t) {
		if !selected[fixture.Name] {
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
			baseline := newCancelContext(0)
			defer baseline.cancel()
			retained, runErr := session.Run(baseline, inputs, RunOptions{})
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
				t.Fatalf("cancellation recovery: %v", err)
			}

			if err = fixture.compareOutputs(retained); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// Runtime Reshape nodes supply genuinely unknown CNN ranks/shapes without
// broadening the public graph-input metadata contract.
func m3DynamicInputs(wire *onnxpb.ModelProto) {
	var (
		inputs []*onnxpb.ValueInfoProto
		nodes  []*onnxpb.NodeProto
	)

	for _, original := range wire.Graph.Input {
		name := original.GetName()
		for _, suffix := range []string{"_flat", "_shape"} {
			info := proto.Clone(original).(*onnxpb.ValueInfoProto)
			info.Name = proto.String(name + suffix)
			info.Type.GetTensorType().Shape.Dim = []*onnxpb.TensorShapeProto_Dimension{{}}
			if suffix == "_shape" {
				info.Type.GetTensorType().ElemType = proto.Int32(7)
			}

			inputs = append(inputs, info)
		}
		nodes = append(nodes, &onnxpb.NodeProto{OpType: proto.String("Reshape"), Input: []string{name + "_flat", name + "_shape"}, Output: []string{name}})
	}
	wire.Graph.Input = inputs
	wire.Graph.Node = append(nodes, wire.Graph.Node...)
	for _, output := range wire.Graph.Output {
		output.Type.GetTensorType().Shape = nil
	}
}

func m3DynamicRun(t *testing.T, session *Session, inputs map[string]*Tensor) (outputs []NamedTensor, err error) {
	t.Helper()
	bindings := make(map[string]*Tensor)
	for name, value := range inputs {
		var values []float32
		if values, err = value.Float32Values(); err != nil {
			return nil, err
		}

		bindings[name+"_flat"] = floatTensor(t, []int64{value.Len()}, values)
		bindings[name+"_shape"] = intTensor(t, []int64{int64(len(value.Shape()))}, value.Shape())
	}
	return session.Run(context.Background(), bindings, RunOptions{})
}

func TestM3OptionalEmptySlots(t *testing.T) {
	for _, name := range []string{"conv_default_no_bias", "max_default"} {
		t.Run(name, func(t *testing.T) {
			var fixture conformanceCase
			for _, candidate := range readM3Cases(t) {
				if candidate.Name == name {
					fixture = candidate
				}
			}
			wire := modelWire(t, "m3/models/"+name)
			if name == "max_default" {
				wire.Graph.Node[0].Output = append(wire.Graph.Node[0].Output, "")
			} else {
				wire.Graph.Node[0].Input = append(wire.Graph.Node[0].Input, "")
			}

			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			step := session.plan.steps[0]
			if name == "max_default" && (len(step.Outputs) != 2 || step.Outputs[1] != -1) || name != "max_default" && (len(step.Inputs) != 3 || step.Inputs[2] != -1) {
				t.Fatalf("optional slot lost: %+v", step)
			}

			inputs, err := fixture.readInputs()
			if err != nil {
				t.Fatal(err)
			}

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
