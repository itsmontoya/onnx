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

func readProfileCases(t *testing.T) (cases []conformanceCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/profiles/provenance.json")
	if err != nil {
		t.Fatal(err)
	}

	var suite struct {
		FormatVersion      int                 `json:"format_version"`
		ONNXSourceRevision string              `json:"onnx_source_revision"`
		ReferenceVersions  map[string]string   `json:"reference_versions"`
		OriginalLicense    string              `json:"original_license"`
		SchemaAudit        conformanceArtifact `json:"schema_audit"`
		Cases              []conformanceCase   `json:"cases"`
	}
	if err = json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}

	versions := map[string]string{"python": "3.9.6", "onnx": "1.16.2", "numpy": "1.26.4", "protobuf": "4.25.3", "onnxruntime": "1.19.2", "torch": "2.2.2"}
	if suite.FormatVersion != 1 || suite.ONNXSourceRevision != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || !maps.Equal(suite.ReferenceVersions, versions) || suite.OriginalLicense == "" || len(suite.Cases) != 67 {
		t.Fatal("profile provenance differs")
	}

	if _, err = suite.SchemaAudit.read(); err != nil {
		t.Fatal(err)
	}

	counts := make(map[int64]map[string]bool)
	names := make(map[string]bool)
	agreements, differences := 0, 0
	for _, fixture := range suite.Cases {
		if fixture.Name == "" || names[fixture.Name] {
			t.Fatal("empty or duplicate profile case name")
		}

		names[fixture.Name] = true
		var observation struct {
			Status string `json:"status"`
		}
		if err = json.Unmarshal(fixture.Observations["reference"], &observation); err != nil || observation.Status != "returned" {
			t.Fatal("profile must retain the executed evaluator output")
		}

		if fixture.Status == "agreement" && fixture.Reason == "" {
			agreements++
		} else if fixture.Status == "difference" && fixture.Name == "bn_stored_statistics_13" && fixture.Reason != "" {
			differences++
		} else {
			t.Fatal("profile reference classification differs")
		}

		if err = json.Unmarshal(fixture.Observations["onnxruntime"], &observation); err != nil || observation.Status != "returned" {
			t.Fatal("profile must retain the executed ORT output")
		}

		if fixture.IRVersion != 8 || len(fixture.Opsets) != 1 || fixture.Opsets[""] != 13 && fixture.Opsets[""] != 17 || fixture.Origin == "" || fixture.License == "" {
			t.Fatalf("profile fixture provenance differs: %+v", fixture)
		}

		if strings.HasPrefix(fixture.Name, "opset") {
			if counts[fixture.Opsets[""]] == nil {
				counts[fixture.Opsets[""]] = make(map[string]bool)
			}

			counts[fixture.Opsets[""]][fixture.Schemas[0].Operator] = true
		}
	}
	if len(counts[13]) != 25 || len(counts[17]) != 25 || agreements != 66 || differences != 1 {
		t.Fatal("each profile must exercise all 25 operators")
	}

	return suite.Cases
}

func TestVersionProfileFixtures(t *testing.T) {
	for _, fixture := range readProfileCases(t) {
		t.Run(fixture.Name, func(t *testing.T) {
			outputs, err := fixture.run()
			if err != nil {
				t.Fatal(err)
			}

			// run closes both owners; returned tensors remain caller-owned.
			if err = fixture.compareOutputs(outputs); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestVersionProfileExporterSessions(t *testing.T) {
	var fixtures []conformanceCase
	for _, fixture := range readProfileCases(t) {
		if strings.HasPrefix(fixture.Name, "pytorch_mlp_batch_") {
			fixtures = append(fixtures, fixture)
		}
	}
	if len(fixtures) != 2 || fixtures[0].Model != fixtures[1].Model {
		t.Fatal("expected two batches from the same unmodified export")
	}

	runConcurrentInference(t, fixtures)
	data, err := fixtures[0].Model.read()
	if err != nil {
		t.Fatal(err)
	}

	model := loadedModel(t, data, Options{})
	if model.Metadata().IRVersion != 8 || model.Metadata().Opset != 17 {
		t.Fatalf("actual imported version lost: %+v", model.Metadata())
	}

	session := modelSession(t, model)
	var retained []NamedTensor
	for index, fixture := range fixtures {
		inputs, inputErr := fixture.readInputs()
		if inputErr != nil {
			t.Fatal(inputErr)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if outputs, runErr := session.Run(ctx, inputs, RunOptions{}); outputs != nil || !errors.Is(runErr, context.Canceled) {
			t.Fatalf("cancelled profile Run = %v, %v", outputs, runErr)
		}

		original := session.plan.limits
		session.plan.limits.MaxLiveBytes = 1
		if outputs, runErr := session.Run(context.Background(), inputs, RunOptions{}); outputs != nil || !errors.Is(runErr, ErrLimit) {
			t.Fatalf("profile live budget = %v, %v", outputs, runErr)
		}

		session.plan.limits = original
		outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
		if runErr != nil {
			t.Fatal(runErr)
		}

		if err = fixture.compareOutputs(outputs); err != nil {
			t.Fatal(err)
		}

		if index == 0 {
			retained = outputs
		}
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	if err = fixtures[0].compareOutputs(retained); err != nil {
		t.Fatal(err)
	}
}

func TestVersionProfileRejections(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		mutate  func(*onnxpb.ModelProto)
		want    error
	}{
		{"reshape old allowzero zero", "opset13_reshape", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("allowzero", 0)}
		}, ErrUnsupported},
		{"shape old start zero", "opset13_shape", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("start", 0)}
		}, ErrUnsupported},
		{"shape old end", "opset13_shape", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("end", 1)}
		}, ErrUnsupported},
		{"bn old training zero", "opset13_batchnormalization", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = append(wire.Graph.Node[0].Attribute, m3Attribute("training_mode", 0))
		}, ErrUnsupported},
		{"reshape mode invalid", "opset17_reshape", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("allowzero", 2)}
		}, ErrInvalidModel},
		{"reshape literal zero infer", "opset17_reshape", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("allowzero", 1)}
		}, ErrInvalidModel},
		{"bn training", "opset17_batchnormalization", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = append(wire.Graph.Node[0].Attribute, m3Attribute("training_mode", 1))
		}, ErrUnsupported},
		{"bn invalid training", "opset17_batchnormalization", func(wire *onnxpb.ModelProto) {
			wire.Graph.Node[0].Attribute = append(wire.Graph.Node[0].Attribute, m3Attribute("training_mode", -1))
		}, ErrUnsupported},
		{"bn extra empty output", "opset17_batchnormalization", func(wire *onnxpb.ModelProto) { wire.Graph.Node[0].Output = append(wire.Graph.Node[0].Output, "") }, ErrUnsupported},
		{"identity extended float16", "opset17_identity", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().ElemType = proto.Int32(10) }, ErrUnsupported},
		{"relu extended int64", "opset17_relu", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().ElemType = proto.Int32(7) }, ErrUnsupported},
		{"arithmetic extended int64", "opset17_add", func(wire *onnxpb.ModelProto) { wire.Graph.Input[0].Type.GetTensorType().ElemType = proto.Int32(7) }, ErrUnsupported},
		{"sequence identity", "opset17_identity", func(wire *onnxpb.ModelProto) {
			wire.Graph.Input[0].Type.Value = &onnxpb.TypeProto_SequenceType{SequenceType: &onnxpb.TypeProto_Sequence{}}
		}, ErrUnsupported},
		{"local function", "opset17_identity", func(wire *onnxpb.ModelProto) { wire.Functions = []*onnxpb.FunctionProto{{}} }, ErrUnsupported},
		{"training information", "opset17_identity", func(wire *onnxpb.ModelProto) { wire.TrainingInfo = []*onnxpb.TrainingInfoProto{{}} }, ErrUnsupported},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "profiles/models/"+tt.fixture)
			tt.mutate(wire)
			data := modelBytes(t, wire)
			model, err := LoadBytes(context.Background(), data, Options{})
			if model != nil || !errors.Is(err, tt.want) {
				t.Fatalf("profile rejection = %v, %v", model, err)
			}

			_, repeated := LoadBytes(context.Background(), data, Options{})
			if repeated == nil || err.Error() != repeated.Error() {
				t.Fatalf("nondeterministic rejection: %v / %v", err, repeated)
			}
		})
	}
	for _, imported := range []int64{1, 12, 14, 15, 16, 18, 21, 99} {
		wire := modelWire(t, "profiles/models/opset17_identity")
		wire.OpsetImport[0].Version = proto.Int64(imported)
		if model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{}); model != nil || !errors.Is(err, ErrUnsupported) {
			t.Fatalf("untested import %d = %v, %v", imported, model, err)
		}
	}
	for _, ir := range []int64{7, 9, 10, 11} {
		wire := modelWire(t, "profiles/models/opset17_identity")
		wire.IrVersion = proto.Int64(ir)
		if model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{}); model != nil || !errors.Is(err, ErrUnsupported) {
			t.Fatalf("untested IR %d = %v, %v", ir, model, err)
		}
	}
}

func TestVersionProfileDynamicReshape(t *testing.T) {
	for _, mode := range []int64{0, 1} {
		wire := modelWire(t, "profiles/models/opset17_reshape")
		wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("allowzero", mode)}
		wire.Graph.Initializer = nil
		wire.Graph.Input = append(wire.Graph.Input, proto.Clone(wire.Graph.Input[0]).(*onnxpb.ValueInfoProto))
		target := wire.Graph.Input[1]
		target.Name = proto.String("target")
		target.Type.GetTensorType().ElemType = proto.Int32(7)
		target.Type.GetTensorType().Shape.Dim = target.Type.GetTensorType().Shape.Dim[:1]
		wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "batch"}
		wire.Graph.Output[0].Type.GetTensorType().Shape = nil
		model := loadedModel(t, modelBytes(t, wire), Options{})
		session := modelSession(t, model)
		inputs := map[string]*Tensor{"x": floatTensor(t, []int64{2, 2}, []float32{1, 2, 3, 4}), "target": intTensor(t, []int64{2}, []int64{2, 2})}
		retained, err := session.Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		for _, invalid := range [][]int64{{0, -1}, {-1, -1}, {3, 2}} {
			if mode == 0 && slices.Equal(invalid, []int64{0, -1}) {
				continue
			}

			inputs["target"] = intTensor(t, []int64{2}, invalid)
			outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
			var diagnostic *Error
			if outputs != nil || !errors.Is(runErr, ErrInvalidModel) || !errors.As(runErr, &diagnostic) || diagnostic.Opset != 17 || diagnostic.NodeIndex != 0 {
				t.Fatalf("dynamic mode %d rejection = %v, %v", mode, outputs, runErr)
			}
		}
		inputs["x"] = floatTensor(t, []int64{0, 2}, nil)
		inputs["target"] = intTensor(t, []int64{2}, []int64{2, 0})
		outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
		if mode == 0 {
			if outputs != nil || !errors.Is(runErr, ErrInvalidModel) {
				t.Fatalf("copy-zero mode should reject changed element count: %v, %v", outputs, runErr)
			}

			inputs["target"] = intTensor(t, []int64{2}, []int64{0, 2})
			outputs, runErr = session.Run(context.Background(), inputs, RunOptions{})
		}

		if runErr != nil || len(outputs) != 1 || outputs[0].Tensor.Len() != 0 {
			t.Fatalf("dynamic zero mode %d recovery = %v, %v", mode, outputs, runErr)
		}

		assertFloats(t, retained[0].Tensor, []int64{2, 2}, []float32{1, 2, 3, 4})
	}
}

func TestVersionProfileOptionalOutputSlots(t *testing.T) {
	wire := modelWire(t, "profiles/models/opset17_maxpool")
	wire.Graph.Node[0].Output = append(wire.Graph.Node[0].Output, "")
	model := loadedModel(t, modelBytes(t, wire), Options{})
	session := modelSession(t, model)
	if !slices.Equal(session.plan.steps[0].Outputs, []int{1, -1}) {
		t.Fatalf("optional output positions lost: %v", session.plan.steps[0].Outputs)
	}

	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{1, 1, 2, 2}, []float32{1, 2, 3, 4})}
	outputs, err := session.Run(context.Background(), inputs, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	assertFloats(t, outputs[0].Tensor, []int64{1, 1, 1, 1}, []float32{4})
	wire.Graph.Node[0].Output[1] = "indices"
	if rejected, loadErr := LoadBytes(context.Background(), modelBytes(t, wire), Options{}); rejected != nil || !errors.Is(loadErr, ErrUnsupported) {
		t.Fatalf("named indices output = %v, %v", rejected, loadErr)
	}
}

func TestVersionProfileShapeWorkspaceAndDiagnostics(t *testing.T) {
	wire := modelWire(t, "profiles/models/opset17_shape")
	wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{m3Attribute("start", 1), m3Attribute("end", 2)}
	wire.Graph.Output[0].Type.GetTensorType().Shape = nil
	model := loadedModel(t, modelBytes(t, wire), Options{})
	session := modelSession(t, model)
	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{2, 2}, []float32{1, 2, 3, 4})}
	original := session.plan.limits
	// The sliced destination has one int64, but the full rank snapshot is live.
	session.plan.limits.MaxWorkspaceBytes = 15
	outputs, err := session.Run(context.Background(), inputs, RunOptions{})
	var diagnostic *Error
	if outputs != nil || !errors.Is(err, ErrLimit) || !errors.As(err, &diagnostic) || diagnostic.Opset != 17 || diagnostic.NodeIndex != 0 {
		t.Fatalf("sliced Shape workspace = %v, %v", outputs, err)
	}

	session.plan.limits.MaxWorkspaceBytes = 16
	session.plan.limits.MaxLiveBytes = 16 + 16 + 8
	if outputs, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	values, valueErr := outputs[0].Tensor.Int64Values()
	if valueErr != nil || !slices.Equal(values, []int64{2}) {
		t.Fatalf("sliced shape = %v, %v", values, valueErr)
	}

	session.plan.limits = original
	wire.Graph.Node[0].Attribute = append(wire.Graph.Node[0].Attribute, m3Attribute("start", 1))
	if rejected, loadErr := LoadBytes(context.Background(), modelBytes(t, wire), Options{}); rejected != nil || !errors.Is(loadErr, ErrInvalidModel) || !strings.Contains(loadErr.Error(), "opset 17") {
		t.Fatalf("graph-level actual import diagnostic = %v, %v", rejected, loadErr)
	}
}

func TestVersionProfileProvenanceRejectsWrongSchema(t *testing.T) {
	fixture := readProfileCases(t)[0]
	data, err := fixture.Model.read()
	if err != nil {
		t.Fatal(err)
	}

	fixture.Schemas[0].Version = 12
	if err = fixture.validateEnvelope(data); err == nil {
		t.Fatal("plausible older schema incorrectly accepted as the applicable schema")
	}
}
