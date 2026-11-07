package onnx

import (
	"context"
	"encoding/json"
	"errors"
	"maps"
	"os"
	"slices"
	"testing"
)

func readInferenceCases(t *testing.T) (cases []conformanceCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/inference/provenance.json")
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
	if suite.FormatVersion != 1 || suite.ONNXSourceRevision != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || !maps.Equal(suite.ReferenceVersions, versions) || suite.OriginalLicense == "" || len(suite.Cases) != 6 {
		t.Fatal("inference provenance differs")
	}

	return suite.Cases
}

func TestInferenceFixtures(t *testing.T) {
	cases := readInferenceCases(t)
	for _, graph := range []string{"mlp", "branch", "shape"} {
		t.Run(graph, func(t *testing.T) {
			var fixtures []conformanceCase
			for _, fixture := range cases {
				if fixture.Model.Path == "inference/models/"+graph+".onnx" {
					fixtures = append(fixtures, fixture)
				}
			}
			if len(fixtures) != 2 {
				t.Fatal("each graph must have two reference batches")
			}

			runInferenceBatches(t, graph, fixtures)
		})
	}
}

func TestInferenceConcurrentSessions(t *testing.T) {
	cases := readInferenceCases(t)
	for _, graph := range []string{"mlp", "branch", "shape"} {
		t.Run(graph, func(t *testing.T) {
			var fixtures []conformanceCase
			for _, fixture := range cases {
				if fixture.Model.Path == "inference/models/"+graph+".onnx" {
					fixtures = append(fixtures, fixture)
				}
			}
			if len(fixtures) != 2 {
				t.Fatal("each graph must have two reference batches")
			}

			runConcurrentInference(t, fixtures)
		})
	}
}

func runConcurrentInference(t *testing.T, fixtures []conformanceCase) {
	t.Helper()
	data, err := fixtures[0].Model.read()
	if err != nil {
		t.Fatal(err)
	}

	model := loadedModel(t, data, Options{})
	var (
		sessions = make([]*Session, len(fixtures))
		bindings = make([]map[string]*Tensor, len(fixtures))
		results  = make([][]NamedTensor, len(fixtures))
		failures = make(chan error, len(fixtures))
		start    = make(chan struct{})
	)

	for index, fixture := range fixtures {
		sessions[index] = modelSession(t, model)
		if bindings[index], err = fixture.readInputs(); err != nil {
			t.Fatal(err)
		}
	}
	for index := range fixtures {
		go func(position int) {
			<-start
			var runErr error
			results[position], runErr = sessions[position].Run(context.Background(), bindings[position], RunOptions{})
			failures <- runErr
		}(index)
	}
	close(start)
	for range fixtures {
		err = errors.Join(err, <-failures)
	}
	if err != nil {
		t.Fatal(err)
	}

	for index, fixture := range fixtures {
		if err = fixture.compareOutputs(results[index]); err != nil {
			t.Fatal(err)
		}
	}
}

func runInferenceBatches(t *testing.T, graph string, fixtures []conformanceCase) {
	t.Helper()
	model, err := LoadFile(context.Background(), "testdata/"+fixtures[0].Model.Path, Options{})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if closeErr := model.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	if graph != "shape" {
		checkInferenceMetadata(t, graph, model.Metadata())
	}

	session := modelSession(t, model)
	var retained [][]NamedTensor
	for index, fixture := range fixtures {
		outputs := runInferenceBatch(t, session, fixture, []int64{1, 3}[index])
		retained = append(retained, outputs)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	for index, fixture := range fixtures {
		if err = fixture.compareOutputs(retained[index]); err != nil {
			t.Fatalf("retained batch after later runs and Close: %v", err)
		}
	}
	if _, err = session.Run(context.Background(), nil, RunOptions{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("closed session: %v", err)
	}
}

func checkInferenceSelection(t *testing.T, session *Session, fixture conformanceCase, inputs map[string]*Tensor) {
	t.Helper()
	selected := fixture
	selected.Outputs = slices.Clone(fixture.Outputs)
	slices.Reverse(selected.Outputs)
	var names []string
	for _, output := range selected.Outputs {
		names = append(names, output.Name)
	}
	outputs, err := session.Run(context.Background(), inputs, RunOptions{Outputs: names})
	if err != nil {
		t.Fatal(err)
	}

	if err = selected.compareOutputs(outputs); err != nil {
		t.Fatal(err)
	}
}

func checkInferenceMetadata(t *testing.T, graph string, metadata Metadata) {
	t.Helper()
	inputCount, outputCount := 1, 1
	if graph == "branch" {
		inputCount, outputCount = 2, 2
	}

	if metadata.Name != graph || metadata.IRVersion != 8 || metadata.Opset != 13 || len(metadata.Inputs) != inputCount || len(metadata.Outputs) != outputCount {
		t.Fatalf("metadata differs: %+v", metadata)
	}

	for _, info := range append(metadata.Inputs, metadata.Outputs...) {
		if info.DataType != Float32 || !info.ShapeKnown || len(info.Dimensions) != 2 || info.Dimensions[0].Kind != SymbolicDimension || info.Dimensions[0].Symbol != "batch" || info.Dimensions[1].Kind != FixedDimension || info.Dimensions[1].Value != 2 {
			t.Fatalf("dynamic leading batch metadata differs: %+v", info)
		}
	}
}

func runInferenceBatch(t *testing.T, session *Session, fixture conformanceCase, batch int64) (outputs []NamedTensor) {
	t.Helper()
	data, err := fixture.Model.read()
	if err != nil {
		t.Fatal(err)
	}

	if err = fixture.validateEnvelope(data); err != nil {
		t.Fatal(err)
	}

	inputs, inputErr := fixture.readInputs()
	if inputErr != nil {
		t.Fatal(inputErr)
	}

	shape := []int64{batch, 2}
	if fixture.Model.Path == "inference/models/shape.onnx" {
		shape = []int64{batch, 2, 3}
	}

	if !slices.Equal(inputs["features"].Shape(), shape) || fixture.Status != "agreement" {
		t.Fatal("fixture batch or reference status differs")
	}

	if outputs, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	if err = fixture.compareOutputs(outputs); err != nil {
		t.Fatal(err)
	}

	checkInferenceSelection(t, session, fixture, inputs)
	return outputs
}
