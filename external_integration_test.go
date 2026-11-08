package onnx

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func readExternalCases(t *testing.T) (cases []conformanceCase) {
	t.Helper()
	data, err := os.ReadFile("testdata/external/provenance.json")
	if err != nil {
		t.Fatal(err)
	}

	var suite struct {
		FormatVersion int                 `json:"format_version"`
		Revision      string              `json:"onnx_source_revision"`
		Source        conformanceArtifact `json:"source_model"`
		Weights       conformanceArtifact `json:"weights"`
		Cases         []conformanceCase   `json:"cases"`
		Ranges        []struct {
			Offset int    `json:"offset"`
			Bytes  int    `json:"bytes"`
			SHA256 string `json:"sha256"`
		} `json:"ranges"`
	}
	if err = json.Unmarshal(data, &suite); err != nil {
		t.Fatal(err)
	}

	if suite.FormatVersion != 1 || suite.Revision != "3bf92c03a9f27eba3bda1e5b9e63ea20ec213557" || len(suite.Cases) != 2 || len(suite.Ranges) != 4 {
		t.Fatal("external fixture provenance differs")
	}

	if _, err = suite.Source.read(); err != nil {
		t.Fatal(err)
	}

	weights, err := suite.Weights.read()
	if err != nil {
		t.Fatal(err)
	}

	for _, selected := range suite.Ranges {
		if selected.Offset < 0 || selected.Bytes < 0 || selected.Offset > len(weights) || selected.Bytes > len(weights)-selected.Offset {
			t.Fatal("invalid external range provenance")
		}

		digest := sha256.Sum256(weights[selected.Offset : selected.Offset+selected.Bytes])
		if hex.EncodeToString(digest[:]) != selected.SHA256 {
			t.Fatal("external range digest differs")
		}
	}
	return suite.Cases
}

func TestExternalMLPEntrypointsAndOwnership(t *testing.T) {
	cases := readExternalCases(t)
	data, err := cases[0].Model.read()
	if err != nil {
		t.Fatal(err)
	}

	for _, entrypoint := range []string{"bytes", "reader", "file"} {
		t.Run(entrypoint, func(t *testing.T) {
			var (
				model   *Model
				loadErr error
			)

			resolver, createErr := NewDirectoryResolver("testdata/external/models")
			if createErr != nil {
				t.Fatal(createErr)
			}

			defer resolver.Close()
			options := Options{ExternalData: resolver}
			switch entrypoint {
			case "bytes":
				model, loadErr = LoadBytes(context.Background(), data, options)
			case "reader":
				model, loadErr = Load(context.Background(), bytes.NewReader(data), options)
			case "file":
				model, loadErr = LoadFile(context.Background(), "testdata/external/models/mlp.onnx", options)
			}
			if loadErr != nil {
				t.Fatal(loadErr)
			}

			defer model.Close()
			if closeErr := resolver.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}

			session := modelSession(t, model)
			var retained [][]NamedTensor
			for _, index := range []int{0, 1, 0} {
				fixture := cases[index]
				inputs, inputErr := fixture.readInputs()
				if inputErr != nil {
					t.Fatal(inputErr)
				}

				outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
				if runErr != nil {
					t.Fatal(runErr)
				}

				if compareErr := fixture.compareOutputs(outputs); compareErr != nil {
					t.Fatal(compareErr)
				}

				retained = append(retained, outputs)
			}
			if closeErr := session.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}

			if closeErr := model.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}

			for index, fixtureIndex := range []int{0, 1, 0} {
				if compareErr := cases[fixtureIndex].compareOutputs(retained[index]); compareErr != nil {
					t.Fatal(compareErr)
				}
			}
		})
	}
	for _, entrypoint := range []string{"bytes", "reader", "file"} {
		var model *Model
		switch entrypoint {
		case "bytes":
			model, err = LoadBytes(context.Background(), data, Options{})
		case "reader":
			model, err = Load(context.Background(), bytes.NewReader(data), Options{})
		case "file":
			model, err = LoadFile(context.Background(), "testdata/external/models/mlp.onnx", Options{})
		}
		if model != nil || !errors.Is(err, ErrUnsupported) || !strings.Contains(err.Error(), "hidden_weight") || !strings.Contains(err.Error(), "weights.bin") {
			t.Fatalf("explicit resolver required for %s: %v, %v", entrypoint, model, err)
		}
	}
}

func TestExternalConcurrentLoadsAndSessions(t *testing.T) {
	cases := readExternalCases(t)
	data, err := cases[0].Model.read()
	if err != nil {
		t.Fatal(err)
	}

	resolver, err := NewDirectoryResolver("testdata/external/models")
	if err != nil {
		t.Fatal(err)
	}

	defer resolver.Close()
	var workers sync.WaitGroup
	for index := range 8 {
		workers.Add(1)
		go func(position int) {
			defer workers.Done()
			model, loadErr := LoadBytes(context.Background(), data, Options{ExternalData: resolver})
			if loadErr != nil {
				t.Error(loadErr)
				return
			}

			defer model.Close()
			session, sessionErr := model.NewSession()
			if sessionErr != nil {
				t.Error(sessionErr)
				return
			}

			defer session.Close()
			fixture := cases[position%2]
			inputs, inputErr := fixture.readInputs()
			if inputErr != nil {
				t.Error(inputErr)
				return
			}

			outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Error(runErr)
				return
			}

			if compareErr := fixture.compareOutputs(outputs); compareErr != nil {
				t.Error(compareErr)
			}
		}(index)
	}
	workers.Wait()
}

func externalScalar(name, location, offset string, dataType int32) (out *onnxpb.TensorProto) {
	var value onnxpb.TensorProto
	value.Name, value.DataType, value.DataLocation = proto.String(name), proto.Int32(dataType), onnxpb.TensorProto_EXTERNAL.Enum()
	value.ExternalData = []*onnxpb.StringStringEntryProto{
		{Key: proto.String("location"), Value: proto.String(location)},
		{Key: proto.String("offset"), Value: proto.String(offset)},
	}
	return &value
}

func scalarInfo(name string, dataType int32) (out *onnxpb.ValueInfoProto) {
	var info onnxpb.ValueInfoProto
	info.Name = proto.String(name)
	info.Type = &onnxpb.TypeProto{Value: &onnxpb.TypeProto_TensorType{TensorType: &onnxpb.TypeProto_Tensor{ElemType: proto.Int32(dataType), Shape: &onnxpb.TensorShapeProto{}}}}
	return &info
}

func TestExternalSharedRangesLocationsAndConstants(t *testing.T) {
	wire := modelWire(t, "profile/identity")
	wire.Graph.Input = nil
	wire.Graph.Initializer = []*onnxpb.TensorProto{
		externalScalar("a", "shared", "3", 1),
		externalScalar("b", "shared", "3", 1),
		externalScalar("c", "other", "0", 1),
	}
	wire.Graph.Node = []*onnxpb.NodeProto{{OpType: proto.String("Constant"), Output: []string{"i"}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: externalScalar("integer", "shared", "7", 7)}}}}
	wire.Graph.Output = []*onnxpb.ValueInfoProto{scalarInfo("a", 1), scalarInfo("b", 1), scalarInfo("c", 1), scalarInfo("i", 7)}
	shared := make([]byte, 15)
	binary.LittleEndian.PutUint32(shared[3:], math.Float32bits(1.25))
	binary.LittleEndian.PutUint64(shared[7:], uint64(1<<60|7))
	other := make([]byte, 4)
	binary.LittleEndian.PutUint32(other, math.Float32bits(-2))
	resolver := &weightResolver{values: map[string][]byte{"shared": shared, "other": other}}
	if rejected, loadErr := LoadBytes(context.Background(), modelBytes(t, wire), Options{ExternalData: resolver, Limits: Limits{MaxInitializerBytes: 19}}); rejected != nil || !errors.Is(loadErr, ErrLimit) {
		t.Fatalf("initializer/Constant aggregate limit: %v, %v", rejected, loadErr)
	}

	model := loadedModel(t, modelBytes(t, wire), Options{ExternalData: resolver, Limits: Limits{MaxInitializerBytes: 20}})
	clear(shared)
	clear(other)
	session := modelSession(t, model)
	for range 2 {
		outputs, err := session.Run(context.Background(), nil, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		assertFloats(t, outputs[0].Tensor, nil, []float32{1.25})
		assertFloats(t, outputs[1].Tensor, nil, []float32{1.25})
		assertFloats(t, outputs[2].Tensor, nil, []float32{-2})
		integers, readErr := outputs[3].Tensor.Int64Values()
		if readErr != nil || len(integers) != 1 || integers[0] != 1<<60|7 {
			t.Fatalf("external exact integer: %v, %v", integers, readErr)
		}

		values, readErr := outputs[0].Tensor.Float32Values()
		if readErr != nil {
			t.Fatal(readErr)
		}

		values[0] = 999
	}
}

func TestExternalStorageFailuresHaveContext(t *testing.T) {
	wire := modelWire(t, "external/models/mlp")
	directory := t.TempDir()
	resolver, err := NewDirectoryResolver(directory)
	if err != nil {
		t.Fatal(err)
	}

	defer resolver.Close()
	for _, path := range []string{"weights.bin", "../escape", "/absolute", "https://example.com/weights"} {
		wire.Graph.Initializer[0].ExternalData[0].Value = proto.String(path)
		model, loadErr := LoadBytes(context.Background(), modelBytes(t, wire), Options{ExternalData: resolver})
		if model != nil || loadErr == nil || !strings.Contains(loadErr.Error(), "hidden_weight") || !strings.Contains(loadErr.Error(), path) {
			t.Fatalf("storage context: %v, %v", model, loadErr)
		}

		if path == "weights.bin" && !errors.Is(loadErr, os.ErrNotExist) {
			t.Fatalf("storage cause lost: %v", loadErr)
		}
	}
	wire.Graph.Initializer[0].ExternalData[0].Value = proto.String("weights.bin")
	if err = os.WriteFile(filepath.Join(directory, "weights.bin"), make([]byte, 2), 0600); err != nil {
		t.Fatal(err)
	}

	model, loadErr := LoadBytes(context.Background(), modelBytes(t, wire), Options{ExternalData: resolver})
	if model != nil || !errors.Is(loadErr, ErrInvalidArgument) || !strings.Contains(loadErr.Error(), "weights.bin") {
		t.Fatalf("truncated directory range: %v, %v", model, loadErr)
	}

	shortResolver := &weightResolver{values: map[string][]byte{"weights.bin": make([]byte, 2)}}
	if model, loadErr = LoadBytes(context.Background(), modelBytes(t, wire), Options{ExternalData: shortResolver}); model != nil || loadErr == nil {
		t.Fatalf("short range: %v, %v", model, loadErr)
	}

}
