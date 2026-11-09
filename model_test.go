package onnx

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func modelFixture(t *testing.T, name string) (data []byte) {
	t.Helper()
	var err error
	if data, err = os.ReadFile("testdata/" + name + ".onnx"); err != nil {
		t.Fatal(err)
	}

	return data
}

func modelWire(t *testing.T, name string) (out *onnxpb.ModelProto) {
	t.Helper()
	var model onnxpb.ModelProto
	if err := proto.Unmarshal(modelFixture(t, name), &model); err != nil {
		t.Fatal(err)
	}

	return &model
}

func modelBytes(t *testing.T, wire *onnxpb.ModelProto) (data []byte) {
	t.Helper()
	var err error
	if data, err = proto.Marshal(wire); err != nil {
		t.Fatal(err)
	}

	return data
}

func loadedModel(t *testing.T, data []byte, options Options) (out *Model) {
	t.Helper()
	var err error
	if out, err = LoadBytes(context.Background(), data, options); err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if closeErr := out.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	return out
}

func TestLoadEntrypoints(t *testing.T) {
	data := modelFixture(t, "profile/identity")
	tests := []struct {
		name string
		load func() (*Model, error)
	}{
		{"bytes", func() (out *Model, err error) { return LoadBytes(context.Background(), data, Options{}) }},
		{"reader", func() (out *Model, err error) { return Load(context.Background(), bytes.NewReader(data), Options{}) }},
		{"file", func() (out *Model, err error) {
			return LoadFile(context.Background(), "testdata/profile/identity.onnx", Options{})
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := tt.load()
			if err != nil {
				t.Fatal(err)
			}

			defer model.Close()
			metadata := model.Metadata()
			if metadata.Name != "profile_identity" || metadata.Producer != "itsmontoya/onnx RT-01" || metadata.IRVersion != 8 || metadata.Opset != 13 || metadata.Inputs[0].Name != "x" || metadata.Outputs[0].Name != "y" {
				t.Fatalf("metadata: %+v", metadata)
			}

			session := modelSession(t, model)
			input := floatTensor(t, []int64{1, 2}, []float32{1, -2})
			outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{})
			if err != nil || len(outputs) != 1 || outputs[0].Name != "y" {
				t.Fatalf("run: %v, %v", outputs, err)
			}

			assertFloats(t, outputs[0].Tensor, []int64{1, 2}, []float32{1, -2})
		})
	}
}

func TestLoadValidationAndCauses(t *testing.T) {
	data := modelFixture(t, "profile/identity")
	readFailure := errors.New("source failure")
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name    string
		ctx     context.Context
		options Options
		failure error
		want    error
	}{
		{"nil context", nil, Options{}, nil, ErrInvalidArgument},
		{"negative limit", context.Background(), Options{Limits: Limits{MaxSessions: -1}}, nil, ErrInvalidArgument},
		{"cancelled", cancelled, Options{}, nil, context.Canceled},
		{"io cause", context.Background(), Options{}, readFailure, readFailure},
		{"external resolver", context.Background(), Options{ExternalData: &testResolver{}}, nil, ErrUnsupported},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			source := &loadReader{data: data, failure: tt.failure}
			model, err := Load(tt.ctx, source, tt.options)
			if model != nil || !errors.Is(err, tt.want) || source.closed {
				t.Fatalf("load: %v, %v; reader closed %v", model, err, source.closed)
			}

			if tt.failure == nil && source.reads != 0 {
				t.Fatalf("invalid load read source %d times", source.reads)
			}

			var diagnostic *Error
			if !errors.As(err, &diagnostic) || diagnostic.Stage != "load" || !strings.HasPrefix(err.Error(), "onnx load") {
				t.Fatalf("load stage: %v", err)
			}
		})
	}

	if model, err := Load(context.Background(), nil, Options{}); model != nil || err != ErrInvalidArgument {
		t.Fatalf("nil reader: %v, %v", model, err)
	}

	if model, err := LoadBytes(context.Background(), []byte{0x80}, Options{}); model != nil || !errors.Is(err, ErrInvalidModel) {
		t.Fatalf("malformed wire: %v, %v", model, err)
	}

	if model, err := LoadBytes(context.Background(), nil, Options{}); model != nil || !errors.Is(err, ErrInvalidModel) {
		t.Fatalf("empty model envelope: %v, %v", model, err)
	}

	missing := filepath.Join(t.TempDir(), "missing.onnx")
	if model, err := LoadFile(context.Background(), missing, Options{}); model != nil || !errors.Is(err, os.ErrNotExist) || !strings.Contains(err.Error(), "open model") {
		t.Fatalf("file cause: %v, %v", model, err)
	}

	if model, err := LoadFile(nil, missing, Options{}); model != nil || !errors.Is(err, ErrInvalidArgument) || errors.Is(err, os.ErrNotExist) {
		t.Fatalf("file pre-I/O validation: %v, %v", model, err)
	}

	if model, err := LoadFile(context.Background(), "", Options{}); model != nil || err != ErrInvalidArgument {
		t.Fatalf("empty path: %v, %v", model, err)
	}
}

func TestLoadRejectsWholeModel(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*onnxpb.ModelProto)
		want      error
		node      bool
	}{
		{"opset", func(model *onnxpb.ModelProto) { model.OpsetImport[0].Version = proto.Int64(14) }, ErrUnsupported, false},
		{"ir", func(model *onnxpb.ModelProto) { model.IrVersion = proto.Int64(9) }, ErrUnsupported, false},
		{"missing input reference", func(model *onnxpb.ModelProto) { model.Graph.Node[0].Input[0] = "absent" }, ErrInvalidModel, false},
		{"unused planned node", func(model *onnxpb.ModelProto) {
			model.Graph.Node = append(model.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Flatten"), Input: []string{"x"}, Output: []string{"unused"}})
		}, ErrUnsupported, true},
		{"unknown attribute", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("extra"), Type: onnxpb.AttributeProto_INT.Enum()}}
		}, ErrUnsupported, true},
		{"external tensor", func(model *onnxpb.ModelProto) {
			model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("stored"), DataType: proto.Int32(1), DataLocation: onnxpb.TensorProto_EXTERNAL.Enum(), ExternalData: []*onnxpb.StringStringEntryProto{{Key: proto.String("location"), Value: proto.String("weights")}}}}
		}, ErrUnsupported, false},
		{"dynamic nonleading input", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0].Type.GetTensorType().Shape.Dim[1].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "M"}
		}, ErrUnsupported, false},
		{"unknown input rank", func(model *onnxpb.ModelProto) { model.Graph.Input[0].Type.GetTensorType().Shape = nil }, ErrUnsupported, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "profile/identity")
			tt.configure(wire)
			model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{})
			var diagnostic *Error
			if model != nil || !errors.Is(err, tt.want) || !errors.As(err, &diagnostic) || diagnostic.Stage != "load" {
				t.Fatalf("load rejection: %v, %v", model, err)
			}

			if tt.node && (diagnostic.NodeIndex < 0 || diagnostic.Operator == "" || diagnostic.Opset != 13) {
				t.Fatalf("node context lost: %+v", diagnostic)
			}
		})
	}
}

func TestModelLifecycleAndMetadata(t *testing.T) {
	model := loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{Limits: Limits{MaxSessions: 1}})
	before := model.Metadata()
	before.Inputs[0].Name = "changed"
	before.Inputs[0].Dimensions[0].Symbol = "changed"
	before.Outputs[0].Dimensions[0].Kind = FixedDimension
	metadata := model.Metadata()
	if metadata.Inputs[0].Name != "x" || metadata.Inputs[0].Dimensions[0].Symbol != "N" || metadata.Outputs[0].Dimensions[0].Kind != SymbolicDimension || !metadata.Inputs[1].Defaulted {
		t.Fatalf("metadata aliases or lacks default: %+v", metadata)
	}

	session := modelSession(t, model)
	if extra, err := model.NewSession(); extra != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("session cap: %v, %v", extra, err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	session = modelSession(t, model)
	plan := session.plan
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}

	if err := model.Close(); err != nil {
		t.Fatal(err)
	}

	if model.plan != nil || !reflect.DeepEqual(model.Metadata(), metadata) {
		t.Fatal("close retained plan or lost metadata")
	}

	if extra, err := model.NewSession(); extra != nil || err != ErrClosed {
		t.Fatalf("new session after close: %v, %v", extra, err)
	}

	input := floatTensor(t, []int64{1, 2}, []float32{2, 3})
	if outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{}); err != nil || len(outputs) != 5 {
		t.Fatalf("existing session after model close: %v, %v", outputs, err)
	}

	if err := session.Close(); err != nil || plan.sessions != 0 {
		t.Fatalf("close after model close did not release slot: %v, %d", err, plan.sessions)
	}

	var zero Model
	for _, invalid := range []*Model{nil, &zero} {
		if !reflect.DeepEqual(invalid.Metadata(), Metadata{}) || invalid.Close() != ErrInvalidArgument {
			t.Fatal("invalid model metadata/close")
		}

		if created, err := invalid.NewSession(); created != nil || err != ErrInvalidArgument {
			t.Fatalf("invalid model session: %v, %v", created, err)
		}
	}
}

func TestModelCloseSessionRace(t *testing.T) {
	data := modelFixture(t, "sessions/empty")
	for repeat := 0; repeat < 40; repeat++ {
		model := loadedModel(t, data, Options{})
		var (
			session   *Session
			createErr error
			closeErr  error
			wait      sync.WaitGroup
		)

		start := make(chan struct{})
		wait.Add(2)
		go func() { defer wait.Done(); <-start; session, createErr = model.NewSession() }()
		go func() { defer wait.Done(); <-start; closeErr = model.Close() }()
		close(start)
		wait.Wait()
		if closeErr != nil {
			t.Fatal(closeErr)
		}

		if createErr == ErrClosed {
			if session != nil {
				t.Fatal("closed model created session")
			}

			continue
		}

		if createErr != nil || session == nil {
			t.Fatalf("creation race: %v, %v", session, createErr)
		}

		if _, err := session.Run(context.Background(), nil, RunOptions{}); err != nil {
			t.Fatalf("winning session unusable: %v", err)
		}

		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLoadReaderBoundsOwnershipAndRecovery(t *testing.T) {
	data := modelFixture(t, "profile/identity")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tests := []struct {
		name    string
		reader  *loadReader
		ctx     context.Context
		options Options
		want    error
	}{
		{"exact bytes", &loadReader{data: data, chunk: 3}, context.Background(), Options{Limits: Limits{MaxModelBytes: int64(len(data))}}, nil},
		{"one byte over", &loadReader{data: data}, context.Background(), Options{Limits: Limits{MaxModelBytes: int64(len(data) - 1)}}, ErrLimit},
		{"no progress", &loadReader{noProgress: true}, context.Background(), Options{}, io.ErrNoProgress},
		{"invalid reader count", &loadReader{invalidCount: -1}, context.Background(), Options{}, ErrInvalidArgument},
		{"cancel during read", &loadReader{data: data, chunk: 1, cancel: cancel}, ctx, Options{}, context.Canceled},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := Load(tt.ctx, tt.reader, tt.options)
			if tt.want == nil {
				if model == nil || err != nil {
					t.Fatalf("exact load: %v, %v", model, err)
				}

				if err = model.Close(); err != nil {
					t.Fatal(err)
				}
			} else if model != nil || !errors.Is(err, tt.want) {
				t.Fatalf("bounded load: %v, %v", model, err)
			}

			if tt.reader.closed || tt.reader.largest > 32<<10 {
				t.Fatalf("reader ownership/chunk: %+v", tt.reader)
			}

			if tt.options.Limits.MaxModelBytes > 0 && int64(tt.reader.consumed) > tt.options.Limits.MaxModelBytes+1 {
				t.Fatal("reader consumed beyond limit plus one")
			}
		})
	}
}

func TestLoadActualWireLengthAndExactLoadBudget(t *testing.T) {
	canonical := modelFixture(t, "profile/identity")
	// IR 8 uses a deliberately overlong varint. Re-marshalling this valid wire
	// removes one byte and cannot reproduce the amount consumed by Load.
	if canonical[0] != 8 || canonical[1] != 8 {
		t.Fatal("fixture no longer starts with IR version 8")
	}

	noncanonical := append([]byte{8, 0x88, 0}, canonical[2:]...)
	baseline := loadedModel(t, canonical, Options{})
	encoded := loadedModel(t, noncanonical, Options{})
	if encoded.plan.loadBytes <= baseline.plan.loadBytes {
		t.Fatal("loader did not account original noncanonical wire length")
	}

	for _, loader := range []struct {
		name string
		load func([]byte, Options) (*Model, error)
	}{
		{"bytes", func(data []byte, options Options) (out *Model, err error) {
			return LoadBytes(context.Background(), data, options)
		}},
		{"reader", func(data []byte, options Options) (out *Model, err error) {
			return Load(context.Background(), bytes.NewReader(data), options)
		}},
	} {
		t.Run(loader.name, func(t *testing.T) {
			for _, tt := range []struct {
				name  string
				data  []byte
				limit int64
				want  error
			}{
				{"exact canonical", canonical, baseline.plan.loadBytes, nil},
				{"canonical one below", canonical, baseline.plan.loadBytes - 1, ErrLimit},
				{"noncanonical exceeds canonical budget", noncanonical, baseline.plan.loadBytes, ErrLimit},
				{"exact noncanonical", noncanonical, encoded.plan.loadBytes, nil},
			} {
				t.Run(tt.name, func(t *testing.T) {
					model, err := loader.load(tt.data, Options{Limits: Limits{MaxLoadBytes: tt.limit}})
					if tt.want != nil {
						if model != nil || !errors.Is(err, tt.want) {
							t.Fatalf("load budget: %v, %v", model, err)
						}

						return
					}

					if model == nil || err != nil {
						t.Fatalf("exact load budget: %v, %v", model, err)
					}

					if err = model.Close(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

func TestLoadConfiguredSemanticLimits(t *testing.T) {
	data := modelFixture(t, "sessions/branch_defaults")
	tests := []struct {
		name   string
		limits Limits
	}{
		{"initializer aggregate", Limits{MaxInitializerBytes: 23}},
		{"tensor bytes", Limits{MaxTensorBytes: 15}},
		{"node count", Limits{MaxNodes: 5}},
		{"value count", Limits{MaxValues: 7}},
		{"name bytes", Limits{MaxNameBytes: 1}},
		{"metadata rank", Limits{MaxRank: 1}},
		{"dimension", Limits{MaxDimension: 1}},
		{"recursion", Limits{MaxRecursion: 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model, err := LoadBytes(context.Background(), data, Options{Limits: tt.limits})
			if model != nil || err == nil {
				t.Fatalf("load ignored limit: %v, %v", model, err)
			}

			// Protobuf recursion failures retain their invalid-wire
			// classification; other configured semantic limits are ErrLimit.
			want := ErrLimit
			if tt.limits.MaxRecursion != 0 {
				want = ErrInvalidModel
			}

			if !errors.Is(err, want) {
				t.Fatalf("limit classification: %v", err)
			}
		})
	}

	loadedModel(t, data, Options{Limits: Limits{MaxInitializerBytes: 24, MaxNodes: 6, MaxValues: 8, MaxAttributes: 1}})
}

var _ io.Reader = (*loadReader)(nil)

func TestLoadCancellationAtEveryBoundary(t *testing.T) {
	data := modelFixture(t, "profile/identity")
	baseline := newCancelContext(0)
	defer baseline.cancel()
	model, err := Load(baseline, bytes.NewReader(data), Options{})
	if err != nil {
		t.Fatal(err)
	}

	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	for boundary := 1; boundary <= baseline.checks; boundary++ {
		ctx := newCancelContext(boundary)
		result, loadErr := Load(ctx, bytes.NewReader(data), Options{})
		ctx.cancel()
		if result != nil || !errors.Is(loadErr, context.Canceled) {
			t.Fatalf("load boundary %d: %v, %v", boundary, result, loadErr)
		}
	}
}
