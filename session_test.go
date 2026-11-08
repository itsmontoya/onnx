package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func modelSession(t *testing.T, model *Model) (out *Session) {
	t.Helper()
	var err error
	if out, err = model.NewSession(); err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if closeErr := out.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	return out
}

func floatTensor(t *testing.T, shape []int64, values []float32) (out *Tensor) {
	t.Helper()
	var err error
	if out, err = NewFloat32(shape, values); err != nil {
		t.Fatal(err)
	}

	return out
}

func intTensor(t *testing.T, shape []int64, values []int64) (out *Tensor) {
	t.Helper()
	var err error
	if out, err = NewInt64(shape, values); err != nil {
		t.Fatal(err)
	}

	return out
}

func assertFloats(t *testing.T, value *Tensor, shape []int64, expected []float32) {
	t.Helper()
	values, err := value.Float32Values()
	if err != nil || !slices.Equal(value.Shape(), shape) || len(values) != len(expected) {
		t.Fatalf("float tensor shape/values: %v, %v, %v", value.Shape(), values, err)
	}

	for index, want := range expected {
		if math.Float32bits(values[index]) != math.Float32bits(want) {
			t.Fatalf("element %d bits = %08x, want %08x", index, math.Float32bits(values[index]), math.Float32bits(want))
		}
	}
}

func TestSessionBranchDefaultsAndOutputSelection(t *testing.T) {
	model := loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{})
	session := modelSession(t, model)
	input := floatTensor(t, []int64{1, 2}, []float32{math.Float32frombits(0x7fc12345), math.Float32frombits(0x80000000)})
	weight := floatTensor(t, []int64{2, 2}, []float32{1, 2, 3, 4})
	tests := []struct {
		name         string
		inputs       map[string]*Tensor
		requested    []string
		names        []string
		weightShape  []int64
		weightValues []float32
	}{
		{"all defaults", map[string]*Tensor{"x": input}, nil, []string{"b", "a", "weight_out", "integer_out", "x"}, []int64{1, 2}, []float32{7, 11}},
		{"empty request", map[string]*Tensor{"x": input}, []string{}, []string{"b", "a", "weight_out", "integer_out", "x"}, []int64{1, 2}, []float32{7, 11}},
		{"override changes leading batch", map[string]*Tensor{"x": input, "weight": weight}, []string{"weight_out", "b", "integer_out"}, []string{"weight_out", "b", "integer_out"}, []int64{2, 2}, []float32{1, 2, 3, 4}},
		{"subset request order", map[string]*Tensor{"x": input}, []string{"x", "a"}, []string{"x", "a"}, nil, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputs, err := session.Run(context.Background(), tt.inputs, RunOptions{Outputs: tt.requested})
			if err != nil || len(outputs) != len(tt.names) {
				t.Fatalf("outputs: %v, %v", outputs, err)
			}

			for index, name := range tt.names {
				output := outputs[index]
				if output.Name != name || output.Tensor.value == input.value || output.Tensor.value == weight.value {
					t.Fatalf("output order/ownership: %+v", output)
				}

				switch name {
				case "b", "a", "x":
					assertFloats(t, output.Tensor, []int64{1, 2}, []float32{math.Float32frombits(0x7fc12345), math.Float32frombits(0x80000000)})
				case "weight_out":
					assertFloats(t, output.Tensor, tt.weightShape, tt.weightValues)
				case "integer_out":
					values, valueErr := output.Tensor.Int64Values()
					if valueErr != nil || !slices.Equal(values, []int64{16777217, math.MinInt64}) {
						t.Fatalf("integer precision: %v, %v", values, valueErr)
					}
				}
			}
		})
	}

	// Supplying a weight as an ordinary graph input never requires an initializer.
	wire := modelWire(t, "sessions/branch_defaults")
	wire.Graph.Initializer = nil
	withoutDefault := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	outputs, err := withoutDefault.Run(context.Background(), map[string]*Tensor{"x": input, "weight": weight}, RunOptions{Outputs: []string{"weight_out"}})
	if err != nil || len(outputs) != 1 {
		t.Fatalf("runtime graph weight: %v, %v", outputs, err)
	}

	assertFloats(t, outputs[0].Tensor, []int64{2, 2}, []float32{1, 2, 3, 4})
}

func TestSessionPassthroughScalarEmptyAndNoOutputs(t *testing.T) {
	model := loadedModel(t, modelFixture(t, "sessions/passthrough"), Options{})
	session := modelSession(t, model)
	input := intTensor(t, nil, []int64{math.MaxInt64})
	outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{})
	if err != nil || len(outputs) != 2 || outputs[0].Name != "stored" || outputs[1].Name != "x" {
		t.Fatalf("passthrough: %v, %v", outputs, err)
	}

	assertFloats(t, outputs[0].Tensor, []int64{2}, []float32{3, math.Float32frombits(0x80000000)})
	values, err := outputs[1].Tensor.Int64Values()
	if err != nil || !slices.Equal(values, []int64{math.MaxInt64}) || outputs[1].Tensor.Rank() != 0 || outputs[1].Tensor.value == input.value {
		t.Fatalf("scalar: %v, %v", values, err)
	}

	if outputs, err = session.Run(context.Background(), map[string]*Tensor{"x": input, "stored": outputs[0].Tensor}, RunOptions{}); outputs != nil || !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("non-input initializer override: %v, %v", outputs, err)
	}

	identity := modelSession(t, loadedModel(t, modelFixture(t, "profile/identity"), Options{}))
	empty := floatTensor(t, []int64{0, 2}, nil)
	if outputs, err = identity.Run(context.Background(), map[string]*Tensor{"x": empty}, RunOptions{}); err != nil || len(outputs) != 1 {
		t.Fatalf("empty: %v, %v", outputs, err)
	}

	assertFloats(t, outputs[0].Tensor, []int64{0, 2}, nil)
	noOutputs := modelSession(t, loadedModel(t, modelFixture(t, "sessions/empty"), Options{}))
	if outputs, err = noOutputs.Run(context.Background(), nil, RunOptions{}); err != nil || len(outputs) != 0 {
		t.Fatalf("no nodes or IO: %v, %v", outputs, err)
	}
}

func TestSessionInputAndOutputErrorsRecover(t *testing.T) {
	session := modelSession(t, loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{}))
	valid := floatTensor(t, []int64{1, 2}, []float32{2, 3})
	tests := []struct {
		name    string
		inputs  map[string]*Tensor
		outputs []string
		value   string
	}{
		{"missing", nil, nil, "x"},
		{"nil tensor", map[string]*Tensor{"x": nil}, nil, "x"},
		{"zero tensor", map[string]*Tensor{"x": {}}, nil, "x"},
		{"unexpected sorted", map[string]*Tensor{"x": valid, "z": valid, "aaa": valid}, nil, "aaa"},
		{"empty input name", map[string]*Tensor{"x": valid, "": valid}, nil, ""},
		{"dtype", map[string]*Tensor{"x": intTensor(t, []int64{1, 2}, []int64{2, 3})}, nil, "x"},
		{"rank", map[string]*Tensor{"x": floatTensor(t, []int64{2}, []float32{2, 3})}, nil, "x"},
		{"fixed axis", map[string]*Tensor{"x": floatTensor(t, []int64{1, 1}, []float32{2})}, nil, "x"},
		{"duplicate requested output", map[string]*Tensor{"x": valid}, []string{"a", "a"}, "a"},
		{"intermediate request", map[string]*Tensor{"x": valid}, []string{"middle"}, "middle"},
		{"unknown requested output", map[string]*Tensor{"x": valid}, []string{"nope"}, "nope"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputs, err := session.Run(context.Background(), tt.inputs, RunOptions{Outputs: tt.outputs})
			var diagnostic *Error
			if outputs != nil || !errors.Is(err, ErrInvalidArgument) || !errors.As(err, &diagnostic) || diagnostic.Stage != "run" || diagnostic.Value != tt.value {
				t.Fatalf("invalid run: %v, %v", outputs, err)
			}

			if outputs, err = session.Run(context.Background(), map[string]*Tensor{"x": valid}, RunOptions{}); err != nil || len(outputs) != 5 {
				t.Fatalf("recovery: %v, %v", outputs, err)
			}
		})
	}
}

func TestSessionSymbolBindingsAndOriginalDeclarations(t *testing.T) {
	wire := modelWire(t, "sessions/branch_defaults")
	wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
	wire.Graph.Output[2].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
	session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	tests := []struct {
		name        string
		batch       int64
		weightBatch int64
		override    bool
		want        error
	}{
		{"default binds", 1, 0, false, nil},
		{"default disagrees", 2, 0, false, ErrInvalidArgument},
		{"supplied symbols disagree", 2, 1, true, ErrInvalidArgument},
		{"override agrees", 2, 2, true, nil},
		{"zero binds afresh", 0, 0, true, nil},
		{"later batch binds afresh", 3, 3, true, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inputs := map[string]*Tensor{"x": floatTensor(t, []int64{tt.batch, 2}, make([]float32, tt.batch*2))}
			if tt.override {
				inputs["weight"] = floatTensor(t, []int64{tt.weightBatch, 2}, make([]float32, tt.weightBatch*2))
			}

			outputs, err := session.Run(context.Background(), inputs, RunOptions{})
			if tt.want != nil {
				if outputs != nil || !errors.Is(err, tt.want) || !strings.Contains(err.Error(), `symbol "N"`) {
					t.Fatalf("symbol mismatch: %v, %v", outputs, err)
				}

				return
			}

			if err != nil || len(outputs) != 5 || outputs[0].Tensor.Shape()[0] != tt.batch {
				t.Fatalf("fresh bindings: %v, %v", outputs, err)
			}
		})
	}

	// Identity inference refines the output N to fixed 1. N's original output
	// declaration must still reject a separate input that bound N to 2.
	wire = modelWire(t, "sessions/branch_defaults")
	wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}
	wire.Graph.Output[4].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}
	wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
	wire.Graph.Output[2].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
	refined := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	inputs := map[string]*Tensor{
		"x":      floatTensor(t, []int64{1, 2}, []float32{2, 3}),
		"weight": floatTensor(t, []int64{2, 2}, []float32{1, 2, 3, 4}),
	}
	outputs, err := refined.Run(context.Background(), inputs, RunOptions{Outputs: []string{"weight_out"}})
	var diagnostic *Error
	if outputs != nil || !errors.Is(err, ErrInvalidModel) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 1 || diagnostic.Value != "a" || !strings.Contains(err.Error(), `symbol "N"`) {
		t.Fatalf("lost original output declaration on unused path: %v, %v", outputs, err)
	}

	inputs["weight"] = floatTensor(t, []int64{1, 2}, []float32{4, 5})
	if outputs, err = refined.Run(context.Background(), inputs, RunOptions{}); err != nil || len(outputs) != 5 {
		t.Fatalf("declaration failure poisoned session: %v, %v", outputs, err)
	}
}

func TestSessionUnknownMetadataAndRuntimeRank(t *testing.T) {
	wire := modelWire(t, "profile/identity")
	wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = nil
	wire.Graph.Output[0].Type.GetTensorType().Shape = nil
	model := loadedModel(t, modelBytes(t, wire), Options{Limits: Limits{MaxRank: 2}})
	metadata := model.Metadata()
	if metadata.Inputs[0].Dimensions[0].Kind != UnknownDimension || metadata.Outputs[0].ShapeKnown {
		t.Fatalf("unknown metadata changed: %+v", metadata)
	}

	session := modelSession(t, model)
	for _, batch := range []int64{0, 2, 1} {
		input := floatTensor(t, []int64{batch, 2}, make([]float32, batch*2))
		outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{})
		if err != nil || len(outputs) != 1 || !slices.Equal(outputs[0].Tensor.Shape(), []int64{batch, 2}) {
			t.Fatalf("unknown shape resolution: %v, %v", outputs, err)
		}
	}

	tooManyAxes := floatTensor(t, []int64{0, 2, 2}, nil)
	if outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": tooManyAxes}, RunOptions{}); outputs != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("runtime rank limit: %v, %v", outputs, err)
	}
}

func TestSessionOriginalIntermediateConstraint(t *testing.T) {
	wire := modelWire(t, "sessions/branch_defaults")
	// Remove the x passthrough declaration's N before making x fixed. A
	// separately bound weight supplies N; middle's own declaration must survive
	// Identity's concrete inference and be checked even when unrequested.
	wire.Graph.Output = wire.Graph.Output[:4]
	wire.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}
	wire.Graph.Input[1].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
	declaration := wire.Graph.Output[0]
	declaration.Name = proto.String("middle")
	wire.Graph.Output = wire.Graph.Output[1:]
	wire.Graph.ValueInfo = []*onnxpb.ValueInfoProto{declaration}
	session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	inputs := map[string]*Tensor{
		"x":      floatTensor(t, []int64{1, 2}, []float32{1, 2}),
		"weight": floatTensor(t, []int64{2, 2}, []float32{1, 2, 3, 4}),
	}
	outputs, err := session.Run(context.Background(), inputs, RunOptions{Outputs: []string{"integer_out"}})
	var diagnostic *Error
	if outputs != nil || !errors.Is(err, ErrInvalidModel) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 0 || diagnostic.Value != "middle" {
		t.Fatalf("original intermediate constraint lost: %v, %v", outputs, err)
	}
}

func TestSessionOutputsSurviveAndConstantsStayPrivate(t *testing.T) {
	data := modelFixture(t, "sessions/branch_defaults")
	model := loadedModel(t, data, Options{})
	session := modelSession(t, model)
	firstInput := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	first, err := session.Run(context.Background(), map[string]*Tensor{"x": firstInput}, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	if first[0].Tensor.value == first[1].Tensor.value || first[0].Tensor.value == first[4].Tensor.value {
		t.Fatal("requested aliases share an output allocation")
	}

	clear(data)
	secondInput := floatTensor(t, []int64{1, 2}, []float32{8, 9})
	second, err := session.Run(context.Background(), map[string]*Tensor{"x": secondInput}, RunOptions{})
	if err != nil || second[0].Tensor.value == first[0].Tensor.value {
		t.Fatalf("later output allocation: %v", err)
	}

	assertFloats(t, first[0].Tensor, []int64{1, 2}, []float32{1, 2})
	// Internal destination access proves that neither the constant nor a later
	// output aliases an earlier output. Public callers cannot perform this write.
	if err = tensor.AddInto(secondInput.value, secondInput.value, first[2].Tensor.value); err != nil {
		t.Fatal(err)
	}

	third, err := session.Run(context.Background(), map[string]*Tensor{"x": secondInput}, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	assertFloats(t, second[2].Tensor, []int64{1, 2}, []float32{7, 11})
	assertFloats(t, third[2].Tensor, []int64{1, 2}, []float32{7, 11})
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	assertFloats(t, first[0].Tensor, []int64{1, 2}, []float32{1, 2})
	assertFloats(t, third[0].Tensor, []int64{1, 2}, []float32{8, 9})
}

func TestSessionBusyAndSeparateSessions(t *testing.T) {
	model := loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{Limits: Limits{MaxSessions: 2}})
	first, second := modelSession(t, model), modelSession(t, model)
	input := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	ctx := newBlockingContext()
	defer ctx.releaseRun()
	finished := make(chan error, 1)
	go func() {
		_, err := first.Run(ctx, map[string]*Tensor{"x": input}, RunOptions{})
		finished <- err
	}()
	select {
	case <-ctx.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("run did not enter context check")
	}
	if outputs, err := first.Run(context.Background(), nil, RunOptions{}); outputs != nil || err != ErrBusy {
		t.Fatalf("overlapping run: %v, %v", outputs, err)
	}

	if err := first.Close(); err != ErrBusy {
		t.Fatalf("close while active: %v", err)
	}

	if outputs, err := second.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{}); err != nil || len(outputs) != 5 {
		t.Fatalf("second session blocked by first: %v, %v", outputs, err)
	}

	if err := model.Close(); err != nil {
		t.Fatal(err)
	}

	ctx.releaseRun()
	select {
	case err := <-finished:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run did not finish")
	}
}

func TestSessionStateAndOperatorFailureRecovery(t *testing.T) {
	var zero Session
	for _, invalid := range []*Session{nil, &zero} {
		if outputs, err := invalid.Run(context.Background(), nil, RunOptions{}); outputs != nil || err != ErrInvalidArgument || invalid.Close() != ErrInvalidArgument {
			t.Fatalf("invalid session: %v, %v", outputs, err)
		}
	}

	model := loadedModel(t, modelFixture(t, "profile/identity"), Options{})
	session := modelSession(t, model)
	if outputs, err := session.Run(nil, nil, RunOptions{}); outputs != nil || err != ErrInvalidArgument {
		t.Fatalf("nil run context: %v, %v", outputs, err)
	}

	input := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	// A compiled public model cannot contain this handler. Exercise the
	// dispatch failure path without adding an unsupported public capability.
	session.plan.steps[0].Operator = "missing"
	outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{})
	var diagnostic *Error
	if outputs != nil || !errors.Is(err, ErrUnsupported) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != 0 || diagnostic.NodeName != "copy" || diagnostic.Operator != "missing" || diagnostic.Opset != 13 || !reflect.DeepEqual(diagnostic.Shapes, [][]int64{{1, 2}}) {
		t.Fatalf("operator failure: %v, %v", outputs, err)
	}

	session.plan.steps[0].Operator = "Identity"
	if outputs, err = session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{}); err != nil || len(outputs) != 1 {
		t.Fatalf("operator failure recovery: %v, %v", outputs, err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	if err = session.Close(); err != nil {
		t.Fatal(err)
	}

	if outputs, err = session.Run(context.Background(), nil, RunOptions{}); outputs != nil || err != ErrClosed {
		t.Fatalf("run after close: %v, %v", outputs, err)
	}
}

func TestSessionLiveAndTensorBudgetsRecover(t *testing.T) {
	tests := []struct {
		name       string
		fixture    string
		limits     Limits
		largeBatch int64
		validBatch int64
	}{
		// x and the Identity alias each charge 8 bytes, followed by an owned
		// 8-byte requested output. Earlier outputs are outside the next run.
		{"live output copy", "profile/identity", Limits{MaxLiveBytes: 24}, 2, 1},
		{"runtime tensor", "profile/identity", Limits{MaxTensorBytes: 8}, 2, 1},
		// Empty tensors still validate every dimension against its limit.
		{"runtime dimension", "profile/identity", Limits{MaxDimension: 2}, 3, 0},
		// Resident weights+Constant=24, bound x/default=16, four float table
		// aliases=32, Constant and integer alias=32, requested b=8: 112.
		{"resident constants and aliases", "sessions/branch_defaults", Limits{MaxLiveBytes: 112}, 2, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			session := modelSession(t, loadedModel(t, modelFixture(t, tt.fixture), Options{Limits: tt.limits}))
			large := floatTensor(t, []int64{tt.largeBatch, 2}, make([]float32, tt.largeBatch*2))
			options := RunOptions{Outputs: []string{"y"}}
			if tt.fixture != "profile/identity" {
				options.Outputs = []string{"b"}
			}

			if outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": large}, options); outputs != nil || !errors.Is(err, ErrLimit) {
				t.Fatalf("run ignored budget: %v, %v", outputs, err)
			}

			valid := floatTensor(t, []int64{tt.validBatch, 2}, make([]float32, tt.validBatch*2))
			for repeat := 0; repeat < 3; repeat++ {
				if outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": valid}, options); err != nil || len(outputs) != 1 {
					t.Fatalf("budget failure recovery %d: %v, %v", repeat, outputs, err)
				}
			}
		})
	}

	// All node outputs remain live even when only b is requested; a one-byte
	// lower bound fails after nodes, rather than hiding unused intermediates.
	session := modelSession(t, loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{Limits: Limits{MaxLiveBytes: 111}}))
	input := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	if outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": input}, RunOptions{Outputs: []string{"b"}}); outputs != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("resident/intermediate/output budget one below: %v, %v", outputs, err)
	}
}

func TestSessionConcurrentIndependentRuns(t *testing.T) {
	model := loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{Limits: Limits{MaxSessions: 2, MaxWorkspaceBytes: 1}})
	first, second := modelSession(t, model), modelSession(t, model)
	inputs := []*Tensor{
		floatTensor(t, []int64{1, 2}, []float32{1, 2}),
		floatTensor(t, []int64{2, 2}, []float32{3, 4, 5, 6}),
	}
	start := make(chan struct{})
	results := make(chan error, 2)
	for index, session := range []*Session{first, second} {
		go func() {
			<-start
			for repeat := 0; repeat < 30; repeat++ {
				outputs, err := session.Run(context.Background(), map[string]*Tensor{"x": inputs[index]}, RunOptions{})
				if err != nil {
					results <- err
					return
				}

				values, valueErr := outputs[0].Tensor.Float32Values()
				if valueErr != nil || len(values) != (index+1)*2 || values[0] != float32(index*2+1) {
					results <- fmt.Errorf("session %d shared data: %v, %v", index, values, valueErr)
					return
				}
			}
			results <- nil
		}()
	}
	close(start)
	for complete := 0; complete < 2; complete++ {
		if err := <-results; err != nil {
			t.Fatal(err)
		}
	}
}

func TestSessionCancellationAtEveryBoundaryAndRecovery(t *testing.T) {
	session := modelSession(t, loadedModel(t, modelFixture(t, "sessions/branch_defaults"), Options{}))
	input := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	inputs := map[string]*Tensor{"x": input}
	baseline := newCancelContext(0)
	defer baseline.cancel()
	if _, err := session.Run(baseline, inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	// Includes checks around all six nodes, binding defaults, every requested
	// output copy and the final publication boundary. No partial results escape.
	for boundary := 1; boundary <= baseline.checks; boundary++ {
		ctx := newCancelContext(boundary)
		outputs, err := session.Run(ctx, inputs, RunOptions{})
		ctx.cancel()
		if outputs != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("run boundary %d: %v, %v", boundary, outputs, err)
		}

		if outputs, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil || len(outputs) != 5 {
			t.Fatalf("recovery after boundary %d: %v, %v", boundary, outputs, err)
		}
	}

	deadline, cancel := context.WithDeadline(context.Background(), time.Unix(0, 0))
	defer cancel()
	if outputs, err := session.Run(deadline, inputs, RunOptions{}); outputs != nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline cause: %v, %v", outputs, err)
	}
}

func TestSessionBudgetRejectionBeforeNumericAllocation(t *testing.T) {
	const payloadBytes int64 = 1 << 20
	tests := []struct {
		name     string
		operator string
		limits   Limits
		node     int
	}{
		{"scratch", "Relu", Limits{MaxWorkspaceBytes: 1024}, 0},
		{"intermediate", "Relu", Limits{MaxLiveBytes: payloadBytes + 1024}, 0},
		{"output clone", "Identity", Limits{MaxLiveBytes: 2*payloadBytes + 1024}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "profile/identity")
			wire.Graph.Node[0].OpType = proto.String(tt.operator)
			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{Limits: tt.limits}))
			small := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{3, 4})}
			retained, err := session.Run(context.Background(), small, RunOptions{})
			if err != nil {
				t.Fatal(err)
			}

			large := map[string]*Tensor{"x": floatTensor(t, []int64{payloadBytes / 8, 2}, make([]float32, payloadBytes/4))}
			var failed []NamedTensor
			result := testing.Benchmark(func(b *testing.B) {
				for b.Loop() {
					failed, err = session.Run(context.Background(), large, RunOptions{})
				}
			})
			var diagnostic *Error
			if failed != nil || !errors.Is(err, ErrLimit) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != tt.node {
				t.Fatalf("budget rejection: %v, %v", failed, err)
			}

			// Any destination, scratch bridge or requested clone would allocate at
			// least a MiB. Only the small table, metadata and errors may allocate.
			if result.AllocedBytesPerOp() > 32<<10 {
				t.Fatalf("rejected run allocated numerical payload: %d bytes", result.AllocedBytesPerOp())
			}

			if _, err = session.Run(context.Background(), small, RunOptions{}); err != nil {
				t.Fatalf("budget failure left session state: %v", err)
			}

			assertFloats(t, retained[0].Tensor, []int64{1, 2}, []float32{3, 4})
		})
	}
}
