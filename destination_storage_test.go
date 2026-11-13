package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func staticDestinationWire(t *testing.T) (out *onnxpb.ModelProto) {
	t.Helper()
	wire := reuseWire(t)
	for _, info := range []*onnxpb.ValueInfoProto{wire.Graph.Input[0], wire.Graph.Output[0]} {
		info.Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}
	}
	return wire
}

func assertDestinationState(t *testing.T, session *Session) (objects int) {
	t.Helper()
	seen := make(map[*tensor.Tensor]bool)
	var bytes int64
	for _, value := range session.destinations.values {
		if value == nil {
			continue
		}

		if seen[value] {
			t.Fatal("two slots own the same destination")
		}

		seen[value] = true
		bytes += value.ByteSize()
		objects++
	}
	if bytes != session.destinations.bytes || bytes > session.destinations.limit {
		t.Fatalf("retained accounting: %d, %+v", bytes, session.destinations)
	}

	return objects
}

func TestDestinationStorageSuccessAccountingAndClose(t *testing.T) {
	model := loadedModel(t, modelBytes(t, staticDestinationWire(t)), Options{})
	session := modelSession(t, model)
	if session.destinations.bytes != 0 || len(session.destinations.values) != len(model.plan.steps) {
		t.Fatal("session eagerly allocated payload or omitted bounded slots")
	}

	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{1, 2})}
	var retained []NamedTensor
	for repeat := 0; repeat < 10; repeat++ {
		run, err := newEvaluation(context.Background(), session.plan)
		if err != nil {
			t.Fatal(err)
		}

		run.storage.destinations = &session.destinations
		run.budget.live += session.destinations.bytes
		outputs, runErr := run.run(inputs, RunOptions{})
		if runErr != nil {
			t.Fatal(runErr)
		}

		if repeat == 0 {
			retained = outputs
		}

		assertFloats(t, outputs[0].Tensor, []int64{1, 2}, []float32{6, 12})
		if assertDestinationState(t, session) != 3 || session.destinations.bytes != 24 || run.budget.live != session.destinations.bytes+24 || run.budget.workspace != 0 {
			t.Fatalf("residency/invocation accounting: %+v, %+v", session.destinations, run.budget)
		}
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := session.Run(context.Background(), inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if session.destinations.bytes != 0 || session.destinations.values != nil || model.plan != nil {
		t.Fatal("Close retained session storage")
	}

	assertFloats(t, retained[0].Tensor, []int64{1, 2}, []float32{6, 12})
}

func TestDestinationStoragePoisonedPayloadsAreOverwritten(t *testing.T) {
	for _, fixture := range []string{"m1/add_matching", "m1/add_broadcast", "m1/matmul_batch", "m1/matmul_empty_contraction", "m1/matmul_empty_output", "m1/relu_finite", "m1/relu_empty", "m3/models/conv_same_upper_bias", "m3/models/max_dilated"} {
		t.Run(fixture, func(t *testing.T) {
			wire := modelWire(t, fixture)
			inputs := numericalInputs(t, wire)
			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			reference, err := session.Run(context.Background(), inputs, RunOptions{})
			if err != nil {
				t.Fatal(err)
			}

			pointers := slices.Clone(session.destinations.values)
			for _, value := range pointers {
				if value == nil {
					continue
				}

				poison := make([]float32, value.Len())
				for index := range poison {
					poison[index] = math.Float32frombits(0x7fc12345)
				}
				source := floatTensor(t, value.Shape(), poison)
				if err = tensor.AddInto(source.value, source.value, value); err != nil {
					t.Fatal(err)
				}
			}
			got, runErr := session.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatal(runErr)
			}

			if !slices.Equal(pointers, session.destinations.values) {
				t.Fatal("warm call replaced a slot")
			}

			for index, output := range got {
				want, valueErr := reference[index].Tensor.Float32Values()
				if valueErr != nil {
					t.Fatal(valueErr)
				}

				assertFloats(t, output.Tensor, reference[index].Tensor.Shape(), want)
			}
			assertDestinationState(t, session)
		})
	}
}

func TestDestinationStorageCancellationDeadlinesAndFailure(t *testing.T) {
	for _, kind := range []string{"cancel", "deadline"} {
		t.Run(kind, func(t *testing.T) {
			session := modelSession(t, loadedModel(t, modelBytes(t, staticDestinationWire(t)), Options{}))
			inputs := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{1, 2})}
			baseline := newCancelContext(0)
			retained, err := session.Run(baseline, inputs, RunOptions{})
			baseline.cancel()
			if err != nil {
				t.Fatal(err)
			}

			pointers := slices.Clone(session.destinations.values)
			for boundary := 1; boundary <= baseline.checks; boundary++ {
				cancellation := newCancelContext(boundary)
				var ctx context.Context = cancellation
				want := context.Canceled
				if kind == "deadline" {
					ctx = newDeadlineContext(cancellation)
					want = context.DeadlineExceeded
				}

				outputs, runErr := session.Run(ctx, inputs, RunOptions{})
				cancellation.cancel()
				if outputs != nil || !errors.Is(runErr, want) {
					t.Fatalf("boundary %d: %v", boundary, runErr)
				}

				outputs, runErr = session.Run(context.Background(), inputs, RunOptions{})
				if runErr != nil {
					t.Fatal(runErr)
				}

				assertFloats(t, outputs[0].Tensor, []int64{1, 2}, []float32{6, 12})
				if !slices.Equal(pointers, session.destinations.values) {
					t.Fatal("cancelled call lost bounded storage")
				}

				assertDestinationState(t, session)
			}
			// Inject a final handler error after earlier private destinations were written.
			// Public execution plans are immutable; this exercises returned-error cleanup.
			last := len(session.plan.steps) - 1
			call := session.plan.calls[last]
			session.plan.calls[last] = nil
			session.plan.steps[last].Operator = "missing"
			if outputs, runErr := session.Run(context.Background(), inputs, RunOptions{}); outputs != nil || !errors.Is(runErr, ErrUnsupported) {
				t.Fatalf("late failure: %v", runErr)
			}

			session.plan.calls[last] = call
			session.plan.steps[last].Operator = "Add"
			recovery := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{3, 4})}
			var recovered []NamedTensor
			if recovered, err = session.Run(context.Background(), recovery, RunOptions{}); err != nil {
				t.Fatal(err)
			}

			assertFloats(t, recovered[0].Tensor, []int64{1, 2}, []float32{18, 24})
			assertDestinationState(t, session)

			assertFloats(t, retained[0].Tensor, []int64{1, 2}, []float32{6, 12})
			if err = session.Close(); err != nil {
				t.Fatal(err)
			}

			if session.destinations.bytes != 0 || session.destinations.values != nil {
				t.Fatal("Close after cancellation/failure retained storage")
			}
		})
	}
}

func TestDestinationStoragePartialReluCancellationRecovers(t *testing.T) {
	wire := modelWire(t, "m1/relu_finite")
	const count = 4096
	for _, info := range []*onnxpb.ValueInfoProto{wire.Graph.Input[0], wire.Graph.Output[0]} {
		info.Type.GetTensorType().Shape.Dim = []*onnxpb.TensorShapeProto_Dimension{{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: count}}}
	}
	session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	values := make([]float32, count)
	for index := range values {
		values[index] = float32(index + 1)
	}
	inputs := map[string]*Tensor{wire.Graph.Input[0].GetName(): floatTensor(t, []int64{count}, values)}
	baseline := newCancelContext(0)
	if _, err := session.Run(baseline, inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	baseline.cancel()
	destination := session.destinations.values[0]
	if destination == nil {
		t.Fatal("large ReLU was not retained")
	}

	negative := make([]float32, count)
	for index := range negative {
		negative[index] = -values[index]
	}
	cancelledInputs := map[string]*Tensor{wire.Graph.Input[0].GetName(): floatTensor(t, []int64{count}, negative)}
	for boundary := 1; boundary <= baseline.checks; boundary++ {
		ctx := newCancelContext(boundary)
		outputs, err := session.Run(ctx, cancelledInputs, RunOptions{})
		ctx.cancel()
		if outputs != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("partial boundary %d: %v", boundary, err)
		}

		outputs, err = session.Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		assertFloats(t, outputs[0].Tensor, []int64{count}, values)
		if session.destinations.values[0] != destination {
			t.Fatal("partial write replaced destination")
		}
	}
}

func TestDestinationStorageQuotaEquivalenceAndAdmission(t *testing.T) {
	data := modelBytes(t, staticDestinationWire(t))
	for limit := int64(8); limit <= 160; limit++ {
		options := Options{Limits: Limits{MaxTensorBytes: 8, MaxWorkspaceBytes: 32, MaxLiveBytes: limit}}
		model := loadedModel(t, data, options)
		candidate, control := modelSession(t, model), modelSession(t, model)
		control.destinations = destinationStorage{}
		inputs := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{1, 2})}
		for repeat := 0; repeat < 3; repeat++ {
			got, gotErr := candidate.Run(context.Background(), inputs, RunOptions{})
			want, wantErr := control.Run(context.Background(), inputs, RunOptions{})
			if !reflect.DeepEqual(gotErr, wantErr) || fmt.Sprint(gotErr) != fmt.Sprint(wantErr) || len(got) != len(want) {
				t.Fatalf("limit %d call %d: %v, control %v", limit, repeat, gotErr, wantErr)
			}

			if gotErr == nil {
				assertFloats(t, got[0].Tensor, []int64{1, 2}, []float32{6, 12})
			}

			assertDestinationState(t, candidate)
		}
		if model.plan.destinationLimit > 0 && candidate.destinations.bytes == 0 {
			t.Fatalf("admitted model at %d retained no storage", limit)
		}
	}
}

func TestDestinationStorageDynamicShapesDoNotAccumulate(t *testing.T) {
	session := modelSession(t, loadedModel(t, modelBytes(t, reuseWire(t)), Options{}))
	for _, batch := range []int64{1, 3, 0, 2, 1, 10, 0} {
		values := make([]float32, batch*2)
		expected := make([]float32, len(values))
		for index := range values {
			values[index] = float32(index)
			expected[index] = 6 * values[index]
		}
		inputs := map[string]*Tensor{"x": floatTensor(t, []int64{batch, 2}, values)}
		outputs, err := session.Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		assertFloats(t, outputs[0].Tensor, []int64{batch, 2}, expected)
		if session.destinations.bytes != 0 || session.destinations.values != nil {
			t.Fatal("dynamic shape accumulated retained storage")
		}
	}
}

func TestDestinationStorageRejectsReplacementAndWrongGeometry(t *testing.T) {
	storage := newDestinationStorage(8, 0, 1)
	budget, err := newRunBudget(Limits{MaxTensorBytes: 16, MaxLiveBytes: 32, MaxWorkspaceBytes: 16}, 0)
	if err != nil {
		t.Fatal(err)
	}

	first := floatTensor(t, []int64{2}, []float32{1, 2})
	storage.keep(0, first.value, budget)
	for _, test := range []struct {
		dataType DataType
		shape    []int64
	}{{Float32, []int64{1, 2}}, {Float32, []int64{1}}, {Int64, []int64{1}}} {
		if storage.get(0, test.dataType, test.shape) != nil {
			t.Fatal("incompatible type/shape borrowed")
		}
	}
	storage.keep(0, floatTensor(t, []int64{1}, []float32{7}).value, budget)
	if storage.get(0, Float32, []int64{2}) != first.value || storage.bytes != 8 || budget.live != 8 {
		t.Fatal("slot replaced or accounting changed")
	}

	storage.keep(1, first.value, budget)
	if storage.bytes != 8 {
		t.Fatal("slot bound exceeded")
	}
}

func TestDestinationStorageConcurrentSessionsAndBusy(t *testing.T) {
	model := loadedModel(t, modelBytes(t, staticDestinationWire(t)), Options{})
	first, second := modelSession(t, model), modelSession(t, model)
	inputs := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{1, 2})}
	for _, session := range []*Session{first, second} {
		if _, err := session.Run(context.Background(), inputs, RunOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	for index, value := range first.destinations.values {
		if value != nil && value == second.destinations.values[index] {
			t.Fatal("sessions share writable destinations")
		}
	}
	ctx := newBlockingContext()
	defer ctx.releaseRun()
	finished := make(chan error, 1)
	go func() { _, err := first.Run(ctx, inputs, RunOptions{}); finished <- err }()
	<-ctx.entered
	if _, err := first.Run(context.Background(), inputs, RunOptions{}); err != ErrBusy {
		t.Fatalf("overlap: %v", err)
	}

	if err := first.Close(); err != ErrBusy {
		t.Fatalf("active Close: %v", err)
	}

	if _, err := second.Run(context.Background(), inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	ctx.releaseRun()
	if err := <-finished; err != nil {
		t.Fatal(err)
	}

	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := second.Run(context.Background(), inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	for cycle := 0; cycle < 5; cycle++ {
		session := modelSession(t, model)
		if _, err := session.Run(context.Background(), inputs, RunOptions{}); err != nil {
			t.Fatal(err)
		}

		if err := session.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestDestinationStorageSimultaneousSessions(t *testing.T) {
	model := loadedModel(t, modelBytes(t, staticDestinationWire(t)), Options{})
	finished := make(chan error, 2)
	start := make(chan struct{})
	sessions := []*Session{modelSession(t, model), modelSession(t, model)}
	for index, session := range sessions {
		inputs := map[string]*Tensor{"x": floatTensor(t, []int64{1, 2}, []float32{float32(index + 1), float32(index + 2)})}
		go func() {
			<-start
			for repeat := 0; repeat < 100; repeat++ {
				outputs, err := session.Run(context.Background(), inputs, RunOptions{})
				if err != nil {
					finished <- err
					return
				}

				values, valueErr := outputs[0].Tensor.Float32Values()
				if valueErr != nil || !slices.Equal(values, []float32{6 * float32(index+1), 6 * float32(index+2)}) {
					finished <- fmt.Errorf("concurrent values: %v, %v", values, valueErr)
					return
				}
			}
			finished <- nil
		}()
	}
	close(start)
	for range sessions {
		if err := <-finished; err != nil {
			t.Fatal(err)
		}
	}
	for index, value := range sessions[0].destinations.values {
		if value != nil && value == sessions[1].destinations.values[index] {
			t.Fatal("concurrent sessions retained shared storage")
		}
	}
}

func TestDestinationStorageMNISTBoundAndLifetimeMap(t *testing.T) {
	model, err := LoadFile(context.Background(), ".cache/compare-runtime/mnist-ir8-opset13.onnx", Options{})
	if errors.Is(err, os.ErrNotExist) {
		t.Skip("prepare optional MNIST fixture with tools/compare_runtime.py")
	}

	if err != nil {
		t.Fatal(err)
	}

	defer model.Close()
	session := modelSession(t, model)
	inputs := map[string]*Tensor{"Input3": floatTensor(t, []int64{1, 1, 28, 28}, make([]float32, 784))}
	if _, err = session.Run(context.Background(), inputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}

	if assertDestinationState(t, session) != 9 || session.destinations.bytes != 83664 || session.destinations.limit != 121296 {
		t.Fatalf("MNIST retained bound: %+v", session.destinations)
	}

	t.Logf("resident=%d load=%d destination_limit=%d minimum_live=%d", model.plan.residentBytes, model.plan.loadBytes, model.plan.destinationLimit, model.plan.destinationMinimum)
	for position, step := range model.plan.steps {
		index := step.Outputs[0]
		t.Logf("node=%02d %s shape=%v slot=%v retire=%v", position, step.Operator, model.plan.values[index].Info.Dimensions, session.destinations.values[position] != nil, model.plan.lifetimes.retire[position])
	}
	// Static default/initializer overrides keep the same shapes; all retained
	// immutable defaults and prepared weight Reshape remain separate destinations.
	for position, call := range model.plan.calls {
		if call.retained != nil && session.destinations.values[position] != nil {
			t.Fatal("immutable retained reshape became mutable")
		}
	}
}

func TestDestinationStorageStaticDefaultsOverridesAndOutputAliases(t *testing.T) {
	wire := staticDestinationWire(t)
	weight := proto.Clone(wire.Graph.Input[0]).(*onnxpb.ValueInfoProto)
	weight.Name = proto.String("weight")
	wire.Graph.Input = append(wire.Graph.Input, weight)
	wire.Graph.Initializer = append(wire.Graph.Initializer, &onnxpb.TensorProto{
		Name: proto.String("weight"), DataType: proto.Int32(1), Dims: []int64{1, 2}, FloatData: []float32{7, 11},
	})
	wire.Graph.Node[0].Input[1] = "weight"
	for _, name := range []string{"a", "alias"} {
		output := proto.Clone(wire.Graph.Output[0]).(*onnxpb.ValueInfoProto)
		output.Name = proto.String(name)
		wire.Graph.Output = append(wire.Graph.Output, output)
	}
	model := loadedModel(t, modelBytes(t, wire), Options{})
	session := modelSession(t, model)
	input := floatTensor(t, []int64{1, 2}, []float32{1, 2})
	override := floatTensor(t, []int64{1, 2}, []float32{10, 20})
	var retained [][]NamedTensor
	for repeat := 0; repeat < 6; repeat++ {
		inputs := map[string]*Tensor{"x": input}
		expected, alias := []float32{12, 21}, []float32{8, 13}
		if repeat%2 != 0 {
			inputs["weight"] = override
			expected, alias = []float32{15, 30}, []float32{11, 22}
		}

		outputs, err := session.Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		assertFloats(t, outputs[0].Tensor, []int64{1, 2}, expected)
		for _, output := range outputs[1:] {
			assertFloats(t, output.Tensor, []int64{1, 2}, alias)
		}
		for left, output := range outputs {
			for _, previous := range outputs[:left] {
				if output.Tensor.value == previous.Tensor.value {
					t.Fatal("published aliases share mutable storage")
				}
			}
			for _, destination := range session.destinations.values {
				if destination == output.Tensor.value {
					t.Fatal("public output borrows reusable destination")
				}
			}
		}
		if assertDestinationState(t, session) == 0 {
			t.Fatal("static override graph did not retain destinations")
		}

		retained = append(retained, outputs)
	}
	source := session.plan.values[session.plan.indices["weight"]].Default.Borrow()
	assertFloats(t, &Tensor{value: source}, []int64{1, 2}, []float32{7, 11})
	assertFloats(t, input, []int64{1, 2}, []float32{1, 2})
	assertFloats(t, override, []int64{1, 2}, []float32{10, 20})
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}

	if err := model.Close(); err != nil {
		t.Fatal(err)
	}

	for index, outputs := range retained {
		expected := []float32{12, 21}
		if index%2 != 0 {
			expected = []float32{15, 30}
		}

		assertFloats(t, outputs[0].Tensor, []int64{1, 2}, expected)
	}
}
