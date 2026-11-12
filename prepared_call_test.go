package onnx

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestPreparedCallStaticEligibilityAndCancellation(t *testing.T) {
	for _, fixture := range []string{
		"m1/add_broadcast", "m1/add_matching", "m1/sub_matching", "m1/mul_matching", "m1/div_matching",
		"m1/matmul_batch", "m1/matmul_dot", "m1/matmul_empty_contraction", "m1/matmul_empty_output",
		"m1/relu_finite", "m1/relu_empty", "m1/sigmoid_finite", "m1/tanh_finite", "m1/softmax_axis_1",
		"m3/models/conv_same_upper_bias", "m3/models/max_dilated", "m3/models/average_ceil_partial_include",
		"m3/models/global_default", "m3/models/bn_rank4_epsilon",
	} {
		t.Run(fixture, func(t *testing.T) {
			wire := modelWire(t, fixture)
			inputs := numericalInputs(t, wire)
			prepared := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control.plan.calls = nil
			if len(prepared.plan.calls) != 1 || prepared.plan.calls[0] == nil {
				t.Fatal("concrete call was not prepared")
			}

			assertPreparedRun(t, prepared, control, inputs)
			for _, call := range prepared.plan.calls {
				if call.cnn != nil && call.cnn.inputs != nil || call.numerical != nil && call.numerical.inputs != nil {
					t.Fatal("descriptor retained runtime operands")
				}
			}
		})
	}
}

func TestPreparedCallImmutableReshape(t *testing.T) {
	for _, source := range []string{"initializer", "constant", "identity"} {
		t.Run(source, func(t *testing.T) {
			wire := modelWire(t, "m2/models/reshape_zero_infer_float32")
			target := &onnxpb.TensorProto{Name: proto.String("shape"), DataType: proto.Int32(7), Dims: []int64{2}, Int64Data: []int64{0, -1}}
			wire.Graph.Input = wire.Graph.Input[:1]
			wire.Graph.Initializer = []*onnxpb.TensorProto{target}
			if source == "constant" {
				wire.Graph.Initializer = nil
				wire.Graph.Node = append([]*onnxpb.NodeProto{{OpType: proto.String("Constant"), Output: []string{"shape"}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: target}}}}, wire.Graph.Node...)
			}

			if source == "identity" {
				target.Name = proto.String("target")
				wire.Graph.Node = append([]*onnxpb.NodeProto{{OpType: proto.String("Identity"), Input: []string{"target"}, Output: []string{"shape"}}}, wire.Graph.Node...)
			}

			prepared := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control.plan.calls = nil
			call := prepared.plan.calls[len(prepared.plan.steps)-1]
			if call == nil || call.shape == nil || call.shape.inputs != nil || !slices.Equal(call.shape.shape, []int64{2, 12}) {
				t.Fatal("immutable target was not prepared without operands")
			}

			for _, value := range []float32{1, 5, -3} {
				data := make([]float32, 24)
				for index := range data {
					data[index] = value + float32(index)
				}
				inputs := map[string]*Tensor{"x": floatTensor(t, []int64{2, 3, 4}, data)}
				assertPreparedRun(t, prepared, control, inputs)
			}
			for _, workspace := range []int64{15, 16} {
				prepared.plan.limits.MaxWorkspaceBytes, control.plan.limits.MaxWorkspaceBytes = workspace, workspace
				inputs := map[string]*Tensor{"x": floatTensor(t, []int64{2, 3, 4}, make([]float32, 24))}
				assertPreparedRun(t, prepared, control, inputs)
			}
		})
	}
}

func TestPreparedCallReshapeDefaultsRemainDynamic(t *testing.T) {
	for _, malformed := range []bool{false, true} {
		t.Run(fmt.Sprint(malformed), func(t *testing.T) {
			wire := modelWire(t, "m2/models/reshape_zero_infer_float32")
			defaults := []int64{0, -1}
			if malformed {
				defaults = []int64{-1, -1}
			}

			wire.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("shape"), DataType: proto.Int32(7), Dims: []int64{2}, Int64Data: defaults}}
			prepared := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control.plan.calls = nil
			if len(prepared.plan.calls) != 0 {
				t.Fatal("overridable shape contents were prepared")
			}

			for _, target := range [][]int64{nil, nil, {0, -1}, {4, 6}, {-1, -1}, nil, {2, 12}, {24}} {
				inputs := map[string]*Tensor{"x": floatTensor(t, []int64{2, 3, 4}, make([]float32, 24))}
				if target != nil {
					inputs["shape"] = intTensor(t, []int64{int64(len(target))}, target)
				}

				assertPreparedRun(t, prepared, control, inputs)
			}
		})
	}
}

func TestPreparedCallDynamicFallback(t *testing.T) {
	for _, fixture := range []string{"m1/add_broadcast", "m1/matmul_batch", "m3/models/conv_same_upper_bias", "m3/models/max_dilated"} {
		t.Run(fixture, func(t *testing.T) {
			wire := modelWire(t, fixture)
			m3DynamicInputs(wire)
			prepared := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control.plan.calls = nil
			if len(prepared.plan.calls) != 0 {
				t.Fatal("unknown dimensions were specialized")
			}

			original := numericalInputs(t, modelWire(t, fixture))
			for _, batch := range []int64{1, 3, 2, 1} {
				inputs := make(map[string]*Tensor)
				for name, input := range original {
					shape := input.Shape()
					if name == "x" || name == "a" {
						shape[0] = batch
						if len(shape) == 4 {
							shape[2] += batch
							shape[3] += batch
						}
					}

					var count int64 = 1
					for _, size := range shape {
						count *= size
					}
					inputs[name+"_flat"] = floatTensor(t, []int64{count}, make([]float32, count))
					inputs[name+"_shape"] = intTensor(t, []int64{int64(len(shape))}, shape)
				}
				assertPreparedRun(t, prepared, control, inputs)
			}
		})
	}
}

func TestPreparedCallSymbolicDimensions(t *testing.T) {
	for _, symbolic := range []bool{false, true} {
		t.Run(fmt.Sprint(symbolic), func(t *testing.T) {
			wire := modelWire(t, "m1/add_matching")
			for _, info := range wire.Graph.Input {
				info.Type.GetTensorType().Shape.Dim[0].Value = nil
				if symbolic {
					info.Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
				}
			}
			wire.Graph.Output[0].Type.GetTensorType().Shape = nil
			prepared := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			control.plan.calls = nil
			if len(prepared.plan.calls) != 0 {
				t.Fatal("symbolic/unknown signature was prepared")
			}

			original := numericalInputs(t, modelWire(t, "m1/add_matching"))
			for _, batch := range []int64{1, 3, 2, 1} {
				inputs := make(map[string]*Tensor)
				for name, input := range original {
					shape := input.Shape()
					shape[0] = batch
					inputs[name] = floatTensor(t, shape, make([]float32, batch*(input.Len()/input.Shape()[0])))
				}
				if _, err := prepared.Run(context.Background(), inputs, RunOptions{}); err != nil {
					t.Fatal(err)
				}

				assertPreparedRun(t, prepared, control, inputs)
				if symbolic {
					for name, input := range original {
						inputs[name] = input
						break
					}
					assertPreparedRun(t, prepared, control, inputs)
				}
			}
		})
	}
}

func TestPreparedCallDefaultPayloadAndConcurrentLifetime(t *testing.T) {
	wire := modelWire(t, "m1/add_matching")
	inputs := numericalInputs(t, wire)
	name := wire.Graph.Input[1].GetName()
	defaults := inputs[name]
	values, err := defaults.Float32Values()
	if err != nil {
		t.Fatal(err)
	}

	wire.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String(name), DataType: proto.Int32(1), Dims: defaults.Shape(), FloatData: values}}
	model := loadedModel(t, modelBytes(t, wire), Options{})
	sessions := []*Session{modelSession(t, model), modelSession(t, model)}
	control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	control.plan.calls = nil
	if len(model.plan.calls) != 1 || model.plan.calls[0] == nil {
		t.Fatal("shape-invariant default was not prepared")
	}

	for _, offset := range []float32{0, 0, 2, -1, 0, 4} {
		current := make(map[string]*Tensor)
		for inputName, input := range inputs {
			if inputName != name {
				current[inputName] = input
			}
		}
		if offset != 0 {
			changed := slices.Clone(values)
			for index := range changed {
				changed[index] += offset
			}
			current[name] = floatTensor(t, defaults.Shape(), changed)
		}
		assertPreparedRun(t, sessions[0], control, current)
	}
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	finished := make(chan error, len(sessions))
	for _, session := range sessions {
		go func() {
			for repeat := 0; repeat < 100; repeat++ {
				if _, runErr := session.Run(context.Background(), inputs, RunOptions{}); runErr != nil {
					finished <- runErr
					return
				}
			}
			finished <- nil
		}()
	}
	for range sessions {
		if err = <-finished; err != nil {
			t.Fatal(err)
		}
	}
	if sessions[0].plan.calls[0].numerical.inputs != nil {
		t.Fatal("shared descriptor retained invocation operands")
	}
}

func TestPreparedCallLoadBudgetFallback(t *testing.T) {
	data := modelFixture(t, "m1/add_broadcast")
	baseline := loadedModel(t, data, Options{})
	minimum := baseline.plan.loadBytes - baseline.plan.callBytes
	for _, limit := range []int64{minimum, baseline.plan.loadBytes - 1, baseline.plan.loadBytes} {
		model := loadedModel(t, data, Options{Limits: Limits{MaxLoadBytes: limit}})
		if model.plan.loadBytes > limit {
			t.Fatal("prepared metadata exceeded load budget")
		}

		if limit == minimum && len(model.plan.calls) != 0 {
			t.Fatal("optional preparation did not fall back at original load limit")
		}

		session := modelSession(t, model)
		inputs := numericalInputs(t, modelWire(t, "m1/add_broadcast"))
		if _, err := session.Run(context.Background(), inputs, RunOptions{}); err != nil {
			t.Fatal(err)
		}
	}
	if model, err := LoadBytes(context.Background(), data, Options{Limits: Limits{MaxLoadBytes: minimum - 1}}); model != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("original required load budget was bypassed: %v, %v", model, err)
	}
}

func assertPreparedRun(t *testing.T, prepared, control *Session, inputs map[string]*Tensor) {
	t.Helper()
	preparedContext, controlContext := newCancelContext(0), newCancelContext(0)
	defer preparedContext.cancel()
	defer controlContext.cancel()
	got, gotErr := prepared.Run(preparedContext, inputs, RunOptions{})
	want, wantErr := control.Run(controlContext, inputs, RunOptions{})
	if fmt.Sprint(gotErr) != fmt.Sprint(wantErr) || preparedContext.checks != controlContext.checks {
		t.Fatalf("behavior/check count: %v/%d, control %v/%d", gotErr, preparedContext.checks, wantErr, controlContext.checks)
	}

	if gotErr != nil {
		if got != nil || want != nil || !reflect.DeepEqual(gotErr, wantErr) {
			t.Fatalf("error diagnostics differ: %+v, %+v", gotErr, wantErr)
		}

		return
	}

	if len(got) != len(want) {
		t.Fatal("output count changed")
	}

	for index, output := range got {
		values, err := want[index].Tensor.Float32Values()
		if err != nil || output.Name != want[index].Name || output.Tensor.value == want[index].Tensor.value {
			t.Fatalf("output identity/type: %v", err)
		}

		assertFloats(t, output.Tensor, want[index].Tensor.Shape(), values)
	}
	for boundary := 1; boundary <= controlContext.checks; boundary++ {
		ctx := newCancelContext(boundary)
		outputs, err := prepared.Run(ctx, inputs, RunOptions{})
		ctx.cancel()
		if outputs != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation boundary %d: %v", boundary, err)
		}
	}
}

func TestPreparedConstantReshapeEligibilityAndOverrides(t *testing.T) {
	for _, source := range []string{"initializer", "constant", "identity", "chain", "default", "default_identity", "default_chain", "input"} {
		for _, target := range []string{"initializer", "constant", "identity", "chain", "default", "default_identity", "default_chain", "input"} {
			t.Run(source+"/"+target, func(t *testing.T) {
				wire := preparedConstantReshapeWire(t, source, target)
				data := modelBytes(t, wire)
				prepared := modelSession(t, loadedModel(t, data, Options{}))
				control := preparedReshapeControl(t, data)
				position := len(prepared.plan.steps) - 1
				eligible := !strings.HasPrefix(source, "default") && source != "input" && !strings.HasPrefix(target, "default") && target != "input"
				var call *preparedCall
				if position < len(prepared.plan.calls) {
					call = prepared.plan.calls[position]
				}

				if (call != nil && call.retained != nil) != eligible {
					t.Fatal("constant payload eligibility differs")
				}

				for _, override := range []int{0, 1, 2, 0} {
					inputs := make(map[string]*Tensor)
					if source == "input" || strings.HasPrefix(source, "default") && override != 0 {
						values := make([]float32, 24)
						for index := range values {
							values[index] = float32(index + override)
						}
						inputs["x"] = floatTensor(t, []int64{2, 3, 4}, values)
					}

					if target == "input" || strings.HasPrefix(target, "default") && override != 0 {
						shape := []int64{0, -1}
						if override != 0 {
							shape = []int64{int64(override * 2), int64(12 / override)}
						}

						inputs["shape"] = intTensor(t, []int64{2}, shape)
					}
					assertPreparedRun(t, prepared, control, inputs)
				}
				if eligible {
					owner := prepared.plan.lifetimes.owners[prepared.plan.steps[position].Outputs[0]]
					if !prepared.plan.lifetimes.pinned[owner] || call.shape.inputs != nil {
						t.Fatal("prepared storage is writable or retains preparation arguments")
					}
				}
			})
		}
	}
}

func TestPreparedConstantReshapeMalformedDefault(t *testing.T) {
	wire := preparedConstantReshapeWire(t, "initializer", "default")
	wire.Graph.Initializer[1].Int64Data = []int64{-1, -1}
	data := modelBytes(t, wire)
	prepared := modelSession(t, loadedModel(t, data, Options{}))
	control := preparedReshapeControl(t, data)
	for _, target := range [][]int64{nil, {0, -1}, {4, 6}, nil} {
		inputs := make(map[string]*Tensor)
		if target != nil {
			inputs["shape"] = intTensor(t, []int64{2}, target)
		}

		assertPreparedRun(t, prepared, control, inputs)
	}
}

func TestPreparedConstantReshapeInvalidConstants(t *testing.T) {
	for _, test := range []struct {
		name   string
		target []int64
	}{
		{name: "negative", target: []int64{-2, 12}},
		{name: "two inferred", target: []int64{-1, -1}},
		{name: "count mismatch", target: []int64{2, 11}},
		{name: "zero beyond rank", target: []int64{2, 3, 4, 0}},
		{name: "overflow", target: []int64{math.MaxInt64, 2}},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire := preparedConstantReshapeWire(t, "initializer", "initializer")
			wire.Graph.Initializer[1].Dims = []int64{int64(len(test.target))}
			wire.Graph.Initializer[1].Int64Data = test.target
			if model, err := LoadBytes(context.Background(), modelBytes(t, wire), Options{}); model != nil || err == nil {
				t.Fatalf("malformed immutable target must retain its compile-time failure: %v, %v", model, err)
			}
		})
	}
}

func TestPreparedConstantReshapeShapesAndTypes(t *testing.T) {
	for _, test := range []struct {
		name      string
		source    []int64
		target    []int64
		allowZero bool
		integer   bool
	}{
		{name: "inferred", source: []int64{2, 3, 4}, target: []int64{0, -1}},
		{name: "empty literal", source: []int64{0, 3, 4}, target: []int64{0, 12}, allowZero: true},
		{name: "empty copied zero", source: []int64{0, 3, 4}, target: []int64{0, 12}},
		{name: "inferred zero", source: []int64{0, 3, 4}, target: []int64{-1, 12}},
		{name: "scalar", source: []int64{1}, target: nil},
		{name: "int64", source: []int64{2, 3, 4}, target: []int64{4, 6}, integer: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			wire := preparedConstantReshapeWire(t, "initializer", "initializer")
			wire.OpsetImport[0].Version = proto.Int64(17)
			wire.Graph.Initializer[0].Dims = test.source
			wire.Graph.Initializer[1].Dims = []int64{int64(len(test.target))}
			wire.Graph.Initializer[1].Int64Data = test.target
			var elements int64 = 1
			for _, size := range test.source {
				elements *= size
			}
			wire.Graph.Initializer[0].FloatData = make([]float32, elements)
			if test.allowZero {
				wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("allowzero"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)}}
			}

			if test.integer {
				wire.Graph.Initializer[0].DataType = proto.Int32(7)
				wire.Graph.Initializer[0].FloatData = nil
				wire.Graph.Initializer[0].Int64Data = make([]int64, elements)
				wire.Graph.Output[0].Type.GetTensorType().ElemType = proto.Int32(7)
			}

			data := modelBytes(t, wire)
			prepared := modelSession(t, loadedModel(t, data, Options{}))
			control := preparedReshapeControl(t, data)
			if prepared.plan.calls[0].retained == nil {
				t.Fatal("valid constant shape was not prepared")
			}

			actual, actualErr := prepared.Run(context.Background(), nil, RunOptions{})
			want, wantErr := control.Run(context.Background(), nil, RunOptions{})
			if actualErr != nil || wantErr != nil || !reflect.DeepEqual(actual[0].Tensor, want[0].Tensor) {
				t.Fatalf("constant shape/type behavior differs: %v, %v", actualErr, wantErr)
			}
		})
	}
}

func TestPreparedConstantReshapeLimits(t *testing.T) {
	wire := preparedConstantReshapeWire(t, "initializer", "initializer")
	data := modelBytes(t, wire)
	baseline := loadedModel(t, data, Options{})
	call := baseline.plan.calls[0]
	if call.retained == nil {
		t.Fatal("missing prepared result")
	}

	bytes := call.retained.ByteSize()
	loadMinimum := baseline.plan.loadBytes - bytes - int64(len(call.shape.shape))*16
	liveMinimum := baseline.plan.limits.MaxLiveBytes - baseline.plan.preparedLiveBudget(context.Background())
	for _, test := range []struct {
		name     string
		limits   Limits
		prepared bool
	}{
		{name: "exact load", limits: Limits{MaxLoadBytes: baseline.plan.loadBytes}, prepared: true},
		{name: "one byte short", limits: Limits{MaxLoadBytes: baseline.plan.loadBytes - 1}},
		{name: "previous load", limits: Limits{MaxLoadBytes: loadMinimum}},
		{name: "original required load", limits: Limits{MaxLoadBytes: loadMinimum - baseline.plan.callBytes}},
		{name: "exact conservative live", limits: Limits{MaxLiveBytes: liveMinimum}, prepared: true},
		{name: "below conservative live", limits: Limits{MaxLiveBytes: liveMinimum - 1}},
		{name: "tight runtime live", limits: Limits{MaxLiveBytes: 416, MaxWorkspaceBytes: 16}},
	} {
		t.Run(test.name, func(t *testing.T) {
			model := loadedModel(t, data, Options{Limits: test.limits})
			retained := len(model.plan.calls) != 0 && model.plan.calls[0] != nil && model.plan.calls[0].retained != nil
			if retained != test.prepared {
				t.Fatal("optional preparation budget boundary changed")
			}

			session := modelSession(t, model)
			if _, err := session.Run(context.Background(), nil, RunOptions{}); err != nil {
				t.Fatal(err)
			}
		})
	}
	if model, err := LoadBytes(context.Background(), data, Options{Limits: Limits{MaxLoadBytes: loadMinimum - baseline.plan.callBytes - 1}}); model != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("original required load limit was bypassed: %v, %v", model, err)
	}

	prepared := modelSession(t, baseline)
	control := preparedReshapeControl(t, data)
	for _, workspace := range []int64{15, 16} {
		prepared.plan.limits.MaxWorkspaceBytes, control.plan.limits.MaxWorkspaceBytes = workspace, workspace
		assertPreparedRun(t, prepared, control, nil)
	}
	for _, workspace := range []int64{15, 16} {
		limited := modelSession(t, loadedModel(t, data, Options{Limits: Limits{MaxWorkspaceBytes: workspace}}))
		if (limited.plan.calls[0].retained != nil) != (workspace == 16) {
			t.Fatal("optional preparation ignored the load-time workspace limit")
		}

		control.plan.limits = limited.plan.limits
		assertPreparedRun(t, limited, control, nil)
	}
}

func TestPreparedConstantReshapeLifetimeAndBits(t *testing.T) {
	wire := preparedConstantReshapeWire(t, "chain", "chain")
	bits := []uint32{0x7fc12345, 0xff812345, 0x80000000, 0x7f800000, 0xff800000, 1}
	for index := range wire.Graph.Initializer[0].FloatData {
		wire.Graph.Initializer[0].FloatData[index] = math.Float32frombits(bits[index%len(bits)])
	}
	for cycle := 0; cycle < 3; cycle++ {
		model := loadedModel(t, modelBytes(t, wire), Options{})
		sessions := []*Session{modelSession(t, model), modelSession(t, model)}
		position := len(model.plan.steps) - 1
		retained := model.plan.calls[position].retained
		if sessions[0].plan.calls[position].retained != sessions[1].plan.calls[position].retained {
			t.Fatal("prepared payload duplicated per session")
		}

		if err := model.Close(); err != nil {
			t.Fatal(err)
		}

		failures := make(chan error, len(sessions))
		for _, session := range sessions {
			go func(current *Session) {
				for repeat := 0; repeat < 10; repeat++ {
					outputs, runErr := current.Run(context.Background(), nil, RunOptions{})
					if runErr != nil {
						failures <- runErr
						return
					}

					values, valuesErr := outputs[0].Tensor.Float32Values()
					if valuesErr != nil {
						failures <- valuesErr
						return
					}

					for index, value := range values {
						if math.Float32bits(value) != bits[index%len(bits)] || outputs[0].Tensor.value == retained {
							failures <- fmt.Errorf("output bits or ownership changed")
							return
						}
					}
				}
				failures <- nil
			}(session)
		}
		for range sessions {
			if err := <-failures; err != nil {
				t.Fatal(err)
			}
		}
		for _, session := range sessions {
			if err := session.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestPreparedConstantReshapeReclaimed(t *testing.T) {
	for _, withSession := range []bool{false, true} {
		t.Run(fmt.Sprint(withSession), func(t *testing.T) {
			model := loadedModel(t, modelBytes(t, preparedConstantReshapeWire(t, "initializer", "initializer")), Options{})
			var session *Session
			if withSession {
				session = modelSession(t, model)
			}

			reclaimed := make(chan struct{}, 1)
			runtime.AddCleanup(model.plan.calls[0].retained, func(done chan struct{}) { done <- struct{}{} }, reclaimed)
			if err := model.Close(); err != nil {
				t.Fatal(err)
			}

			if withSession {
				runtime.GC()
				runtime.KeepAlive(session)
				select {
				case <-reclaimed:
					t.Fatal("payload reclaimed with surviving session")
				default:
				}
				if err := session.Close(); err != nil {
					t.Fatal(err)
				}
			}

			for attempt := 0; attempt < 30; attempt++ {
				runtime.GC()
				select {
				case <-reclaimed:
					return
				case <-time.After(10 * time.Millisecond):
				}
			}
			t.Fatal("prepared payload survived closure")
		})
	}
}

func TestPreparedConstantReshapeCumulativeBudgetAndAliases(t *testing.T) {
	wire := preparedConstantReshapeWire(t, "initializer", "initializer")
	second := proto.Clone(wire.Graph.Node[0]).(*onnxpb.NodeProto)
	second.Output = []string{"second"}
	wire.Graph.Node = append(wire.Graph.Node, second,
		&onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"y"}, Output: []string{"alias"}},
		&onnxpb.NodeProto{OpType: proto.String("Add"), Input: []string{"alias", "second"}, Output: []string{"sum"}})
	wire.Graph.Output[0].Name = proto.String("sum")
	data := modelBytes(t, wire)
	baseline := loadedModel(t, data, Options{})
	if baseline.plan.calls[0].retained == nil || baseline.plan.calls[1].retained == nil {
		t.Fatal("two immutable results were not prepared")
	}

	limited := loadedModel(t, data, Options{Limits: Limits{MaxLoadBytes: baseline.plan.loadBytes - 1}})
	count := 0
	for _, call := range limited.plan.calls {
		if call != nil && call.retained != nil {
			count++
		}
	}
	if count != 1 || limited.plan.loadBytes > baseline.plan.loadBytes-1 {
		t.Fatal("cumulative preparation budget was exceeded")
	}

	prepared := modelSession(t, baseline)
	control := preparedReshapeControl(t, data)
	assertPreparedRun(t, prepared, control, nil)
	owner := prepared.plan.lifetimes.owners[prepared.plan.indices["alias"]]
	if !prepared.plan.lifetimes.pinned[owner] {
		t.Fatal("Identity alias of prepared output is not pinned")
	}

	for _, call := range prepared.plan.calls[:2] {
		values, err := call.retained.Float32Values()
		if err != nil {
			t.Fatal(err)
		}

		for index, value := range values {
			if value != float32(index-12) {
				t.Fatal("downstream execution overwrote prepared storage")
			}
		}
	}
}

func TestPreparedConstantReshapeLoadCancellation(t *testing.T) {
	data := modelBytes(t, preparedConstantReshapeWire(t, "chain", "chain"))
	count := newCancelContext(0)
	defer count.cancel()
	model, err := LoadBytes(count, data, Options{})
	if err != nil {
		t.Fatal(err)
	}

	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	for boundary := 1; boundary <= count.checks; boundary++ {
		ctx := newCancelContext(boundary)
		model, err = LoadBytes(ctx, data, Options{})
		ctx.cancel()
		if model != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("load cancellation boundary %d: %v", boundary, err)
		}
	}
}

func TestPreparedConstantReshapeConservativeLiveFallback(t *testing.T) {
	for _, mode := range []string{"unknown value", "optional normalization buffers"} {
		t.Run(mode, func(t *testing.T) {
			wire := preparedConstantReshapeWire(t, "initializer", "initializer")
			inputs := make(map[string]*Tensor)
			if mode == "unknown value" {
				input := proto.Clone(modelWire(t, "m2/models/reshape_zero_infer_float32").Graph.Input[0]).(*onnxpb.ValueInfoProto)
				input.Name = proto.String("unused")
				input.Type.GetTensorType().Shape.Dim[0].Value = nil
				wire.Graph.Input = append(wire.Graph.Input, input)
				inputs["unused"] = floatTensor(t, []int64{1, 3, 4}, make([]float32, 12))
			} else {
				wire.OpsetImport[0].Version = proto.Int64(17)
				scale := make([]float32, 12)
				for index := range scale {
					scale[index] = 1
				}
				wire.Graph.Initializer = append(wire.Graph.Initializer, &onnxpb.TensorProto{Name: proto.String("scale"), DataType: proto.Int32(1), Dims: []int64{12}, FloatData: scale})
				wire.Graph.Node = append(wire.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("LayerNormalization"), Input: []string{"y", "scale"}, Output: []string{"normalized"}})
				wire.Graph.Output[0].Name = proto.String("normalized")
			}

			data := modelBytes(t, wire)
			prepared := modelSession(t, loadedModel(t, data, Options{}))
			control := preparedReshapeControl(t, data)
			if prepared.plan.calls[0].retained != nil {
				t.Fatal("unbounded live storage enabled optional preparation")
			}

			assertPreparedRun(t, prepared, control, inputs)
		})
	}
}

func TestPreparedConstantReshapeInvalidBindingBudget(t *testing.T) {
	wire := preparedConstantReshapeWire(t, "initializer", "initializer")
	input := proto.Clone(modelWire(t, "m2/models/reshape_zero_infer_float32").Graph.Input[1]).(*onnxpb.ValueInfoProto)
	input.Name = proto.String("unused")
	input.Type.GetTensorType().ElemType = proto.Int32(1)
	input.Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}
	wire.Graph.Input = append(wire.Graph.Input, input)
	data := modelBytes(t, wire)
	model := loadedModel(t, data, Options{Limits: Limits{MaxLiveBytes: 532, MaxWorkspaceBytes: 16, MaxTensorBytes: 1024}})
	prepared := modelSession(t, model)
	control := preparedReshapeControl(t, data)
	control.plan.limits = prepared.plan.limits
	if prepared.plan.calls[0].retained != nil {
		t.Fatal("optional resident payload can change a binding error")
	}

	inputs := map[string]*Tensor{"unused": floatTensor(t, []int64{100}, make([]float32, 100))}
	if _, err := prepared.Run(context.Background(), inputs, RunOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid signature lost priority to resident-memory failure: %v", err)
	}

	assertPreparedRun(t, prepared, control, inputs)
	assertPreparedRun(t, prepared, control, map[string]*Tensor{"unused": floatTensor(t, []int64{1}, []float32{1})})
}

func preparedReshapeControl(t *testing.T, data []byte) (session *Session) {
	session = modelSession(t, loadedModel(t, data, Options{}))
	for _, call := range session.plan.calls {
		if call != nil && call.retained != nil {
			session.plan.residentBytes -= call.retained.ByteSize()
		}
	}
	session.plan.calls = nil
	var err error
	if session.plan.lifetimes, err = newValueLifetimes(context.Background(), session.plan); err != nil {
		t.Fatal(err)
	}

	return session
}

func preparedConstantReshapeWire(t *testing.T, source, target string) (wire *onnxpb.ModelProto) {
	wire = modelWire(t, "m2/models/reshape_zero_infer_float32")
	wire.Graph.Output[0].Type.GetTensorType().Shape = nil
	values := make([]float32, 24)
	for index := range values {
		values[index] = float32(index - 12)
	}
	constants := []*onnxpb.TensorProto{
		{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{2, 3, 4}, FloatData: values},
		{Name: proto.String("shape"), DataType: proto.Int32(7), Dims: []int64{2}, Int64Data: []int64{0, -1}},
	}
	for index, mode := range []string{source, target} {
		if mode == "input" {
			continue
		}

		value := constants[index]
		name := value.GetName()
		if mode == "default_identity" || mode == "default_chain" {
			wire.Graph.Initializer = append(wire.Graph.Initializer, value)
			nodes := []*onnxpb.NodeProto{{OpType: proto.String("Identity"), Input: []string{name}, Output: []string{name + "_alias"}}}
			if mode == "default_chain" {
				nodes = append(nodes, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: nodes[0].Output, Output: []string{name + "_alias2"}})
			}

			wire.Graph.Node[len(wire.Graph.Node)-1].Input[index] = nodes[len(nodes)-1].Output[0]
			wire.Graph.Node = append(nodes, wire.Graph.Node...)
			continue
		}

		if mode != "default" {
			inputs := wire.Graph.Input[:0]
			for _, input := range wire.Graph.Input {
				if input.GetName() != name {
					inputs = append(inputs, input)
				}
			}
			wire.Graph.Input = inputs
		}
		if mode == "constant" {
			wire.Graph.Node = append([]*onnxpb.NodeProto{{OpType: proto.String("Constant"), Output: []string{name}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: value}}}}, wire.Graph.Node...)
			continue
		}

		if mode == "identity" || mode == "chain" {
			value.Name = proto.String(name + "_constant")
			nodes := []*onnxpb.NodeProto{{OpType: proto.String("Identity"), Input: []string{value.GetName()}, Output: []string{name}}}
			if mode == "chain" {
				nodes[0].Output = []string{name + "_alias"}
				nodes = append(nodes, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: nodes[0].Output, Output: []string{name}})
			}

			wire.Graph.Node = append(nodes, wire.Graph.Node...)
		}
		wire.Graph.Initializer = append(wire.Graph.Initializer, value)
	}
	return wire
}
