package onnx

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"testing"

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
