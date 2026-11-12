package onnx

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestEvaluationRetainedConstantsNeverBecomeDestinations(t *testing.T) {
	tests := []struct {
		name     string
		dataType int32
		count    int64
	}{
		{"float bits", 1, 2},
		{"integers", 7, 2},
		{"empty floats", 1, 0},
		{"empty integers", 7, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "profile/identity")
			initializer := &onnxpb.TensorProto{Name: proto.String("x"), DataType: proto.Int32(tt.dataType), Dims: []int64{tt.count}}
			if tt.dataType == 1 && tt.count != 0 {
				initializer.FloatData = []float32{math.Float32frombits(0x7fc12345), math.Float32frombits(0x80000000)}
			} else if tt.count != 0 {
				initializer.Int64Data = []int64{math.MinInt64, math.MaxInt64}
			}

			constant := proto.Clone(initializer).(*onnxpb.TensorProto)
			constant.Name = nil
			wire.Graph.Input, wire.Graph.ValueInfo = nil, nil
			wire.Graph.Initializer = []*onnxpb.TensorProto{initializer}
			wire.Graph.Node = []*onnxpb.NodeProto{
				{OpType: proto.String("Constant"), Output: []string{"constant"}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: constant}}},
				{OpType: proto.String("Identity"), Input: []string{"x"}, Output: []string{"alias"}},
				{OpType: proto.String("Transpose"), Input: []string{"alias"}, Output: []string{"left"}},
				{OpType: proto.String("Identity"), Input: []string{"constant"}, Output: []string{"constant_alias"}},
				{OpType: proto.String("Transpose"), Input: []string{"constant_alias"}, Output: []string{"right"}},
				{OpType: proto.String("Concat"), Input: []string{"left", "right"}, Output: []string{"y"}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(0)}}},
			}
			wire.Graph.Output = []*onnxpb.ValueInfoProto{{Name: proto.String("y")}}
			for _, output := range wire.Graph.Output {
				output.Type = &onnxpb.TypeProto{Value: &onnxpb.TypeProto_TensorType{TensorType: &onnxpb.TypeProto_Tensor{ElemType: proto.Int32(tt.dataType)}}}
			}
			model := loadedModel(t, modelBytes(t, wire), Options{})
			plan := model.plan
			sources := []*tensor.Tensor{plan.values[plan.indices["x"]].Default.Borrow(), plan.steps[0].Constant.Borrow()}
			for repeat := 0; repeat < 3; repeat++ {
				var (
					run     *evaluation
					outputs []NamedTensor
					err     error
				)

				if run, err = newEvaluation(context.Background(), plan); err != nil {
					t.Fatal(err)
				}

				if err = run.bind(nil); err != nil {
					t.Fatal(err)
				}

				if run.storage.values[plan.indices["x"]] != sources[0] {
					t.Fatal("initializer was not retained")
				}

				for position, step := range plan.steps {
					if err = run.execute(step); err != nil {
						t.Fatal(err)
					}

					if position == 0 && run.storage.values[step.Outputs[0]] != sources[1] {
						t.Fatal("Constant was not retained")
					}

					run.storage.after(position)
					if slices.Contains(sources, run.storage.free) {
						t.Fatal("retained source became writable scratch")
					}
				}
				run.storage.trim()
				// Inspect publication independently of output pinning: neither
				// constant alias is a declared output of this plan.
				if outputs, err = run.publish([]int{plan.indices["y"], plan.indices["alias"], plan.indices["constant_alias"]}); err != nil {
					t.Fatal(err)
				}

				for slot, source := range sources {
					output := outputs[slot+1].Tensor
					if output.value == source || !slices.Equal(output.Shape(), []int64{tt.count}) {
						t.Fatal("publication borrowed model storage")
					}

					if tt.dataType == 1 {
						assertFloats(t, &Tensor{value: source}, []int64{tt.count}, initializer.FloatData)
						assertFloats(t, output, []int64{tt.count}, initializer.FloatData)
					} else {
						var values []int64
						if values, err = source.Int64Values(); err != nil || !slices.Equal(values, initializer.Int64Data) {
							t.Fatalf("retained integers changed: %v, %v", values, err)
						}
					}

					// Even an internal write to a published alias must not reach the model.
					var zeros *tensor.Tensor
					if zeros, err = tensor.New(source.DataType(), []int64{tt.count}); err != nil {
						t.Fatal(err)
					}

					if err = tensor.ReshapeInto(zeros, output.value); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
}
