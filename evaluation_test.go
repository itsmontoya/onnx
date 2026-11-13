package onnx

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"github.com/itsmontoya/onnx/internal/operators"
	"google.golang.org/protobuf/proto"
)

func TestEvaluationCheckValueActualMetadataAndDeclarations(t *testing.T) {
	var (
		fixed = graph.Dimension{Kind: graph.FixedDimension, Value: 2}
		first = graph.Dimension{Kind: graph.SymbolicDimension, Symbol: "first"}
		last  = graph.Dimension{Kind: graph.SymbolicDimension, Symbol: "last"}
	)

	tests := []struct {
		name         string
		actualType   DataType
		shape        []int64
		info         graph.ValueInfo
		declarations []graph.ValueInfo
		wantError    string
		wantSymbols  map[string]int64
	}{
		{
			name: "original symbols and inferred fixed shape", actualType: Float32, shape: []int64{2},
			info: graph.ValueInfo{DataType: Float32, ShapeKnown: true, Dimensions: []graph.Dimension{fixed}},
			declarations: []graph.ValueInfo{
				{DataType: Float32, ShapeKnown: true, Dimensions: []graph.Dimension{first}},
				{DataType: Float32, ShapeKnown: true, Dimensions: []graph.Dimension{last}},
			},
			wantSymbols: map[string]int64{"first": 2, "last": 2},
		},
		{
			name: "empty symbolic tensor", actualType: Int64, shape: []int64{0},
			info:         graph.ValueInfo{DataType: Int64},
			declarations: []graph.ValueInfo{{ShapeKnown: true, Dimensions: []graph.Dimension{first}}},
			wantSymbols:  map[string]int64{"first": 0},
		},
		{
			name: "scalar with unknown dtype declarations", actualType: Float32,
			info:         graph.ValueInfo{ShapeKnown: true},
			declarations: []graph.ValueInfo{{}, {ShapeKnown: true}},
		},
		{
			name: "actual dtype violates original declaration", actualType: Int64, shape: []int64{2},
			info:         graph.ValueInfo{DataType: Int64},
			declarations: []graph.ValueInfo{{DataType: Float32}}, wantError: "dtype",
		},
		{
			name: "actual dtype violates inferred info", actualType: Int64, shape: []int64{2},
			info:         graph.ValueInfo{DataType: Float32},
			declarations: []graph.ValueInfo{{DataType: Int64}, {DataType: Int64}}, wantError: "dtype",
		},
		{
			name: "original rank error precedes inferred dtype error", actualType: Float32, shape: []int64{2},
			info:         graph.ValueInfo{DataType: Int64},
			declarations: []graph.ValueInfo{{DataType: Float32, ShapeKnown: true}}, wantError: "rank",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				backend *tensor.Tensor
				err     error
				run     = evaluation{
					ctx:     context.Background(),
					plan:    &executionPlan{limits: Limits{MaxRank: 4, MaxDimension: 8}},
					symbols: make(map[string]int64),
				}
			)

			if backend, err = tensor.New(tt.actualType, tt.shape); err != nil {
				t.Fatal(err)
			}

			err = run.checkValue(operators.Value{Info: tt.info, Declarations: tt.declarations}, backend, ErrInvalidArgument)
			if tt.wantError != "" {
				if !errors.Is(err, ErrInvalidArgument) || !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("validation error: %v, want %s", err, tt.wantError)
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if len(run.symbols) != len(tt.wantSymbols) {
				t.Fatalf("symbol bindings: %v, want %v", run.symbols, tt.wantSymbols)
			}

			for symbol, size := range tt.wantSymbols {
				if actual, exists := run.symbols[symbol]; !exists || actual != size {
					t.Fatalf("symbol %s: %d, %t, want %d", symbol, actual, exists, size)
				}
			}
		})
	}
}

func TestEvaluationArgumentsOptionalAndWideCalls(t *testing.T) {
	tests := []struct {
		name     string
		operator string
		inputs   []string
		shape    []int64
		want     []float32
	}{
		{"omitted bias after populated bias", "Gemm", []string{"a", "w", ""}, []int64{1, 2}, []float32{4, 7}},
		{"five argument fallback", "Concat", []string{"a", "a", "a", "a", "a"}, []int64{1, 10}, []float32{4, 7, 4, 7, 4, 7, 4, 7, 4, 7}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			wire := modelWire(t, "profile/identity")
			wire.Graph.Input, wire.Graph.ValueInfo = nil, nil
			wire.Graph.Output[0].Type.GetTensorType().Shape = nil
			wire.Graph.Initializer = []*onnxpb.TensorProto{
				{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{1, 2}, FloatData: []float32{1, 2}},
				{Name: proto.String("w"), DataType: proto.Int32(1), Dims: []int64{2, 2}, FloatData: []float32{1, 0, 0, 1}},
				{Name: proto.String("b"), DataType: proto.Int32(1), Dims: []int64{2}, FloatData: []float32{3, 5}},
			}
			wire.Graph.Node = []*onnxpb.NodeProto{
				{OpType: proto.String("Gemm"), Input: []string{"x", "w", "b"}, Output: []string{"a"}},
				{OpType: proto.String(tt.operator), Input: tt.inputs, Output: []string{"y"}},
			}
			if tt.operator == "Concat" {
				wire.Graph.Node[1].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)}}
			}

			session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
			for range 3 {
				outputs, err := session.Run(context.Background(), nil, RunOptions{})
				if err != nil {
					t.Fatal(err)
				}

				assertFloats(t, outputs[0].Tensor, tt.shape, tt.want)
			}
		})
	}
}

func TestEvaluationNormalizationOptionalArguments(t *testing.T) {
	wire := modelWire(t, "normalization/models/optional_inverse_no_bias")
	inputs := map[string]*Tensor{
		"x":     floatTensor(t, []int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		"scale": floatTensor(t, []int64{3}, []float32{1, 1, 1}),
	}
	control := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	expected, err := control.Run(context.Background(), inputs, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	first := proto.Clone(wire.Graph.Node[0]).(*onnxpb.NodeProto)
	first.Name = proto.String("with_bias")
	first.Input = append(first.Input, "bias")
	first.Output = []string{"biased"}
	wire.Graph.Node[0].Input = append(wire.Graph.Node[0].Input, "")
	wire.Graph.Node = append([]*onnxpb.NodeProto{first}, wire.Graph.Node...)
	wire.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("bias"), DataType: proto.Int32(1), Dims: []int64{3}, FloatData: []float32{3, 5, 7}}}
	session := modelSession(t, loadedModel(t, modelBytes(t, wire), Options{}))
	if len(session.plan.steps) != 2 {
		t.Fatal("missing preceding biased call")
	}

	for range 3 {
		outputs, runErr := session.Run(context.Background(), inputs, RunOptions{})
		if runErr != nil || len(outputs) != len(expected) {
			t.Fatalf("normalization outputs: %v, %v", outputs, runErr)
		}

		for index, output := range expected {
			values, valuesErr := output.Tensor.Float32Values()
			if valuesErr != nil {
				t.Fatal(valuesErr)
			}

			assertFloats(t, outputs[index].Tensor, output.Tensor.Shape(), values)
		}
	}
}

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
