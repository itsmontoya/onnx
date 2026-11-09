package operators

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/capability"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
	"google.golang.org/protobuf/proto"
)

func planRegistry(t *testing.T) (out *Registry) {
	t.Helper()
	data, err := os.ReadFile("../../capabilities.json")
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Operators []capability.Record `json:"operators"`
	}

	if err = json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}

	registry, err := NewRegistry(manifest.Operators)
	if err != nil {
		t.Fatal(err)
	}

	return registry
}

func planFixture() (out *onnxpb.ModelProto) {
	var model onnxpb.ModelProto
	model.IrVersion = proto.Int64(8)
	model.OpsetImport = []*onnxpb.OperatorSetIdProto{{Version: proto.Int64(13)}}
	model.Graph = &onnxpb.GraphProto{
		Name:   proto.String("plan"),
		Input:  []*onnxpb.ValueInfoProto{planInfo("x", 1, 2)},
		Output: []*onnxpb.ValueInfoProto{planInfo("y", 1, 2)},
		Node:   []*onnxpb.NodeProto{{Name: proto.String("copy"), OpType: proto.String("Identity"), Input: []string{"x"}, Output: []string{"y"}}},
	}
	return &model
}

func planInfo(name string, dataType int32, sizes ...int64) (out *onnxpb.ValueInfoProto) {
	var info onnxpb.ValueInfoProto
	info.Name = proto.String(name)
	info.Type = &onnxpb.TypeProto{Value: &onnxpb.TypeProto_TensorType{TensorType: &onnxpb.TypeProto_Tensor{
		ElemType: proto.Int32(dataType), Shape: &onnxpb.TensorShapeProto{},
	}}}
	for _, size := range sizes {
		info.Type.GetTensorType().Shape.Dim = append(info.Type.GetTensorType().Shape.Dim, &onnxpb.TensorShapeProto_Dimension{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: size}})
	}
	return &info
}

func builtGraph(t *testing.T, model *onnxpb.ModelProto) (out *graph.Graph) {
	t.Helper()
	wire, err := proto.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}

	source, err := graph.Build(context.Background(), model, int64(len(wire)), graph.Limits{})
	if err != nil {
		t.Fatal(err)
	}

	return source
}

func TestPlanInference(t *testing.T) {
	registry := planRegistry(t)
	tests := []struct {
		name      string
		configure func(*onnxpb.ModelProto)
		shape     []graph.Dimension
		deferred  bool
		dataType  tensor.DataType
	}{
		{"fixed", func(model *onnxpb.ModelProto) {}, []graph.Dimension{{Kind: graph.FixedDimension, Value: 2}}, false, tensor.Float32},
		{"scalar", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0] = planInfo("x", 7)
			model.Graph.Output[0] = planInfo("y", 7)
		}, nil, false, tensor.Int64},
		{"empty", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0] = planInfo("x", 1, 0, 3)
			model.Graph.Output[0] = planInfo("y", 1, 0, 3)
		}, []graph.Dimension{{Kind: graph.FixedDimension, Value: 0}, {Kind: graph.FixedDimension, Value: 3}}, false, tensor.Float32},
		{"unknown leading batch", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{}
			model.Graph.Output[0].Type.GetTensorType().Shape = nil
		}, []graph.Dimension{{Kind: graph.UnknownDimension}}, true, tensor.Float32},
		{"symbolic batch", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
			model.Graph.Output[0].Type.GetTensorType().Shape = nil
		}, []graph.Dimension{{Kind: graph.SymbolicDimension, Symbol: "N"}}, true, tensor.Float32},
		{"output constraint on dynamic size", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{}
		}, []graph.Dimension{{Kind: graph.FixedDimension, Value: 2}}, false, tensor.Float32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := planFixture()
			tt.configure(model)
			// The extra Identity has no intermediate value_info at all.
			model.Graph.Node[0].Output[0] = "middle"
			model.Graph.Node = append(model.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"middle"}, Output: []string{"y"}})
			plan, err := Compile(context.Background(), builtGraph(t, model), registry, Limits{})
			if err != nil {
				t.Fatal(err)
			}

			value, exists := plan.Value("y")
			if !exists || !value.Info.ShapeKnown || value.Deferred != tt.deferred || value.Info.DataType != tt.dataType || len(value.Info.Dimensions) != len(tt.shape) {
				t.Fatalf("inferred value: %+v", value)
			}

			for axis, dimension := range tt.shape {
				if value.Info.Dimensions[axis] != dimension {
					t.Fatalf("axis %d = %+v, want %+v", axis, value.Info.Dimensions[axis], dimension)
				}
			}

			if len(plan.Steps()) != 2 || plan.Steps()[1].SchemaVersion != 13 || len(value.Declarations) != 1 {
				t.Fatalf("missing node/declaration: %+v", plan.Steps())
			}
		})
	}
}

func TestPlanConstantAndDefaults(t *testing.T) {
	registry := planRegistry(t)
	for _, tt := range []struct {
		name     string
		dataType int32
		shape    []int64
		floats   []float32
		integers []int64
	}{
		{"float scalar", 1, nil, []float32{3.5}, nil},
		{"exact integer", 7, []int64{1}, nil, []int64{16777217}},
		{"empty", 7, []int64{2, 0, 3}, nil, nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := planFixture()
			model.Graph.Input = nil
			model.Graph.Output[0] = planInfo("y", tt.dataType, tt.shape...)
			model.Graph.Node[0] = &onnxpb.NodeProto{OpType: proto.String("Constant"), Output: []string{"y"}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(tt.dataType), Dims: tt.shape, FloatData: tt.floats, Int64Data: tt.integers}}}}
			model.Graph.Node[0].Output[0] = "constant"
			model.Graph.Node = append(model.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"constant"}, Output: []string{"y"}})
			source := builtGraph(t, model)
			constant := source.Nodes()[0].Attributes[0].Tensor
			plan, err := Compile(context.Background(), source, registry, Limits{})
			if err != nil {
				t.Fatal(err)
			}

			_, before := source.Usage()
			_, after := plan.Usage()
			if before != after || plan.Steps()[0].Constant != constant {
				t.Fatal("compilation copied or lost immutable constant storage")
			}

			value, exists := plan.Value("y")
			if !exists || value.Deferred || len(value.Info.Dimensions) != len(tt.shape) {
				t.Fatalf("constant inference: %+v", value)
			}

			if tt.dataType == 7 {
				values, valuesErr := plan.Steps()[0].Constant.Int64Values()
				if valuesErr != nil || !reflect.DeepEqual(values, tt.integers) && len(tt.integers) != 0 {
					t.Fatalf("constant values = %v, %v", values, valuesErr)
				}
			}
		})
	}

	model := planFixture()
	model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: "N"}
	model.Graph.Output[0].Type.GetTensorType().Shape = nil
	model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{2}, FloatData: []float32{1, 2}}}
	plan, err := Compile(context.Background(), builtGraph(t, model), registry, Limits{})
	if err != nil {
		t.Fatal(err)
	}

	value, _ := plan.Value("y")
	if !value.Deferred || value.Info.Dimensions[0].Kind != graph.SymbolicDimension || !plan.Inputs()[0].Defaulted {
		t.Fatalf("default froze the input contract: %+v", value)
	}
}

func TestPlanRejectsStaticCombinations(t *testing.T) {
	registry := planRegistry(t)
	tests := []struct {
		name      string
		configure func(*onnxpb.ModelProto)
		kind      error
		text      string
		node      int
	}{
		{"extra attribute", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum()}}
		}, graph.ErrUnsupported, `attribute "axis"`, 0},
		{"extra input", func(model *onnxpb.ModelProto) { model.Graph.Node[0].Input = []string{"x", "x"} }, graph.ErrUnsupported, "input slot count", 0},
		{"missing input", func(model *onnxpb.ModelProto) { model.Graph.Node[0].Input = nil }, graph.ErrInvalidModel, "input slot count", 0},
		{"empty required input", func(model *onnxpb.ModelProto) { model.Graph.Node[0].Input[0] = "" }, graph.ErrInvalidModel, "input[0] is empty", 0},
		{"empty required output", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].Output[0] = ""
			model.Graph.Output = nil
		}, graph.ErrInvalidModel, "output[0] is empty", 0},
		{"extra nonempty output", func(model *onnxpb.ModelProto) { model.Graph.Node[0].Output = []string{"y", "extra"} }, graph.ErrUnsupported, "output slot count", 0},
		{"extra empty output", func(model *onnxpb.ModelProto) { model.Graph.Node[0].Output = []string{"y", ""} }, graph.ErrUnsupported, "output slot count", 0},
		{"contradictory rank", func(model *onnxpb.ModelProto) { model.Graph.Output[0] = planInfo("y", 1) }, graph.ErrInvalidModel, "inferred rank", 0},
		{"contradictory dtype", func(model *onnxpb.ModelProto) { model.Graph.Output[0] = planInfo("y", 7, 2) }, graph.ErrInvalidModel, "inferred type", 0},
		{"contradictory size", func(model *onnxpb.ModelProto) { model.Graph.Output[0] = planInfo("y", 1, 3) }, graph.ErrInvalidModel, "inferred axis", 0},
		{"dead planned node", func(model *onnxpb.ModelProto) {
			model.Graph.Node = append(model.Graph.Node, &onnxpb.NodeProto{Name: proto.String("dead"), OpType: proto.String("Flatten"), Input: []string{"x"}, Output: []string{"unused"}})
		}, graph.ErrUnsupported, "is planned for RT-12", 1},
		{"unknown node", func(model *onnxpb.ModelProto) { model.Graph.Node[0].OpType = proto.String("Unimplemented") }, graph.ErrUnsupported, "no schema", 0},
		{"missing constant value", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].OpType = proto.String("Constant")
			model.Graph.Node[0].Input = nil
		}, graph.ErrInvalidModel, "exactly one supported", 0},
		{"wrong constant attribute type", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].OpType = proto.String("Constant")
			model.Graph.Node[0].Input = nil
			model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(2)}}
		}, graph.ErrInvalidModel, "has type INT, require TENSOR", 0},
		{"scalar constant declared vector", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].OpType = proto.String("Constant")
			model.Graph.Node[0].Input = nil
			model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("value_float"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(1)}}
		}, graph.ErrInvalidModel, "inferred rank", 0},
		{"string constant form", func(model *onnxpb.ModelProto) {
			model.Graph.Node[0].OpType = proto.String("Constant")
			model.Graph.Node[0].Input = nil
			model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("value_string"), Type: onnxpb.AttributeProto_STRING.Enum(), S: []byte("x")}}
		}, graph.ErrUnsupported, `attribute "value_string"`, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := planFixture()
			tt.configure(model)
			source := builtGraph(t, model)
			var first string
			for repeat := 0; repeat < 3; repeat++ {
				plan, err := Compile(context.Background(), source, registry, Limits{})
				var diagnostic *runtimeerror.Error
				if plan != nil || !errors.Is(err, tt.kind) || !errors.As(err, &diagnostic) || !strings.Contains(err.Error(), tt.text) {
					t.Fatalf("Compile = %v, %v", plan, err)
				}

				if diagnostic.NodeIndex != tt.node || diagnostic.Domain != "" || diagnostic.Opset != 13 || diagnostic.Cause == nil {
					t.Fatalf("diagnostic = %+v", diagnostic)
				}

				if repeat > 0 && first != err.Error() {
					t.Fatalf("nondeterministic error: %q != %q", first, err.Error())
				}

				first = err.Error()
			}
		})
	}
}

func TestPlanPassthroughAndOwnership(t *testing.T) {
	registry := planRegistry(t)
	model := planFixture()
	model.Graph.Node = nil
	model.Graph.Output[0] = planInfo("x", 1, 2)
	model.MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("author"), Value: proto.String("original")}}
	source := builtGraph(t, model)
	plan, err := Compile(context.Background(), source, registry, Limits{})
	if err != nil {
		t.Fatal(err)
	}

	if len(plan.Steps()) != 0 || plan.Outputs()[0].Name != "x" {
		t.Fatal("passthrough did not compile")
	}

	model.Graph.Input[0].Name = proto.String("mutated")
	plan.Metadata().Properties[0].Value = "changed"
	plan.Inputs()[0].Dimensions[0].Value = 9
	plan.Outputs()[0].Dimensions[0].Value = 9
	plan.Values()[0].Info.Dimensions[0].Value = 9
	plan.Values()[0].Declarations[0].Dimensions[0].Value = 9
	value, _ := plan.Value("x")
	value.Info.Dimensions[0].Value = 9
	if plan.Metadata().Properties[0].Value != "original" || plan.Inputs()[0].Name != "x" || plan.Inputs()[0].Dimensions[0].Value != 2 || plan.Values()[0].Info.Dimensions[0].Value != 2 || plan.Values()[0].Declarations[0].Dimensions[0].Value != 2 {
		t.Fatal("plan snapshots share mutable slices")
	}

	if _, exists := plan.Value("missing"); exists {
		t.Fatal("unexpected value")
	}

	var wait sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for repeat := 0; repeat < 10; repeat++ {
				plan.Metadata()
				plan.Inputs()
				plan.Outputs()
				plan.Values()
				plan.Steps()
				plan.Value("x")
				plan.Usage()
			}
		}()
	}
	wait.Wait()

	model = planFixture()
	model.Graph.Node[0].DocString = proto.String("node docs")
	model.Graph.Node[0].MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("node"), Value: proto.String("original")}}
	plan, err = Compile(context.Background(), builtGraph(t, model), registry, Limits{})
	if err != nil {
		t.Fatal(err)
	}

	plan.Steps()[0].Inputs[0] = 99
	plan.Steps()[0].Outputs[0] = 99
	plan.Steps()[0].Metadata[0].Value = "mutated"
	if plan.Steps()[0].Inputs[0] != 0 || plan.Steps()[0].Outputs[0] != 1 || plan.Steps()[0].DocString != "node docs" || plan.Steps()[0].Metadata[0].Value != "original" {
		t.Fatal("step snapshots share slots")
	}
}

func TestPlanBudgetsAndNoNumericOutputs(t *testing.T) {
	registry := planRegistry(t)
	source := builtGraph(t, planFixture())
	plan, err := Compile(context.Background(), source, registry, Limits{})
	if err != nil {
		t.Fatal(err)
	}

	bytes, _ := plan.Usage()
	if _, err = Compile(context.Background(), source, registry, Limits{MaxLoadBytes: bytes}); err != nil {
		t.Fatalf("exact budget: %v", err)
	}

	if _, err = Compile(context.Background(), source, registry, Limits{MaxLoadBytes: bytes - 1}); !errors.Is(err, graph.ErrLimit) {
		t.Fatalf("budget boundary: %v", err)
	}

	graphBytes, _ := source.Usage()
	if _, err = Compile(context.Background(), source, registry, Limits{MaxLoadBytes: graphBytes}); !errors.Is(err, graph.ErrLimit) {
		t.Fatalf("snapshot reservation: %v", err)
	}

	for _, limits := range []Limits{{MaxTensorBytes: 7}, {MaxRank: 1, MaxDimension: 1}} {
		if _, err = Compile(context.Background(), source, registry, limits); !errors.Is(err, graph.ErrLimit) {
			t.Fatalf("static shape limit %+v: %v", limits, err)
		}
	}

	model := planFixture()
	model.Graph.Input[0] = planInfo("x", 1, 4<<20)
	model.Graph.Output[0] = planInfo("y", 1, 4<<20)
	source = builtGraph(t, model)
	result := testing.Benchmark(func(b *testing.B) {
		for iteration := 0; iteration < b.N; iteration++ {
			if _, err = Compile(context.Background(), source, registry, Limits{}); err != nil {
				b.Fatal(err)
			}
		}
	})
	if result.AllocedBytesPerOp() >= 1<<20 {
		t.Fatalf("metadata inference allocated numeric-sized storage: %d bytes", result.AllocedBytesPerOp())
	}
}

func TestPlanInvalidArgumentsAndCancellation(t *testing.T) {
	registry := planRegistry(t)
	source := builtGraph(t, planFixture())
	for _, tt := range []struct {
		ctx      context.Context
		source   *graph.Graph
		registry *Registry
		limits   Limits
	}{
		{nil, source, registry, Limits{}},
		{context.Background(), nil, registry, Limits{}},
		{context.Background(), source, nil, Limits{}},
		{context.Background(), new(graph.Graph), registry, Limits{}},
		{context.Background(), source, registry, Limits{MaxLoadBytes: -1}},
	} {
		if _, err := Compile(tt.ctx, tt.source, tt.registry, tt.limits); !errors.Is(err, runtimeerror.ErrInvalidArgument) {
			t.Fatalf("invalid Compile arguments: %v", err)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := Compile(ctx, source, registry, Limits{})
	var diagnostic *runtimeerror.Error
	if !errors.Is(err, context.Canceled) || !errors.As(err, &diagnostic) || diagnostic.NodeIndex != -1 {
		t.Fatalf("cancellation cause: %v", err)
	}
}

func TestPlanDiagnosticShapes(t *testing.T) {
	registry := planRegistry(t)
	for _, tt := range []struct {
		name    string
		sizes   []int64
		unknown bool
	}{
		{"scalar", nil, false},
		{"empty", []int64{0, 3}, false},
		{"fixed", []int64{2}, false},
		{"dynamic", []int64{2}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := planFixture()
			model.Graph.Input[0] = planInfo("x", 1, tt.sizes...)
			if tt.unknown {
				model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{}
			}

			model.Graph.Output[0].Type.GetTensorType().Shape = nil
			model.Graph.Node[0].OpType = proto.String("Unknown")
			_, err := Compile(context.Background(), builtGraph(t, model), registry, Limits{})
			var diagnostic *runtimeerror.Error
			if !errors.As(err, &diagnostic) || len(diagnostic.Shapes) != 1 {
				t.Fatalf("diagnostic: %v", err)
			}

			if tt.unknown {
				if diagnostic.Shapes[0] != nil {
					t.Fatalf("unknown shape became concrete: %v", diagnostic.Shapes)
				}
			} else if diagnostic.Shapes[0] == nil || len(diagnostic.Shapes[0]) != len(tt.sizes) {
				t.Fatalf("concrete shape lost (including scalar): %v", diagnostic.Shapes)
			}
		})
	}
}

func TestPlanCapabilityTypeConstraints(t *testing.T) {
	registry := planRegistry(t)
	// A narrowed test subset verifies that the compiler applies manifest dtype
	// restrictions in addition to the structural float32/int64 envelope.
	records := make([]capability.Record, len(registry.records))
	for index, record := range registry.records {
		records[index] = record.Clone()
		records[index].DataTypes = []string{"float32"}
	}
	var err error
	if registry, err = NewRegistry(records); err != nil {
		t.Fatal(err)
	}

	model := planFixture()
	model.Graph.Input[0] = planInfo("x", 7, 2)
	model.Graph.Output[0] = planInfo("y", 7, 2)
	if _, err = Compile(context.Background(), builtGraph(t, model), registry, Limits{}); !errors.Is(err, graph.ErrUnsupported) || !strings.Contains(err.Error(), "input value") {
		t.Fatalf("Identity dtype constraint: %v", err)
	}

	model.Graph.Input = nil
	model.Graph.Node[0].OpType = proto.String("Constant")
	model.Graph.Node[0].Input = nil
	model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("value"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(7), Dims: []int64{2}, Int64Data: []int64{1, 2}}}}
	if _, err = Compile(context.Background(), builtGraph(t, model), registry, Limits{}); !errors.Is(err, graph.ErrUnsupported) || !strings.Contains(err.Error(), "output type") {
		t.Fatalf("Constant dtype constraint: %v", err)
	}
}

func TestPlanStaticShapeLimits(t *testing.T) {
	registry := planRegistry(t)
	for _, tt := range []struct {
		name   string
		sizes  []int64
		limits Limits
		text   string
	}{
		{"rank", []int64{0, 2}, Limits{MaxRank: 1}, "planned rank"},
		{"unused empty dimension", []int64{0, 3}, Limits{MaxDimension: 2}, "planned axis"},
		{"tensor bytes", []int64{3}, Limits{MaxTensorBytes: 8}, "planned tensor bytes"},
		{"product overflow", []int64{1<<31 - 1, 1<<31 - 1, 1<<31 - 1}, Limits{}, "planned shape"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := planFixture()
			model.Graph.Input[0] = planInfo("x", 1, tt.sizes...)
			model.Graph.Output[0] = planInfo("y", 1, tt.sizes...)
			_, err := Compile(context.Background(), builtGraph(t, model), registry, tt.limits)
			if !errors.Is(err, graph.ErrLimit) || !strings.Contains(err.Error(), tt.text) {
				t.Fatalf("static limit: %v", err)
			}
		})
	}
}

func TestPlanSnapshotBytesWithoutNumericalCopies(t *testing.T) {
	registry := planRegistry(t)
	var baseline int64
	for _, count := range []int{1, 4096} {
		model := planFixture()
		model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("unused"), DataType: proto.Int32(1), Dims: []int64{int64(count)}, FloatData: make([]float32, count)}}
		plan, err := Compile(context.Background(), builtGraph(t, model), registry, Limits{})
		if err != nil {
			t.Fatal(err)
		}

		bytes, err := plan.SnapshotBytes(context.Background())
		if err != nil || bytes <= 0 {
			t.Fatalf("snapshot estimate: %d, %v", bytes, err)
		}

		if baseline != 0 && bytes != baseline {
			t.Fatalf("snapshot estimate charges shared numerical storage: %d != %d", bytes, baseline)
		}

		baseline = bytes
		if allocations := testing.AllocsPerRun(10, func() { _, err = plan.SnapshotBytes(context.Background()) }); allocations != 0 || err != nil {
			t.Fatalf("snapshot estimator allocated: %g, %v", allocations, err)
		}

		if _, err = plan.SnapshotBytes(nil); !errors.Is(err, graph.ErrInvalidArgument) {
			t.Fatalf("nil snapshot context: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err = plan.SnapshotBytes(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled snapshot: %v", err)
		}
	}
}

func TestPlanNumericalMetadataOnly(t *testing.T) {
	registry := planRegistry(t)
	for _, operator := range []string{"Add", "MatMul", "Relu"} {
		t.Run(operator, func(t *testing.T) {
			model := planFixture()
			model.Graph.Input[0] = planInfo("x", 1, 4<<20, 1)
			model.Graph.Output[0] = planInfo("y", 1, 4<<20, 1)
			model.Graph.Node[0].OpType = proto.String(operator)
			if operator == "Add" {
				model.Graph.Node[0].Input = []string{"x", "x"}
			}

			if operator == "MatMul" {
				model.Graph.Input = append(model.Graph.Input, planInfo("weight", 1, 1, 1))
				model.Graph.Node[0].Input = []string{"x", "weight"}
			}

			source := builtGraph(t, model)
			result := testing.Benchmark(func(b *testing.B) {
				for iteration := 0; iteration < b.N; iteration++ {
					if _, err := Compile(context.Background(), source, registry, Limits{}); err != nil {
						b.Fatal(err)
					}
				}
			})
			if result.AllocedBytesPerOp() >= 1<<20 {
				t.Fatalf("metadata inference allocated numeric storage: %d bytes", result.AllocedBytesPerOp())
			}
		})
	}
}
