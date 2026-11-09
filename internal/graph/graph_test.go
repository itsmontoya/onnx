package graph

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
)

func graphFixture() (out *onnxpb.ModelProto) {
	var model onnxpb.ModelProto
	model.IrVersion = proto.Int64(8)
	model.OpsetImport = []*onnxpb.OperatorSetIdProto{{Version: proto.Int64(13)}}
	model.Graph = &onnxpb.GraphProto{
		Name:   proto.String("test"),
		Input:  []*onnxpb.ValueInfoProto{tensorInfo("x", 1, symbolicDimension("N"), fixedDimension(2))},
		Output: []*onnxpb.ValueInfoProto{tensorInfo("y", 1, symbolicDimension("N"), fixedDimension(2))},
		Node:   []*onnxpb.NodeProto{{Name: proto.String("label"), OpType: proto.String("Identity"), Input: []string{"x"}, Output: []string{"y"}}},
	}
	return &model
}

func tensorInfo(name string, elementType int32, dimensions ...*onnxpb.TensorShapeProto_Dimension) (out *onnxpb.ValueInfoProto) {
	var info onnxpb.ValueInfoProto
	info.Name = proto.String(name)
	info.Type = &onnxpb.TypeProto{Value: &onnxpb.TypeProto_TensorType{TensorType: &onnxpb.TypeProto_Tensor{
		ElemType: proto.Int32(elementType), Shape: &onnxpb.TensorShapeProto{Dim: dimensions},
	}}}
	return &info
}

func fixedDimension(size int64) (out *onnxpb.TensorShapeProto_Dimension) {
	var dimension onnxpb.TensorShapeProto_Dimension
	dimension.Value = &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: size}
	return &dimension
}

func symbolicDimension(symbol string) (out *onnxpb.TensorShapeProto_Dimension) {
	var dimension onnxpb.TensorShapeProto_Dimension
	dimension.Value = &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: symbol}
	return &dimension
}

func encodedLength(t *testing.T, model *onnxpb.ModelProto) (bytes int64) {
	t.Helper()
	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}

	return int64(len(data))
}

func TestGraphLayouts(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*onnxpb.ModelProto)
		values    []string
		outputs   []string
		defaulted bool
	}{
		{"identity without intermediate metadata", func(model *onnxpb.ModelProto) {}, []string{"x", "y"}, []string{"y"}, false},
		{"empty graph", func(model *onnxpb.ModelProto) {
			model.Graph.Input = nil
			model.Graph.Output = nil
			model.Graph.Node = nil
		}, []string{}, []string{}, false},
		{"no declared outputs", func(model *onnxpb.ModelProto) {
			model.Graph.Output = nil
		}, []string{"x", "y"}, []string{}, false},
		{"input passthrough", func(model *onnxpb.ModelProto) {
			model.Graph.Node = nil
			model.Graph.Output = []*onnxpb.ValueInfoProto{tensorInfo("x", 1, symbolicDimension("N"), fixedDimension(2))}
		}, []string{"x"}, []string{"x"}, false},
		{"initializer passthrough", func(model *onnxpb.ModelProto) {
			model.Graph.Node = nil
			model.Graph.Input = nil
			model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(7), Int64Data: []int64{16777217}}}
			model.Graph.Output = []*onnxpb.ValueInfoProto{tensorInfo("w", 7)}
		}, []string{"w"}, []string{"w"}, false},
		{"default with symbolic batch", func(model *onnxpb.ModelProto) {
			model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{1, 2}, FloatData: []float32{3, 4}}}
		}, []string{"x", "y"}, []string{"y"}, true},
		{"default with unknown leading batch", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = &onnxpb.TensorShapeProto_Dimension{}
			model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{0, 2}}}
		}, []string{"x", "y"}, []string{"y"}, true},
		{"multiple IO and repeated labels", func(model *onnxpb.ModelProto) {
			model.Graph.Input = append(model.Graph.Input, tensorInfo("b", 1, symbolicDimension("N"), fixedDimension(2)))
			model.Graph.Node[0].Input = []string{"x", "b"}
			model.Graph.Node[0].OpType = proto.String("Add")
			model.Graph.Node = append(model.Graph.Node, &onnxpb.NodeProto{Name: proto.String("label"), OpType: proto.String("Identity"), Input: []string{"b"}, Output: []string{"z"}})
			model.Graph.Output = append(model.Graph.Output, tensorInfo("z", 1, symbolicDimension("N"), fixedDimension(2)))
		}, []string{"x", "b", "y", "z"}, []string{"y", "z"}, false},
		{"scalar", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0] = tensorInfo("x", 1)
			model.Graph.Output[0] = tensorInfo("y", 1)
		}, []string{"x", "y"}, []string{"y"}, false},
		{"empty tensor", func(model *onnxpb.ModelProto) {
			model.Graph.Input[0] = tensorInfo("x", 7, fixedDimension(0), fixedDimension(2))
			model.Graph.Output[0] = tensorInfo("y", 7, fixedDimension(0), fixedDimension(2))
		}, []string{"x", "y"}, []string{"y"}, false},
		{"unknown output rank and dimensions", func(model *onnxpb.ModelProto) {
			model.Graph.Output[0].Type.GetTensorType().Shape = nil
			model.Graph.ValueInfo = []*onnxpb.ValueInfoProto{tensorInfo("y", 1, &onnxpb.TensorShapeProto_Dimension{}, symbolicDimension("any"))}
		}, []string{"x", "y"}, []string{"y"}, false},
		{"untyped intermediate declaration", func(model *onnxpb.ModelProto) {
			model.Graph.ValueInfo = []*onnxpb.ValueInfoProto{{Name: proto.String("y")}}
		}, []string{"x", "y"}, []string{"y"}, false},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			model := graphFixture()
			testcase.configure(model)
			compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
			if err != nil {
				t.Fatal(err)
			}

			values := compiled.Values()
			names := make([]string, len(values))
			for index, value := range values {
				names[index] = value.Name
			}

			if !reflect.DeepEqual(names, testcase.values) {
				t.Fatalf("values %v, want %v", names, testcase.values)
			}

			outputs := compiled.Outputs()
			names = make([]string, len(outputs))
			for index, output := range outputs {
				names[index] = output.Name
			}

			if !reflect.DeepEqual(names, testcase.outputs) {
				t.Fatalf("outputs %v, want %v", names, testcase.outputs)
			}

			if inputs := compiled.Inputs(); len(inputs) != 0 && inputs[0].Defaulted != testcase.defaulted {
				t.Fatalf("default flag: %v", inputs)
			}

			if value, found := compiled.Value("missing"); found || value.Name != "" {
				t.Fatalf("missing lookup: %v, %v", value, found)
			}
		})
	}
}

func TestGraphBranchJoinAndPlaceholders(t *testing.T) {
	model := graphFixture()
	model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(1), Dims: []int64{2}, FloatData: []float32{1, 2}}}
	model.Graph.Node = []*onnxpb.NodeProto{
		{Name: proto.String("same"), OpType: proto.String("Add"), Input: []string{"x", "w"}, Output: []string{"left"}},
		{Name: proto.String("same"), OpType: proto.String("Sub"), Input: []string{"x", "w"}, Output: []string{"right"}},
		{OpType: proto.String("Add"), Input: []string{"left", "right"}, Output: []string{"y"}},
		// Structurally retained despite no declared output path. RT-07 must reject
		// the unknown operator and validate the exact optional-slot positions.
		{OpType: proto.String("Unresolved"), Input: []string{"x", "", "w"}, Output: []string{"unused", "", "tail", ""}},
	}
	compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	nodes := compiled.Nodes()
	if len(nodes) != 4 || !reflect.DeepEqual(nodes[3].Inputs, []string{"x", "", "w"}) || !reflect.DeepEqual(nodes[3].Outputs, []string{"unused", "", "tail", ""}) {
		t.Fatalf("node/slot loss: %v", nodes)
	}

	for index, node := range nodes {
		if node.Index != index {
			t.Fatalf("node index %d, want %d", node.Index, index)
		}
	}

	if value, found := compiled.Value("y"); !found || value.Producer != 2 {
		t.Fatalf("join producer: %v, %v", value, found)
	}

	if value, found := compiled.Value("w"); !found || value.Producer != -1 || value.Input || value.Default == nil {
		t.Fatalf("shared initializer source: %v, %v", value, found)
	}

	if _, found := compiled.Value(""); found {
		t.Fatal("empty placeholder became a named value")
	}
}

func TestGraphInvalidStructure(t *testing.T) {
	tests := []struct {
		name      string
		configure func(*onnxpb.ModelProto)
		want      error
		text      string
	}{
		{"missing IR", func(m *onnxpb.ModelProto) { m.IrVersion = nil }, ErrInvalidModel, "ir_version"},
		{"unsupported IR", func(m *onnxpb.ModelProto) { m.IrVersion = proto.Int64(9) }, ErrUnsupported, "IR 8"},
		{"missing import", func(m *onnxpb.ModelProto) { m.OpsetImport = nil }, ErrInvalidModel, "opset_import"},
		{"nil import", func(m *onnxpb.ModelProto) { m.OpsetImport[0] = nil }, ErrInvalidModel, "opset_import[0]"},
		{"missing import version", func(m *onnxpb.ModelProto) { m.OpsetImport[0].Version = nil }, ErrInvalidModel, "version"},
		{"duplicate import", func(m *onnxpb.ModelProto) { m.OpsetImport = append(m.OpsetImport, m.OpsetImport[0]) }, ErrInvalidModel, "duplicate import"},
		{"later opset", func(m *onnxpb.ModelProto) { m.OpsetImport[0].Version = proto.Int64(14) }, ErrUnsupported, "version 14"},
		{"older opset", func(m *onnxpb.ModelProto) { m.OpsetImport[0].Version = proto.Int64(12) }, ErrUnsupported, "version 12"},
		{"custom import", func(m *onnxpb.ModelProto) { m.OpsetImport[0].Domain = proto.String("custom") }, ErrUnsupported, "custom"},
		{"literal ai.onnx", func(m *onnxpb.ModelProto) { m.OpsetImport[0].Domain = proto.String("ai.onnx") }, ErrUnsupported, "ai.onnx"},
		{"additional custom import", func(m *onnxpb.ModelProto) {
			m.OpsetImport = append(m.OpsetImport, &onnxpb.OperatorSetIdProto{Domain: proto.String("custom"), Version: proto.Int64(1)})
		}, ErrUnsupported, "custom"},
		{"missing graph", func(m *onnxpb.ModelProto) { m.Graph = nil }, ErrInvalidModel, "graph name"},
		{"empty graph name", func(m *onnxpb.ModelProto) { m.Graph.Name = proto.String("") }, ErrInvalidModel, "graph name"},
		{"functions", func(m *onnxpb.ModelProto) { m.Functions = []*onnxpb.FunctionProto{{}} }, ErrUnsupported, "functions"},
		{"training", func(m *onnxpb.ModelProto) { m.TrainingInfo = []*onnxpb.TrainingInfoProto{{}} }, ErrUnsupported, "training_info"},
		{"sparse initializer", func(m *onnxpb.ModelProto) { m.Graph.SparseInitializer = []*onnxpb.SparseTensorProto{{}} }, ErrUnsupported, "sparse_initializer"},
		{"nil input", func(m *onnxpb.ModelProto) { m.Graph.Input[0] = nil }, ErrInvalidModel, "value[0]"},
		{"empty input name", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Name = nil }, ErrInvalidModel, "value[0]"},
		{"duplicate input", func(m *onnxpb.ModelProto) { m.Graph.Input = append(m.Graph.Input, m.Graph.Input[0]) }, ErrInvalidModel, "input \"x\""},
		{"input missing type", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type = nil }, ErrInvalidModel, "value \"x\""},
		{"input unknown rank", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type.GetTensorType().Shape = nil }, ErrUnsupported, "value \"x\""},
		{"undefined input type", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type.GetTensorType().ElemType = nil }, ErrInvalidModel, "element type"},
		{"unsupported input type", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type.GetTensorType().ElemType = proto.Int32(11) }, ErrUnsupported, "value \"x\""},
		{"negative dimension", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type.GetTensorType().Shape.Dim[1] = fixedDimension(-1) }, ErrInvalidModel, "axis 1"},
		{"nil dimension", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type.GetTensorType().Shape.Dim[1] = nil }, ErrInvalidModel, "axis 1"},
		{"empty symbol", func(m *onnxpb.ModelProto) { m.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = symbolicDimension("") }, ErrInvalidModel, "dimension symbol"},
		{"nonleading symbol", func(m *onnxpb.ModelProto) {
			m.Graph.Input[0].Type.GetTensorType().Shape.Dim[1] = symbolicDimension("features")
		}, ErrUnsupported, "axis 1"},
		{"nonleading unknown", func(m *onnxpb.ModelProto) {
			m.Graph.Input[0].Type.GetTensorType().Shape.Dim[1] = &onnxpb.TensorShapeProto_Dimension{}
		}, ErrUnsupported, "axis 1"},
		{"nil output", func(m *onnxpb.ModelProto) { m.Graph.Output[0] = nil }, ErrInvalidModel, "value[0]"},
		{"duplicate graph output", func(m *onnxpb.ModelProto) { m.Graph.Output = append(m.Graph.Output, m.Graph.Output[0]) }, ErrInvalidModel, "output value \"y\""},
		{"output missing type", func(m *onnxpb.ModelProto) { m.Graph.Output[0].Type = nil }, ErrInvalidModel, "value \"y\""},
		{"output without producer", func(m *onnxpb.ModelProto) { m.Graph.Output[0].Name = proto.String("absent") }, ErrInvalidModel, "value \"absent\""},
		{"nil initializer", func(m *onnxpb.ModelProto) { m.Graph.Initializer = []*onnxpb.TensorProto{nil} }, ErrInvalidModel, "initializer[0]"},
		{"unnamed initializer", func(m *onnxpb.ModelProto) { m.Graph.Initializer = []*onnxpb.TensorProto{{}} }, ErrInvalidModel, "initializer[0]"},
		{"duplicate initializer", func(m *onnxpb.ModelProto) {
			m.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(1), FloatData: []float32{1}}, {Name: proto.String("w"), DataType: proto.Int32(1), FloatData: []float32{2}}}
		}, ErrInvalidModel, "value \"w\""},
		{"default type mismatch", func(m *onnxpb.ModelProto) {
			m.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(7), Dims: []int64{1, 2}, Int64Data: []int64{1, 2}}}
		}, ErrInvalidModel, "default initializer value \"x\""},
		{"default rank mismatch", func(m *onnxpb.ModelProto) {
			m.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(1), FloatData: []float32{1}}}
		}, ErrInvalidModel, "declared rank"},
		{"default fixed mismatch", func(m *onnxpb.ModelProto) {
			m.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{1, 3}, FloatData: []float32{1, 2, 3}}}
		}, ErrInvalidModel, "fixed axis 1"},
		{"nil node", func(m *onnxpb.ModelProto) { m.Graph.Node[0] = nil }, ErrInvalidModel, "node[0]"},
		{"missing operator", func(m *onnxpb.ModelProto) { m.Graph.Node[0].OpType = nil }, ErrInvalidModel, "node[0]"},
		{"no node slots", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Input = nil; m.Graph.Node[0].Output = nil }, ErrInvalidModel, "node[0]"},
		{"node custom domain", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Domain = proto.String("custom") }, ErrUnsupported, "node[0]"},
		{"node ai.onnx alias", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Domain = proto.String("ai.onnx") }, ErrUnsupported, "node[0]"},
		{"node function overload", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Overload = proto.String("special") }, ErrUnsupported, "node[0]"},
		{"missing node input", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Input[0] = "absent" }, ErrInvalidModel, "input[0] value \"absent\""},
		{"forward reference", func(m *onnxpb.ModelProto) {
			m.Graph.Node[0].Input[0] = "later"
			m.Graph.Node = append(m.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"x"}, Output: []string{"later"}})
		}, ErrInvalidModel, "node[0]"},
		{"self cycle", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Input[0] = "y" }, ErrInvalidModel, "cycle"},
		{"two node cycle", func(m *onnxpb.ModelProto) {
			m.Graph.Node[0].Input[0] = "later"
			m.Graph.Node = append(m.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"y"}, Output: []string{"later"}})
		}, ErrInvalidModel, "cycle"},
		{"node overwrites input", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Output[0] = "x" }, ErrInvalidModel, "value \"x\""},
		{"repeated node output", func(m *onnxpb.ModelProto) { m.Graph.Node[0].Output = []string{"y", "y"} }, ErrInvalidModel, "output[1]"},
		{"node overwrites initializer", func(m *onnxpb.ModelProto) {
			m.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("y"), DataType: proto.Int32(1), FloatData: []float32{1}}}
		}, ErrInvalidModel, "value \"y\""},
		{"duplicate node producers", func(m *onnxpb.ModelProto) {
			m.Graph.Node = append(m.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"x"}, Output: []string{"y"}})
		}, ErrInvalidModel, "node[1]"},
		{"invalid unused node", func(m *onnxpb.ModelProto) {
			m.Graph.Node = append(m.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Unresolved"), Input: []string{"absent"}, Output: []string{"unused"}})
		}, ErrInvalidModel, "node[1]"},
		{"orphan value_info", func(m *onnxpb.ModelProto) { m.Graph.ValueInfo = []*onnxpb.ValueInfoProto{tensorInfo("absent", 1)} }, ErrInvalidModel, "value_info value \"absent\""},
		{"duplicate value_info", func(m *onnxpb.ModelProto) {
			m.Graph.ValueInfo = []*onnxpb.ValueInfoProto{{Name: proto.String("y")}, {Name: proto.String("y")}}
		}, ErrInvalidModel, "duplicate declaration"},
		{"contradictory metadata type", func(m *onnxpb.ModelProto) {
			m.Graph.ValueInfo = []*onnxpb.ValueInfoProto{tensorInfo("y", 7, symbolicDimension("N"), fixedDimension(2))}
		}, ErrInvalidModel, "contradictory declared types"},
		{"contradictory metadata rank", func(m *onnxpb.ModelProto) { m.Graph.ValueInfo = []*onnxpb.ValueInfoProto{tensorInfo("y", 1)} }, ErrInvalidModel, "contradictory declared ranks"},
		{"contradictory metadata dimension", func(m *onnxpb.ModelProto) {
			m.Graph.ValueInfo = []*onnxpb.ValueInfoProto{tensorInfo("y", 1, symbolicDimension("N"), fixedDimension(3))}
		}, ErrInvalidModel, "contradictory fixed axis 1"},
		{"initializer metadata contradiction", func(m *onnxpb.ModelProto) {
			m.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(1), FloatData: []float32{1}}}
			m.Graph.ValueInfo = []*onnxpb.ValueInfoProto{tensorInfo("w", 7)}
		}, ErrInvalidModel, "value \"w\""},
		{"duplicate model metadata", func(m *onnxpb.ModelProto) {
			m.MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("same")}, {Key: proto.String("same")}}
		}, ErrInvalidModel, "metadata key \"same\""},
		{"nil node metadata", func(m *onnxpb.ModelProto) { m.Graph.Node[0].MetadataProps = []*onnxpb.StringStringEntryProto{nil} }, ErrInvalidModel, "node[0]"},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			model := graphFixture()
			testcase.configure(model)
			compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
			if compiled != nil || !errors.Is(err, testcase.want) || !strings.Contains(err.Error(), testcase.text) {
				t.Fatalf("Build: %v, %v; want %v containing %q", compiled, err, testcase.want, testcase.text)
			}
		})
	}
}

func TestGraphUnsupportedValueKinds(t *testing.T) {
	types := []struct {
		name string
		wire *onnxpb.TypeProto
	}{
		{"sequence", &onnxpb.TypeProto{Value: &onnxpb.TypeProto_SequenceType{SequenceType: &onnxpb.TypeProto_Sequence{}}}},
		{"map", &onnxpb.TypeProto{Value: &onnxpb.TypeProto_MapType{MapType: &onnxpb.TypeProto_Map{}}}},
		{"optional", &onnxpb.TypeProto{Value: &onnxpb.TypeProto_OptionalType{OptionalType: &onnxpb.TypeProto_Optional{}}}},
		{"sparse", &onnxpb.TypeProto{Value: &onnxpb.TypeProto_SparseTensorType{SparseTensorType: &onnxpb.TypeProto_SparseTensor{}}}},
	}

	for _, testcase := range types {
		t.Run(testcase.name, func(t *testing.T) {
			model := graphFixture()
			model.Graph.Output[0].Type = testcase.wire
			compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
			if compiled != nil || !errors.Is(err, ErrUnsupported) || !strings.Contains(err.Error(), "value \"y\"") {
				t.Fatalf("unsupported value: %v, %v", compiled, err)
			}
		})
	}
}

func TestGraphProfileFixtureAndActualLength(t *testing.T) {
	data, err := os.ReadFile("../../testdata/profile/identity.onnx")
	if err != nil {
		t.Fatal(err)
	}

	// A repeated known field with a noncanonical varint disappears on re-encode.
	// The build must charge its original bytes rather than the smaller proto.Size.
	data = append(data, 0x08, 0x88, 0x00)
	data = protowire.AppendTag(data, 99, protowire.BytesType)
	data = protowire.AppendString(data, "unknown")
	model, err := decode.ParseModel(context.Background(), data, decode.Limits{})
	if err != nil {
		t.Fatal(err)
	}

	compiled, err := Build(context.Background(), model, int64(len(data)), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	load, _ := compiled.Usage()
	if int64(proto.Size(model)) >= int64(len(data)) {
		t.Fatal("fixture does not distinguish encoded size from proto.Size")
	}

	limits := Limits{Tensor: decode.TensorLimits{MaxLoadBytes: load}}
	if _, err = Build(context.Background(), model, int64(len(data)), limits); err != nil {
		t.Fatalf("exact load boundary: %v", err)
	}

	limits.Tensor.MaxLoadBytes--
	if compiled, err = Build(context.Background(), model, int64(len(data)), limits); compiled != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("encoded+decoded+snapshot boundary: %v, %v", compiled, err)
	}

	if len(model.ProtoReflect().GetUnknown()) == 0 {
		t.Fatal("source unknown fields were discarded")
	}
}

func TestGraphConcurrentSnapshots(t *testing.T) {
	model := graphFixture()
	compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	var readers sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		readers.Go(func() {
			for iteration := 0; iteration < 20; iteration++ {
				inputs := compiled.Inputs()
				inputs[0].Dimensions[0].Symbol = "changed"
				nodes := compiled.Nodes()
				nodes[0].Inputs[0] = "changed"
				compiled.Values()
				compiled.Outputs()
				compiled.Metadata()
			}
		})
	}
	readers.Wait()
	if compiled.Inputs()[0].Dimensions[0].Symbol != "N" || compiled.Nodes()[0].Inputs[0] != "x" {
		t.Fatal("snapshot mutation changed graph")
	}
}

func TestGraphDefaultContractRemainsSymbolic(t *testing.T) {
	model := graphFixture()
	model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("x"), DataType: proto.Int32(1), Dims: []int64{1, 2}, FloatData: []float32{3, 4}}}
	compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	value, found := compiled.Value("x")
	if !found || !value.Input || value.Default == nil || value.Declarations[0].Dimensions[0].Kind != SymbolicDimension {
		t.Fatalf("default metadata: %v", value)
	}

	// Batch three is permitted by the input contract even though the default is
	// batch one. This is metadata validation, not session execution.
	override, err := tensor.FromFloat32([]int64{3, 2}, make([]float32, 6))
	if err != nil {
		t.Fatal(err)
	}

	constant := &Constant{value: override}
	if err = validateConstant(compiled.Inputs()[0], constant); err != nil {
		t.Fatalf("symbolic override was narrowed to default shape: %v", err)
	}
}
