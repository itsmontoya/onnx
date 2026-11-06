package graph

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestAttributeSnapshots(t *testing.T) {
	model := graphFixture()
	model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{
		{Name: proto.String("f"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(math.Float32frombits(0x7fc01234))},
		{Name: proto.String("i"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(math.MinInt64)},
		{Name: proto.String("s"), Type: onnxpb.AttributeProto_STRING.Enum(), S: []byte("raw")},
		{Name: proto.String("fs"), Type: onnxpb.AttributeProto_FLOATS.Enum(), Floats: []float32{math.Float32frombits(0x80000000), 2}},
		{Name: proto.String("is"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{16777217, math.MaxInt64}},
		{Name: proto.String("ss"), Type: onnxpb.AttributeProto_STRINGS.Enum(), Strings: [][]byte{[]byte("one"), []byte("two")}},
		{Name: proto.String("t"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(7), Int64Data: []int64{16777217}}},
		{Name: proto.String("ts"), Type: onnxpb.AttributeProto_TENSORS.Enum(), Tensors: []*onnxpb.TensorProto{{DataType: proto.Int32(1), FloatData: []float32{2}}}},
		{Name: proto.String("empty"), Type: onnxpb.AttributeProto_INTS.Enum()},
		{Name: proto.String("zero"), Type: onnxpb.AttributeProto_INT.Enum()},
		{Name: proto.String("empty string"), Type: onnxpb.AttributeProto_STRING.Enum()},
	}
	compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	attributes := compiled.Nodes()[0].Attributes
	if math.Float32bits(attributes[0].Float) != 0x7fc01234 || attributes[1].Int != math.MinInt64 || string(attributes[2].String) != "raw" || math.Float32bits(attributes[3].Floats[0]) != 0x80000000 || attributes[4].Ints[0] != 16777217 || string(attributes[5].Strings[1]) != "two" {
		t.Fatalf("attribute values changed: %v", attributes)
	}

	if values, accessErr := attributes[6].Tensor.Int64Values(); accessErr != nil || !reflect.DeepEqual(values, []int64{16777217}) {
		t.Fatalf("tensor attribute: %v, %v", values, accessErr)
	}

	if values, accessErr := attributes[7].Tensors[0].Float32Values(); accessErr != nil || !reflect.DeepEqual(values, []float32{2}) {
		t.Fatalf("tensor list attribute: %v, %v", values, accessErr)
	}

	if attributes[8].Kind != IntsAttribute || len(attributes[8].Ints) != 0 || attributes[9].Kind != IntAttribute || attributes[9].Int != 0 || attributes[10].Kind != StringAttribute || len(attributes[10].String) != 0 {
		t.Fatal("default scalar/empty list kinds were lost")
	}

	// Mutate original payloads and separate accessor snapshots independently.
	model.Graph.Node[0].Attribute[2].S[0] = 'X'
	model.Graph.Node[0].Attribute[3].Floats[0] = 99
	model.Graph.Node[0].Attribute[4].Ints[0] = 99
	model.Graph.Node[0].Attribute[5].Strings[0][0] = 'X'
	model.Graph.Node[0].Attribute[6].T.Int64Data[0] = 99
	model.Graph.Node[0].Attribute[7].Tensors[0].FloatData[0] = 99
	attributes[2].String[0] = 'Y'
	attributes[3].Floats[0] = 100
	attributes[4].Ints[0] = 100
	attributes[5].Strings[0][0] = 'Y'
	attributes[7].Tensors[0] = nil
	fresh := compiled.Nodes()[0].Attributes
	if string(fresh[2].String) != "raw" || math.Float32bits(fresh[3].Floats[0]) != 0x80000000 || fresh[4].Ints[0] != 16777217 || string(fresh[5].Strings[0]) != "one" || fresh[7].Tensors[0] == nil {
		t.Fatal("wire or snapshot mutation changed attributes")
	}

	if values, accessErr := fresh[6].Tensor.Int64Values(); accessErr != nil || values[0] != 16777217 {
		t.Fatalf("wire mutation changed tensor: %v, %v", values, accessErr)
	}
}

func TestAttributeInvalidStructure(t *testing.T) {
	tests := []struct {
		name string
		wire *onnxpb.AttributeProto
		want error
	}{
		{"nil", nil, ErrInvalidModel},
		{"unnamed", &onnxpb.AttributeProto{Type: onnxpb.AttributeProto_INT.Enum()}, ErrInvalidModel},
		{"missing type", &onnxpb.AttributeProto{Name: proto.String("a")}, ErrInvalidModel},
		{"undefined type", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_UNDEFINED.Enum()}, ErrInvalidModel},
		{"unknown type", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_AttributeType(99).Enum()}, ErrUnsupported},
		{"mismatched payload", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_INT.Enum(), F: proto.Float32(0)}, ErrInvalidModel},
		{"multiple payloads", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(0), Floats: []float32{0}}, ErrInvalidModel},
		{"missing tensor", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TENSOR.Enum()}, ErrInvalidModel},
		{"nil tensor list entry", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TENSORS.Enum(), Tensors: []*onnxpb.TensorProto{nil}}, ErrInvalidModel},
		{"ref attribute", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_INT.Enum(), RefAttrName: proto.String("a")}, ErrUnsupported},
		{"subgraph type", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_GRAPH.Enum()}, ErrUnsupported},
		{"subgraph payload", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_INT.Enum(), G: &onnxpb.GraphProto{}}, ErrUnsupported},
		{"subgraph list", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_GRAPHS.Enum(), Graphs: []*onnxpb.GraphProto{{}}}, ErrUnsupported},
		{"sparse payload", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_SPARSE_TENSOR.Enum(), SparseTensor: &onnxpb.SparseTensorProto{}}, ErrUnsupported},
		{"sparse list", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_SPARSE_TENSORS.Enum(), SparseTensors: []*onnxpb.SparseTensorProto{{}}}, ErrUnsupported},
		{"type attribute", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TYPE_PROTO.Enum(), Tp: &onnxpb.TypeProto{}}, ErrUnsupported},
		{"type list", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TYPE_PROTOS.Enum()}, ErrUnsupported},
		{"tensor wrong dtype", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(11)}}, ErrUnsupported},
		{"tensor external data", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(1), DataLocation: onnxpb.TensorProto_EXTERNAL.Enum()}}, ErrUnsupported},
		{"malformed tensor", &onnxpb.AttributeProto{Name: proto.String("a"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(7), Int64Data: []int64{1, 2}}}, decode.ErrInvalidTensor},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			model := graphFixture()
			model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{testcase.wire}
			compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
			if compiled != nil || !errors.Is(err, testcase.want) || !strings.Contains(err.Error(), "node[0]") || !strings.Contains(err.Error(), "attribute") {
				t.Fatalf("attribute rejection: %v, %v", compiled, err)
			}
		})
	}

	model := graphFixture()
	model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("same"), Type: onnxpb.AttributeProto_INT.Enum()}, {Name: proto.String("same"), Type: onnxpb.AttributeProto_INT.Enum()}}
	if compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{}); compiled != nil || !errors.Is(err, ErrInvalidModel) || !strings.Contains(err.Error(), "duplicate attribute") {
		t.Fatalf("duplicate attributes: %v, %v", compiled, err)
	}
}
