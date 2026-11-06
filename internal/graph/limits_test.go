package graph

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestGraphLimits(t *testing.T) {
	tests := []struct {
		name      string
		limits    Limits
		configure func(*onnxpb.ModelProto)
		text      string
	}{
		{"nodes", Limits{MaxNodes: 1}, func(m *onnxpb.ModelProto) { m.Graph.Node = append(m.Graph.Node, &onnxpb.NodeProto{}) }, "node count"},
		{"values", Limits{MaxValues: 1}, func(m *onnxpb.ModelProto) {}, "value \"y\""},
		{"attributes aggregate", Limits{MaxAttributes: 1}, func(m *onnxpb.ModelProto) {
			m.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("a"), Type: onnxpb.AttributeProto_INT.Enum()}}
			m.Graph.Node = append(m.Graph.Node, &onnxpb.NodeProto{OpType: proto.String("Identity"), Input: []string{"x"}, Output: []string{"z"}, Attribute: []*onnxpb.AttributeProto{{Name: proto.String("b"), Type: onnxpb.AttributeProto_INT.Enum()}}})
		}, "node[1]"},
		{"rank", Limits{Tensor: decode.TensorLimits{MaxRank: 1}}, func(m *onnxpb.ModelProto) {}, "value \"x\""},
		{"dimension after zero", Limits{Tensor: decode.TensorLimits{MaxDimension: 2}}, func(m *onnxpb.ModelProto) {
			m.Graph.Input[0] = tensorInfo("x", 1, fixedDimension(0), fixedDimension(3))
		}, "axis 1"},
		{"name", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) { m.Graph.Name = proto.String("long-name") }, "model.graph.name"},
		{"symbol", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) {
			m.Graph.Input[0].Type.GetTensorType().Shape.Dim[0] = symbolicDimension("long-name")
		}, "dim_param"},
		{"producer", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) { m.ProducerName = proto.String("long-name") }, "producer_name"},
		{"documentation", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) { m.Graph.DocString = proto.String("long-name") }, "doc_string"},
		{"metadata string", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) {
			m.MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("key"), Value: proto.String("long-name")}}
		}, "metadata_props[0].value"},
		{"attribute byte string", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) {
			m.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("a"), Type: onnxpb.AttributeProto_STRING.Enum(), S: []byte("long-name")}}
		}, "node[0].attribute[0].s"},
		{"attribute byte strings", Limits{MaxNameBytes: 8}, func(m *onnxpb.ModelProto) {
			m.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("a"), Type: onnxpb.AttributeProto_STRINGS.Enum(), Strings: [][]byte{[]byte("long-name")}}}
		}, "strings[0]"},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			model := graphFixture()
			testcase.configure(model)
			compiled, err := Build(context.Background(), model, encodedLength(t, model), testcase.limits)
			if compiled != nil || !errors.Is(err, ErrLimit) || !strings.Contains(err.Error(), testcase.text) {
				t.Fatalf("limit: %v, %v; want %q", compiled, err, testcase.text)
			}
		})
	}
}

func TestGraphAggregateConstantAndSnapshotBudgets(t *testing.T) {
	model := graphFixture()
	model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(1), FloatData: []float32{1}}}
	model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{
		{Name: proto.String("tensor"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(7), RawData: []byte{1, 0, 0, 1, 0, 0, 0, 0}}},
		{Name: proto.String("list"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{1, 2, 3}},
	}
	encoded := encodedLength(t, model)
	compiled, err := Build(context.Background(), model, encoded, Limits{})
	if err != nil {
		t.Fatal(err)
	}

	load, owned := compiled.Usage()
	if owned != 12 {
		t.Fatalf("owned bytes %d, want 12", owned)
	}

	decoder, err := decode.NewTensorDecoder(context.Background(), model, encoded, decode.TensorLimits{})
	if err != nil {
		t.Fatal(err)
	}

	decodedLoad, _ := decoder.Usage()
	// Two two-axis declarations, one int64 list snapshot, one typed copy,
	// and raw staging plus its owned copy all coexist with the complete wire.
	if load != decodedLoad+2*2*8+3*8+4+2*8 {
		t.Fatalf("load %d, want %d", load, decodedLoad+2*2*8+3*8+4+2*8)
	}

	tests := []struct {
		name   string
		limits decode.TensorLimits
		want   error
	}{
		{"exact combined budgets", decode.TensorLimits{MaxLoadBytes: load, MaxInitializerBytes: owned}, nil},
		{"combined load exceeds by one", decode.TensorLimits{MaxLoadBytes: load - 1}, ErrLimit},
		{"combined constants exceed by one", decode.TensorLimits{MaxInitializerBytes: owned - 1}, ErrLimit},
		{"tensor limit covers attributes", decode.TensorLimits{MaxTensorBytes: 7}, ErrLimit},
		{"decoded traversal recursion", decode.TensorLimits{MaxRecursion: 2}, ErrLimit},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			result, buildErr := Build(context.Background(), model, encoded, Limits{Tensor: testcase.limits})
			if testcase.want == nil {
				if result == nil || buildErr != nil {
					t.Fatalf("exact boundary: %v, %v", result, buildErr)
				}

				return
			}

			if result != nil || !errors.Is(buildErr, testcase.want) {
				t.Fatalf("budget failure: %v, %v", result, buildErr)
			}
		})
	}
}

func TestGraphInvalidArguments(t *testing.T) {
	tests := []struct {
		name   string
		ctx    context.Context
		model  *onnxpb.ModelProto
		bytes  int64
		limits Limits
	}{
		{"nil context", nil, graphFixture(), 0, Limits{}},
		{"nil model", context.Background(), nil, 0, Limits{}},
		{"negative encoded size", context.Background(), graphFixture(), -1, Limits{}},
		{"negative nodes", context.Background(), graphFixture(), 0, Limits{MaxNodes: -1}},
		{"negative values", context.Background(), graphFixture(), 0, Limits{MaxValues: -1}},
		{"negative attributes", context.Background(), graphFixture(), 0, Limits{MaxAttributes: -1}},
		{"negative names", context.Background(), graphFixture(), 0, Limits{MaxNameBytes: -1}},
		{"negative tensor limit", context.Background(), graphFixture(), 0, Limits{Tensor: decode.TensorLimits{MaxLoadBytes: -1}}},
		{"negative rank", context.Background(), graphFixture(), 0, Limits{Tensor: decode.TensorLimits{MaxRank: -1}}},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			compiled, err := Build(testcase.ctx, testcase.model, testcase.bytes, testcase.limits)
			if compiled != nil || !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("invalid argument: %v, %v", compiled, err)
			}
		})
	}
}
