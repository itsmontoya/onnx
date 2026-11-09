package decode

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestDecodedPayloadDiagnosticOrder(t *testing.T) {
	wire := &onnxpb.TensorProto{
		Dims: []int64{1}, DataType: proto.Int32(1), FloatData: []float32{1},
		Name: proto.String("weight"), DocString: proto.String("description"),
	}
	for repeat := 0; repeat < 100; repeat++ {
		decoder, err := NewTensorDecoder(context.Background(), wire, 0, TensorLimits{MaxLoadBytes: 12})
		if decoder != nil || !errors.Is(err, ErrLimit) || !strings.Contains(err.Error(), "onnx.TensorProto.float_data") {
			t.Fatalf("unstable first overflowing field: %v, %v", decoder, err)
		}
	}
}

func TestDecodedPayloadWidths(t *testing.T) {
	// Account each decoded element at storage width, even when the eventual
	// tensor type/attribute will be rejected by semantic validation.
	wire := &onnxpb.TensorProto{
		DataType: proto.Int32(1), Dims: []int64{0}, FloatData: []float32{1, 2},
		Int32Data: []int32{1, 2}, Int64Data: []int64{1}, DoubleData: []float64{1},
		Uint64Data: []uint64{1}, RawData: []byte{1, 2, 3}, StringData: [][]byte{{1, 2}, {3}},
		Name: proto.String("name"), DocString: proto.String("doc"),
	}

	wire.ProtoReflect().SetUnknown([]byte{0xa0, 0x06, 0x01})
	// 4 + 8 + 8 + 8 + 8 + 8 + 8 + 3 + 3 + 4 + 3 + 3 = 68.
	decoder, err := NewTensorDecoder(context.Background(), wire, 5, TensorLimits{MaxLoadBytes: 73})
	if err != nil {
		t.Fatal(err)
	}

	if load, owned := decoder.Usage(); load != 73 || owned != 0 {
		t.Fatalf("decoded payload width accounting: %d, %d", load, owned)
	}

	if value, decodeErr := NewTensorDecoder(context.Background(), wire, 5, TensorLimits{MaxLoadBytes: 72}); value != nil || !errors.Is(decodeErr, ErrLimit) {
		t.Fatalf("decoded payload over limit: %v, %v", value, decodeErr)
	}

	attribute := &onnxpb.AttributeProto{F: proto.Float32(1), I: proto.Int64(16777217), S: []byte{1, 2}, Floats: []float32{1}, Ints: []int64{2}, Strings: [][]byte{{3}}}
	decoder, err = NewTensorDecoder(context.Background(), attribute, 0, TensorLimits{})
	if err != nil {
		t.Fatal(err)
	}

	if load, owned := decoder.Usage(); load != 27 || owned != 0 {
		t.Fatalf("attribute payloads not charged: %d, %d", load, owned)
	}
}

func TestTensorDecoderSparseAndRecursion(t *testing.T) {
	sparse := &onnxpb.SparseTensorProto{}
	for _, wire := range []proto.Message{
		sparse,
		&onnxpb.ModelProto{Graph: &onnxpb.GraphProto{SparseInitializer: []*onnxpb.SparseTensorProto{sparse}}},
		&onnxpb.AttributeProto{SparseTensor: sparse},
		&onnxpb.AttributeProto{SparseTensors: []*onnxpb.SparseTensorProto{sparse}},
	} {
		if value, err := NewTensorDecoder(context.Background(), wire, 0, TensorLimits{}); value != nil || !errors.Is(err, ErrUnsupported) {
			t.Fatalf("sparse %T: %v, %v", wire, value, err)
		}
	}

	value := &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}
	attribute := &onnxpb.AttributeProto{T: value}
	if decoder, err := NewTensorDecoder(context.Background(), attribute, 0, TensorLimits{MaxRecursion: 1}); decoder != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("payload recursion: %v, %v", decoder, err)
	}

	if _, err := NewTensorDecoder(context.Background(), attribute, 0, TensorLimits{MaxRecursion: 2}); err != nil {
		t.Fatalf("payload recursion exact boundary: %v", err)
	}

	// Constructed objects can contain cycles even though parsed wire cannot.
	graph := &onnxpb.GraphProto{}
	graph.Node = []*onnxpb.NodeProto{{Attribute: []*onnxpb.AttributeProto{{G: graph}}}}
	if decoder, err := NewTensorDecoder(context.Background(), graph, 0, TensorLimits{}); decoder != nil || !errors.Is(err, ErrLimit) {
		t.Fatalf("cyclic object recursion: %v, %v", decoder, err)
	}
}
