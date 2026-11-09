package decode

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/emptypb"
)

func TestTensorDecoderReservePayload(t *testing.T) {
	ctx := context.Background()
	wire := &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}
	decoder, err := NewTensorDecoder(ctx, wire, 10, TensorLimits{MaxLoadBytes: 30})
	if err != nil {
		t.Fatal(err)
	}

	before, _ := decoder.Usage()
	if err = decoder.ReservePayload(ctx, 30-before); err != nil {
		t.Fatal(err)
	}

	if load, owned := decoder.Usage(); load != 30 || owned != 0 {
		t.Fatalf("reservation usage: %d, %d", load, owned)
	}

	tests := []struct {
		name    string
		decoder *TensorDecoder
		ctx     context.Context
		bytes   int64
		want    error
	}{
		{"excess", decoder, ctx, 1, ErrLimit},
		{"negative", decoder, ctx, -1, ErrInvalidArgument},
		{"nil decoder", nil, ctx, 0, ErrInvalidArgument},
		{"zero decoder", &TensorDecoder{}, ctx, 0, ErrInvalidArgument},
		{"nil context", decoder, nil, 0, ErrInvalidArgument},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			if reserveErr := testcase.decoder.ReservePayload(testcase.ctx, testcase.bytes); !errors.Is(reserveErr, testcase.want) {
				t.Fatalf("reservation: %v", reserveErr)
			}

			if load, owned := decoder.Usage(); load != 30 || owned != 0 {
				t.Fatal("failed reservation changed usage")
			}
		})
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err = decoder.ReservePayload(cancelled, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled reservation: %v", err)
	}
}

func TestTensorDecoderLiteralWire(t *testing.T) {
	tests := []struct {
		name     string
		wire     []byte
		shape    []int64
		bits     []uint32
		integers []int64
	}{
		{
			name: "typed float special bits",
			wire: []byte{0x08, 0x05, 0x10, 0x01, 0x22, 0x14,
				0x01, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0x80, 0x7f,
				0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0xc0, 0x3f},
			shape: []int64{5}, bits: []uint32{0x7fc00001, 0x7f800000, 0xff800000, 0x80000000, 0x3fc00000},
		},
		{
			name: "raw float special bits",
			wire: []byte{0x08, 0x05, 0x10, 0x01, 0x4a, 0x14,
				0x01, 0x00, 0xc0, 0x7f, 0x00, 0x00, 0x80, 0x7f,
				0x00, 0x00, 0x80, 0xff, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0xc0, 0x3f},
			shape: []int64{5}, bits: []uint32{0x7fc00001, 0x7f800000, 0xff800000, 0x80000000, 0x3fc00000},
		},
		{
			name: "packed exact integers",
			wire: []byte{0x08, 0x02, 0x10, 0x07, 0x3a, 0x0e,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x81, 0x80, 0x80, 0x08},
			shape: []int64{2}, integers: []int64{-1, 16777217},
		},
		{
			name: "unpacked exact integers",
			wire: []byte{0x08, 0x02, 0x10, 0x07, 0x38,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x38, 0x81, 0x80, 0x80, 0x08},
			shape: []int64{2}, integers: []int64{-1, 16777217},
		},
		{
			name: "raw exact integers and extrema",
			wire: []byte{0x08, 0x04, 0x10, 0x07, 0x4a, 0x20,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
				0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
				0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f},
			shape: []int64{4}, integers: []int64{-1, 16777217, math.MinInt64, math.MaxInt64},
		},
		{
			name:  "unpacked float scalar",
			wire:  []byte{0x10, 0x01, 0x25, 0x00, 0x00, 0x80, 0x3f},
			shape: []int64{}, bits: []uint32{0x3f800000},
		},
		{
			name:  "raw int scalar",
			wire:  []byte{0x10, 0x07, 0x4a, 0x08, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00},
			shape: []int64{}, integers: []int64{16777217},
		},
		{
			name:  "empty absent payload",
			wire:  []byte{0x08, 0x02, 0x08, 0x00, 0x08, 0x03, 0x10, 0x01},
			shape: []int64{2, 0, 3}, bits: []uint32{},
		},
		{
			name:  "empty present raw payload",
			wire:  []byte{0x08, 0x00, 0x10, 0x07, 0x4a, 0x00},
			shape: []int64{0}, integers: []int64{},
		},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			var (
				wire    *onnxpb.TensorProto
				decoder *TensorDecoder
				value   *tensor.Tensor
				err     error
			)

			if wire, err = ParseTensor(context.Background(), testcase.wire, Limits{}); err != nil {
				t.Fatal(err)
			}

			if decoder, err = NewTensorDecoder(context.Background(), wire, int64(len(testcase.wire)), TensorLimits{}); err != nil {
				t.Fatal(err)
			}

			if value, err = decoder.Decode(context.Background(), wire); err != nil {
				t.Fatal(err)
			}

			// Published storage must survive clearing both serialized and decoded data.
			clear(testcase.wire)
			clear(wire.RawData)
			clear(wire.FloatData)
			clear(wire.Int64Data)
			clear(wire.Dims)
			if !reflect.DeepEqual(value.Shape(), testcase.shape) {
				t.Fatalf("shape %v, want %v", value.Shape(), testcase.shape)
			}

			if testcase.bits != nil {
				var floats []float32
				if floats, err = value.Float32Values(); err != nil {
					t.Fatal(err)
				}

				bits := make([]uint32, len(floats))
				for index, item := range floats {
					bits[index] = math.Float32bits(item)
				}

				if !reflect.DeepEqual(bits, testcase.bits) {
					t.Fatalf("float bits %x, want %x", bits, testcase.bits)
				}
			} else {
				var integers []int64
				if integers, err = value.Int64Values(); err != nil {
					t.Fatal(err)
				}

				if !reflect.DeepEqual(integers, testcase.integers) {
					t.Fatalf("integers %v, want %v", integers, testcase.integers)
				}
			}
		})
	}
}

func TestTensorDecoderInvalidValues(t *testing.T) {
	tests := []struct {
		name   string
		value  *onnxpb.TensorProto
		limits TensorLimits
		want   error
		text   string
	}{
		{"missing type", &onnxpb.TensorProto{}, TensorLimits{}, ErrUnsupported, "type 0"},
		{"double type", &onnxpb.TensorProto{DataType: proto.Int32(11)}, TensorLimits{}, ErrUnsupported, "type 11"},
		{"unknown type", &onnxpb.TensorProto{DataType: proto.Int32(999)}, TensorLimits{}, ErrUnsupported, "type 999"},
		{"segment", &onnxpb.TensorProto{Segment: &onnxpb.TensorProto_Segment{}}, TensorLimits{}, ErrUnsupported, "segments"},
		{"external", &onnxpb.TensorProto{DataLocation: onnxpb.TensorProto_EXTERNAL.Enum()}, TensorLimits{}, ErrUnsupported, "resolver"},
		{"unknown location", &onnxpb.TensorProto{DataLocation: onnxpb.TensorProto_DataLocation(9).Enum()}, TensorLimits{}, ErrUnsupported, "location 9"},
		{"external metadata", &onnxpb.TensorProto{ExternalData: []*onnxpb.StringStringEntryProto{{Key: proto.String("location"), Value: proto.String("weights.bin")}}}, TensorLimits{}, ErrUnsupported, "external"},
		{"scalar missing value", &onnxpb.TensorProto{DataType: proto.Int32(1)}, TensorLimits{}, ErrInvalidTensor, "count 0, want 1"},
		{"negative dimension", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{-1}}, TensorLimits{}, ErrInvalidTensor, "axis 0"},
		{"negative after zero", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{0, -1}}, TensorLimits{}, ErrInvalidTensor, "axis 1"},
		{"wrong float count", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{2}, FloatData: []float32{1}}, TensorLimits{}, ErrInvalidTensor, "count 1, want 2"},
		{"wrong int count", &onnxpb.TensorProto{DataType: proto.Int32(7), Int64Data: []int64{1, 2}}, TensorLimits{}, ErrInvalidTensor, "count 2, want 1"},
		{"empty with data", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{0}, FloatData: []float32{1}}, TensorLimits{}, ErrInvalidTensor, "want 0"},
		{"short raw float", &onnxpb.TensorProto{DataType: proto.Int32(1), RawData: []byte{0, 0, 0}}, TensorLimits{}, ErrInvalidTensor, "raw byte count 3, want 4"},
		{"long raw int", &onnxpb.TensorProto{DataType: proto.Int32(7), RawData: make([]byte, 9)}, TensorLimits{}, ErrInvalidTensor, "raw byte count 9, want 8"},
		{"present empty scalar raw", &onnxpb.TensorProto{DataType: proto.Int32(1), RawData: []byte{}}, TensorLimits{}, ErrInvalidTensor, "raw byte count 0, want 4"},
		{"conflicting raw float", &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}, RawData: make([]byte, 4)}, TensorLimits{}, ErrInvalidTensor, "conflicting"},
		{"conflicting empty raw int", &onnxpb.TensorProto{DataType: proto.Int32(7), Int64Data: []int64{1}, RawData: []byte{}}, TensorLimits{}, ErrInvalidTensor, "conflicting"},
		{"float with int payload", &onnxpb.TensorProto{DataType: proto.Int32(1), Int64Data: []int64{1}}, TensorLimits{}, ErrInvalidTensor, "incompatible"},
		{"int with float payload", &onnxpb.TensorProto{DataType: proto.Int32(7), FloatData: []float32{1}}, TensorLimits{}, ErrInvalidTensor, "incompatible"},
		{"int32 payload", &onnxpb.TensorProto{DataType: proto.Int32(1), Int32Data: []int32{1}}, TensorLimits{}, ErrInvalidTensor, "incompatible"},
		{"empty string element", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{0}, StringData: [][]byte{{}}}, TensorLimits{}, ErrInvalidTensor, "incompatible"},
		{"double payload", &onnxpb.TensorProto{DataType: proto.Int32(1), DoubleData: []float64{1}}, TensorLimits{}, ErrInvalidTensor, "incompatible"},
		{"uint64 payload", &onnxpb.TensorProto{DataType: proto.Int32(7), Uint64Data: []uint64{1}}, TensorLimits{}, ErrInvalidTensor, "incompatible"},
		{"rank limit", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{1, 0}}, TensorLimits{MaxRank: 1}, ErrLimit, "rank 2"},
		{"dimension limit after zero", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{0, 3}}, TensorLimits{MaxDimension: 2}, ErrLimit, "axis 1"},
		{"huge allocation", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{1 << 30}}, TensorLimits{}, ErrLimit, "tensor bytes"},
		{"product overflow", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{math.MaxInt64, 2}}, TensorLimits{MaxDimension: math.MaxInt64}, ErrInvalidTensor, "shape"},
		{"byte overflow", &onnxpb.TensorProto{DataType: proto.Int32(7), Dims: []int64{math.MaxInt64/8 + 1}}, TensorLimits{MaxDimension: math.MaxInt64}, ErrInvalidTensor, "shape"},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			testcase.value.Name = proto.String("weight")
			decoder, err := NewTensorDecoder(context.Background(), testcase.value, 0, testcase.limits)
			if err != nil {
				t.Fatal(err)
			}

			beforeLoad, beforeOwned := decoder.Usage()
			value, err := decoder.Decode(context.Background(), testcase.value)
			if value != nil || !errors.Is(err, testcase.want) || !strings.Contains(err.Error(), testcase.text) || !strings.Contains(err.Error(), `"weight"`) {
				t.Fatalf("got %v, %v; want %v containing %q and tensor name", value, err, testcase.want, testcase.text)
			}

			if load, owned := decoder.Usage(); load != beforeLoad || owned != beforeOwned {
				t.Fatalf("failed conversion changed usage: %d, %d", load, owned)
			}
		})
	}
}

func TestTensorDecoderRegistrationAndOwnership(t *testing.T) {
	initializer := &onnxpb.TensorProto{DataType: proto.Int32(7), Dims: []int64{2, 2}, Int64Data: []int64{math.MinInt64, math.MaxInt64, -1, 16777217}}
	constant := &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{1, 2, 1, 2}, FloatData: []float32{1, 2, 3, 4}}
	model := &onnxpb.ModelProto{Graph: &onnxpb.GraphProto{Initializer: []*onnxpb.TensorProto{initializer}, Node: []*onnxpb.NodeProto{{Attribute: []*onnxpb.AttributeProto{{T: constant}}}}}}
	decoder, err := NewTensorDecoder(context.Background(), model, 17, TensorLimits{})
	if err != nil {
		t.Fatal(err)
	}

	for _, wire := range []*onnxpb.TensorProto{initializer, constant} {
		value, decodeErr := decoder.Decode(context.Background(), wire)
		if decodeErr != nil || !reflect.DeepEqual(value.Shape(), wire.Dims) {
			t.Fatalf("nested value: %v, %v", value, decodeErr)
		}
	}

	// Repeated calls own disjoint destinations even for a shared initializer.
	first, err := decoder.Decode(context.Background(), initializer)
	if err != nil {
		t.Fatal(err)
	}

	second, err := decoder.Decode(context.Background(), initializer)
	if err != nil {
		t.Fatal(err)
	}

	if first == second {
		t.Fatal("decodes shared a backend tensor")
	}

	values, err := second.Int64Values()
	if err != nil || !reflect.DeepEqual(values, []int64{math.MinInt64, math.MaxInt64, -1, 16777217}) {
		t.Fatalf("exact extrema lost: %v, %v", values, err)
	}

	if load, owned := decoder.Usage(); load <= owned || owned != 112 {
		t.Fatalf("aggregate copies not charged: %d, %d", load, owned)
	}
}

func TestTensorDecoderInvalidArguments(t *testing.T) {
	wire := &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}
	var nilWire *onnxpb.TensorProto
	tests := []struct {
		name    string
		ctx     context.Context
		wire    proto.Message
		encoded int64
	}{
		{"nil context", nil, wire, 0},
		{"nil message", context.Background(), nil, 0},
		{"typed nil message", context.Background(), nilWire, 0},
		{"negative bytes", context.Background(), wire, -1},
		{"foreign protobuf", context.Background(), &emptypb.Empty{}, 0},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			value, err := NewTensorDecoder(testcase.ctx, testcase.wire, testcase.encoded, TensorLimits{})
			if value != nil || !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("got %v, %v", value, err)
			}
		})
	}

	decoder, err := NewTensorDecoder(context.Background(), wire, 0, TensorLimits{})
	if err != nil {
		t.Fatal(err)
	}

	var nilDecoder *TensorDecoder
	for _, invalid := range []*TensorDecoder{nilDecoder, {}} {
		if value, decodeErr := invalid.Decode(context.Background(), wire); value != nil || !errors.Is(decodeErr, ErrInvalidArgument) {
			t.Fatalf("invalid decoder: %v, %v", value, decodeErr)
		}
	}

	for _, invalid := range []*onnxpb.TensorProto{nil, {}} {
		if value, decodeErr := decoder.Decode(context.Background(), invalid); value != nil || !errors.Is(decodeErr, ErrInvalidArgument) {
			t.Fatalf("unregistered tensor: %v, %v", value, decodeErr)
		}
	}

	if _, err = decoder.Decode(nil, wire); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil decode context: %v", err)
	}

	if load, owned := nilDecoder.Usage(); load != 0 || owned != 0 {
		t.Fatalf("nil decoder usage: %d, %d", load, owned)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if value, decodeErr := NewTensorDecoder(ctx, wire, 0, TensorLimits{}); value != nil || !errors.Is(decodeErr, context.Canceled) {
		t.Fatalf("cancelled constructor: %v, %v", value, decodeErr)
	}

	if value, decodeErr := decoder.Decode(ctx, wire); value != nil || !errors.Is(decodeErr, context.Canceled) {
		t.Fatalf("cancelled decode: %v, %v", value, decodeErr)
	}
}

func TestTensorDecoderRejectedLiteralWire(t *testing.T) {
	tests := []struct {
		name string
		wire []byte
		want error
	}{
		{"truncated raw field", []byte{0x10, 0x01, 0x4a, 0x04, 0x00}, ErrInvalidWire},
		{"partial packed float", []byte{0x10, 0x01, 0x22, 0x01, 0x00}, ErrInvalidWire},
		{"malformed integer", []byte{0x10, 0x07, 0x38, 0x80}, ErrInvalidWire},
		{"scalar missing payload", []byte{0x10, 0x01}, ErrInvalidTensor},
		{"short raw scalar", []byte{0x10, 0x01, 0x4a, 0x03, 0, 0, 0}, ErrInvalidTensor},
		{"typed and raw scalar", []byte{0x10, 0x01, 0x25, 0, 0, 0x80, 0x3f, 0x4a, 0x04, 0, 0, 0x80, 0x3f}, ErrInvalidTensor},
		{"wrong integer count", []byte{0x08, 0x02, 0x10, 0x07, 0x38, 0x01}, ErrInvalidTensor},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			wire, err := ParseTensor(context.Background(), testcase.wire, Limits{})
			if err != nil {
				if !errors.Is(err, testcase.want) {
					t.Fatalf("parse: %v, want %v", err, testcase.want)
				}

				return
			}

			decoder, err := NewTensorDecoder(context.Background(), wire, int64(len(testcase.wire)), TensorLimits{})
			if err != nil {
				t.Fatal(err)
			}

			if value, decodeErr := decoder.Decode(context.Background(), wire); value != nil || !errors.Is(decodeErr, testcase.want) {
				t.Fatalf("decode: %v, %v; want %v", value, decodeErr, testcase.want)
			}
		})
	}
}
