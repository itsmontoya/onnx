package decode

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
)

func TestParseModelFixture(t *testing.T) {
	var (
		data  []byte
		model *onnxpb.ModelProto
		err   error
	)

	if data, err = os.ReadFile("../../testdata/profile/identity.onnx"); err != nil {
		t.Fatal(err)
	}

	if model, err = ParseModel(context.Background(), data, Limits{MaxBytes: int64(len(data))}); err != nil {
		t.Fatal(err)
	}

	if model.GetIrVersion() != 8 || len(model.GetOpsetImport()) != 1 || model.OpsetImport[0].GetVersion() != 13 {
		t.Fatalf("unexpected model envelope: %v", model)
	}

	if graph := model.GetGraph(); graph == nil || len(graph.Node) != 1 || graph.Node[0].GetOpType() != "Identity" {
		t.Fatalf("unexpected identity graph: %v", model.GetGraph())
	}
}

func TestParseTensorLiteralWire(t *testing.T) {
	tests := []struct {
		name      string
		wire      []byte
		dataType  int32
		floatBits []uint32
		integers  []int64
		raw       []byte
	}{
		{
			name:     "packed float32",
			wire:     []byte{0x08, 0x02, 0x10, 0x01, 0x22, 0x08, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x10, 0xc0},
			dataType: 1, floatBits: []uint32{0x3fc00000, 0xc0100000},
		},
		{
			name:     "unpacked float32 special bits",
			wire:     []byte{0x08, 0x02, 0x10, 0x01, 0x25, 0x01, 0x00, 0xc0, 0x7f, 0x25, 0x00, 0x00, 0x00, 0x80},
			dataType: 1, floatBits: []uint32{0x7fc00001, 0x80000000},
		},
		{
			name: "packed exact int64",
			wire: []byte{0x0a, 0x01, 0x02, 0x10, 0x07, 0x3a, 0x0e,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x81, 0x80, 0x80, 0x08},
			dataType: 7, integers: []int64{-1, 16777217},
		},
		{
			name: "unpacked exact int64",
			wire: []byte{0x08, 0x02, 0x10, 0x07, 0x38,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01, 0x38, 0x81, 0x80, 0x80, 0x08},
			dataType: 7, integers: []int64{-1, 16777217},
		},
		{
			name:     "owned raw bytes",
			wire:     []byte{0x08, 0x02, 0x10, 0x01, 0x4a, 0x08, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x10, 0xc0},
			dataType: 1, raw: []byte{0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x10, 0xc0},
		},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			value, err := ParseTensor(context.Background(), testcase.wire, Limits{})
			if err != nil {
				t.Fatal(err)
			}

			if value.GetDataType() != testcase.dataType || !reflect.DeepEqual(value.GetDims(), []int64{2}) {
				t.Fatalf("wrong type/shape: %v", value)
			}

			var actualBits []uint32
			for _, item := range value.GetFloatData() {
				actualBits = append(actualBits, math.Float32bits(item))
			}

			if !reflect.DeepEqual(actualBits, testcase.floatBits) || !reflect.DeepEqual(value.GetInt64Data(), testcase.integers) {
				t.Fatalf("typed values: float bits %x, integers %v", actualBits, value.GetInt64Data())
			}

			clear(testcase.wire)
			if !bytes.Equal(value.GetRawData(), testcase.raw) {
				t.Fatalf("raw data borrowed or decoded incorrectly: %x", value.GetRawData())
			}
		})
	}
}

func TestUnknownFields(t *testing.T) {
	// Field 100 (varint 123) is unknown; both generated messages must retain it.
	unknown := []byte{0xa0, 0x06, 0x7b}
	model, err := ParseModel(context.Background(), append([]byte{0x08, 0x08}, unknown...), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	value, err := ParseTensor(context.Background(), append([]byte{0x10, 0x07}, unknown...), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	for _, message := range []proto.Message{model, value} {
		if !bytes.Equal(message.ProtoReflect().GetUnknown(), unknown) {
			t.Fatalf("unknown field lost from %T", message)
		}

		encoded, marshalErr := proto.Marshal(message)
		if marshalErr != nil || !bytes.HasSuffix(encoded, unknown) {
			t.Fatalf("unknown field round trip: %x, %v", encoded, marshalErr)
		}
	}
}

func TestMalformedWire(t *testing.T) {
	tests := []struct {
		name string
		data []byte
	}{
		{"truncated varint", []byte{0x08, 0x80}},
		{"truncated message", []byte{0x3a, 0x02, 0x08}},
		{"invalid field zero", []byte{0x00}},
		{"invalid wire type", []byte{0x0f}},
		{"unmatched group", []byte{0x0c}},
		{"overflow varint", []byte{0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			value, err := ParseModel(context.Background(), testcase.data, Limits{})
			if value != nil || !errors.Is(err, ErrInvalidWire) || !strings.Contains(err.Error(), "model") {
				t.Fatalf("got %v, %v; want contextual malformed wire error", value, err)
			}
		})
	}

	value, err := ParseTensor(context.Background(), []byte{0x22, 0x01, 0x00}, Limits{})
	if value != nil || !errors.Is(err, ErrInvalidWire) {
		t.Fatalf("partial packed float32: got %v, %v", value, err)
	}
}

func TestRecursionLimit(t *testing.T) {
	var (
		model   onnxpb.ModelProto
		encoded []byte
		err     error
	)

	model.Graph = &onnxpb.GraphProto{}
	graph := model.Graph
	for depth := 0; depth < 80; depth++ {
		nested := &onnxpb.GraphProto{}
		graph.Node = []*onnxpb.NodeProto{{Attribute: []*onnxpb.AttributeProto{{G: nested}}}}
		graph = nested
	}

	if encoded, err = proto.Marshal(&model); err != nil {
		t.Fatal(err)
	}

	for _, limit := range []int{0, 8} {
		if value, decodeErr := ParseModel(context.Background(), encoded, Limits{MaxRecursion: limit}); value != nil || !errors.Is(decodeErr, ErrInvalidWire) {
			t.Fatalf("recursion %d: got %v, %v", limit, value, decodeErr)
		}
	}

	if _, err = ParseModel(context.Background(), encoded, Limits{MaxRecursion: 300}); err != nil {
		t.Fatalf("explicit larger recursion limit: %v", err)
	}

	// The official decoder uses a separate fixed cap for unknown groups.
	groups := append(bytes.Repeat([]byte{0x0b}, 80), bytes.Repeat([]byte{0x0c}, 80)...)
	if _, err = ParseModel(context.Background(), groups, Limits{MaxRecursion: 1}); err != nil {
		t.Fatalf("unknown groups below the library cap: %v", err)
	}

	depth := protowire.DefaultRecursionLimit + 1
	groups = append(bytes.Repeat([]byte{0x0b}, depth), bytes.Repeat([]byte{0x0c}, depth)...)
	if _, err = ParseModel(context.Background(), groups, Limits{}); err != nil {
		t.Fatalf("unknown-group library boundary: %v", err)
	}

	depth++
	groups = append(bytes.Repeat([]byte{0x0b}, depth), bytes.Repeat([]byte{0x0c}, depth)...)
	if _, err = ParseModel(context.Background(), groups, Limits{}); !errors.Is(err, ErrInvalidWire) {
		t.Fatalf("unknown-group library recursion cap: %v", err)
	}
}

func TestReadModelBoundsAndOwnership(t *testing.T) {
	tests := []struct {
		name string
		data []byte
		max  int64
		want error
	}{
		{"exact limit", []byte{0x08, 0x08}, 2, nil},
		{"oversize before decode", []byte{0xff, 0xff, 0xff, 0xff}, 2, ErrLimit},
		{"empty wire is semantic work", nil, 2, nil},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			reader := &testReader{data: testcase.data, chunk: 1}
			value, err := ReadModel(context.Background(), reader, Limits{MaxBytes: testcase.max})
			if !errors.Is(err, testcase.want) || (err != nil && value != nil) {
				t.Fatalf("got %v, %v; want %v", value, err, testcase.want)
			}

			if reader.closed || int64(reader.readBytes) > testcase.max+1 {
				t.Fatalf("reader ownership/bound: closed %v, bytes %d", reader.closed, reader.readBytes)
			}
		})
	}

	oversized := bytes.Repeat([]byte{0xff}, 1<<20)
	if _, err := ParseModel(context.Background(), oversized, Limits{MaxBytes: 1}); !errors.Is(err, ErrLimit) {
		t.Fatalf("byte input must be checked before unmarshal: %v", err)
	}

	reader := &testReader{data: oversized}
	if _, err := ReadModel(context.Background(), reader, Limits{MaxBytes: 7}); !errors.Is(err, ErrLimit) || reader.readBytes != 8 || reader.largestRequest != 8 {
		t.Fatalf("read limit probe: bytes %d, largest request %d, error %v", reader.readBytes, reader.largestRequest, err)
	}

	tensorReader := &testReader{data: []byte{0x10, 0x07}}
	if value, err := ReadTensor(context.Background(), tensorReader, Limits{}); err != nil || value.GetDataType() != 7 || tensorReader.closed {
		t.Fatalf("tensor reader: %v, %v, closed=%v", value, err, tensorReader.closed)
	}
}

func TestReadFailuresAndCancellation(t *testing.T) {
	failure := errors.New("fixture storage failure")
	tests := []struct {
		name   string
		reader testReader
		want   error
	}{
		{"reader failure", testReader{failure: failure}, failure},
		{"data with failure", testReader{data: []byte{0x08, 0x08}, failure: failure}, failure},
		{"no progress", testReader{noProgress: true}, io.ErrNoProgress},
		{"negative count", testReader{invalidCount: -1}, ErrInvalidArgument},
		{"excess count", testReader{invalidCount: 100000}, ErrInvalidArgument},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			value, err := ReadModel(context.Background(), &testcase.reader, Limits{})
			if value != nil || !errors.Is(err, testcase.want) || !strings.Contains(err.Error(), "read model") || testcase.reader.closed {
				t.Fatalf("got %v, %v; closed=%v", value, err, testcase.reader.closed)
			}
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	reader := &testReader{data: []byte{0x08, 0x08}}
	if _, err := ReadModel(ctx, reader, Limits{}); !errors.Is(err, context.Canceled) || reader.reads != 0 {
		t.Fatalf("pre-cancelled read: %v, reads %d", err, reader.reads)
	}

	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	reader = &testReader{data: []byte{0x08, 0x08}, chunk: 1, cancel: cancel}
	if _, err := ReadModel(ctx, reader, Limits{}); !errors.Is(err, context.Canceled) || reader.reads != 1 {
		t.Fatalf("cancellation between reads: %v, reads %d", err, reader.reads)
	}

	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	reader = &testReader{data: []byte{0x08, 0x08}, cancel: cancel}
	if value, err := ReadModel(ctx, reader, Limits{}); value != nil || !errors.Is(err, context.Canceled) || reader.reads != 1 {
		t.Fatalf("cancellation on data plus EOF: %v, %v, reads %d", value, err, reader.reads)
	}
}

func TestInvalidDecoderArguments(t *testing.T) {
	tests := []struct {
		name   string
		ctx    context.Context
		reader io.Reader
		limits Limits
	}{
		{"nil context", nil, bytes.NewReader(nil), Limits{}},
		{"nil reader", context.Background(), nil, Limits{}},
		{"negative limit", context.Background(), bytes.NewReader(nil), Limits{MaxBytes: -1}},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			value, err := ReadModel(testcase.ctx, testcase.reader, testcase.limits)
			if value != nil || !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("got %v, %v; want invalid argument", value, err)
			}
		})
	}
}
