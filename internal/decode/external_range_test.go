package decode

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func externalTensor(dataType int32, shape []int64, entries ...string) (out *onnxpb.TensorProto) {
	var value onnxpb.TensorProto
	value.Name, value.DataType, value.Dims = proto.String("external_weight"), proto.Int32(dataType), shape
	value.DataLocation = onnxpb.TensorProto_EXTERNAL.Enum()
	for index := 0; index < len(entries); index += 2 {
		value.ExternalData = append(value.ExternalData, &onnxpb.StringStringEntryProto{Key: proto.String(entries[index]), Value: proto.String(entries[index+1])})
	}
	return &value
}

func TestExternalMetadataBeforeIO(t *testing.T) {
	tests := []struct {
		name    string
		entries []string
		mutate  func(*onnxpb.TensorProto)
		want    error
	}{
		{"missing location", nil, nil, ErrInvalidTensor},
		{"empty location", []string{"location", ""}, nil, ErrInvalidTensor},
		{"duplicate", []string{"location", "a", "location", "a"}, nil, ErrInvalidTensor},
		{"unknown", []string{"location", "a", "basepath", "b"}, nil, ErrUnsupported},
		{"checksum", []string{"location", "a", "checksum", "abc"}, nil, ErrUnsupported},
		{"empty offset", []string{"location", "a", "offset", ""}, nil, ErrInvalidTensor},
		{"negative offset", []string{"location", "a", "offset", "-1"}, nil, ErrInvalidTensor},
		{"plus offset", []string{"location", "a", "offset", "+1"}, nil, ErrInvalidTensor},
		{"space offset", []string{"location", "a", "offset", " 1"}, nil, ErrInvalidTensor},
		{"hex offset", []string{"location", "a", "offset", "0x1"}, nil, ErrInvalidTensor},
		{"decimal offset", []string{"location", "a", "offset", "1.0"}, nil, ErrInvalidTensor},
		{"unicode offset", []string{"location", "a", "offset", "１"}, nil, ErrInvalidTensor},
		{"offset overflow", []string{"location", "a", "offset", "9223372036854775808"}, nil, ErrInvalidTensor},
		{"range overflow", []string{"location", "a", "offset", "9223372036854775807"}, nil, ErrInvalidTensor},
		{"negative length", []string{"location", "a", "length", "-4"}, nil, ErrInvalidTensor},
		{"length overflow", []string{"location", "a", "length", "9223372036854775808"}, nil, ErrInvalidTensor},
		{"short length", []string{"location", "a", "length", "3"}, nil, ErrInvalidTensor},
		{"long length", []string{"location", "a", "length", "5"}, nil, ErrInvalidTensor},
		{"nil entry", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.ExternalData = append(value.ExternalData, nil) }, ErrInvalidTensor},
		{"raw conflict", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.RawData = []byte{} }, ErrInvalidTensor},
		{"typed conflict", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.FloatData = []float32{1} }, ErrInvalidTensor},
		{"wrong typed conflict", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.Int64Data = []int64{1} }, ErrInvalidTensor},
		{"segment", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.Segment = &onnxpb.TensorProto_Segment{} }, ErrUnsupported},
		{"dtype", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.DataType = proto.Int32(11) }, ErrUnsupported},
		{"negative shape", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.Dims = []int64{-1} }, ErrInvalidTensor},
		{"shape overflow", []string{"location", "a"}, func(value *onnxpb.TensorProto) { value.Dims = []int64{math.MaxInt64, math.MaxInt64} }, ErrLimit},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			wire := externalTensor(1, nil, testcase.entries...)
			if testcase.mutate != nil {
				testcase.mutate(wire)
			}

			resolver := &rangeResolver{}
			decoder, err := NewTensorDecoderWithResolver(context.Background(), wire, 0, TensorLimits{}, resolver)
			if err != nil {
				t.Fatal(err)
			}

			beforeLoad, beforeOwned := decoder.Usage()
			value, err := decoder.Decode(context.Background(), wire)
			if value != nil || !errors.Is(err, testcase.want) || resolver.calls != 0 || !strings.Contains(err.Error(), "external_weight") {
				t.Fatalf("conversion: %v, %v, calls %d", value, err, resolver.calls)
			}

			if load, owned := decoder.Usage(); load != beforeLoad || owned != beforeOwned {
				t.Fatalf("failed conversion usage: %d %d", load, owned)
			}
		})
	}
}

func TestExternalRangeReadLifecycle(t *testing.T) {
	readFailure := errors.New("read cause")
	resolveFailure := errors.New("resolve cause")
	closeFailure := errors.New("close cause")
	tests := []struct {
		name       string
		reader     *rangeReader
		resolveErr error
		want       error
		second     error
	}{
		{"short chunks", &rangeReader{data: []byte{1, 2, 3, 4, 99}, step: 1}, nil, nil, nil},
		{"final EOF", &rangeReader{data: []byte{1, 2, 3, 4}}, nil, nil, nil},
		{"truncated", &rangeReader{data: []byte{1, 2}}, nil, io.ErrUnexpectedEOF, nil},
		{"empty", &rangeReader{}, nil, io.ErrUnexpectedEOF, nil},
		{"read failure", &rangeReader{data: []byte{1}, failure: readFailure}, nil, readFailure, nil},
		{"EOF with cause", &rangeReader{data: []byte{1, 2, 3, 4}, failure: errors.Join(io.EOF, readFailure)}, nil, readFailure, nil},
		{"full read failure", &rangeReader{data: []byte{1, 2, 3, 4}, failure: readFailure}, nil, readFailure, nil},
		{"close failure", &rangeReader{data: []byte{1, 2, 3, 4}, closeErr: closeFailure}, nil, closeFailure, nil},
		{"read and close", &rangeReader{failure: readFailure, closeErr: closeFailure}, nil, readFailure, closeFailure},
		{"resolve and reader", &rangeReader{}, resolveFailure, resolveFailure, nil},
		{"resolve and close", &rangeReader{closeErr: closeFailure}, resolveFailure, resolveFailure, closeFailure},
		{"resolve without reader", nil, resolveFailure, resolveFailure, nil},
		{"nil reader", nil, nil, ErrInvalidArgument, nil},
		{"no progress", &rangeReader{noProgress: true}, nil, io.ErrNoProgress, nil},
		{"invalid count", &rangeReader{badCount: true}, nil, ErrInvalidArgument, nil},
	}
	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			resolver := &rangeResolver{failure: testcase.resolveErr}
			if testcase.reader != nil {
				resolver.reader = testcase.reader
			}

			rangeValue := externalRange{location: "weights.bin", offset: 7, length: 4}
			data, err := rangeValue.read(context.Background(), resolver)
			if !errors.Is(err, testcase.want) || (testcase.second != nil && !errors.Is(err, testcase.second)) {
				t.Fatalf("read error: %v", err)
			}

			if err == nil && !bytes.Equal(data, []byte{1, 2, 3, 4}) {
				t.Fatalf("bounded data: %v", data)
			}

			if err != nil && data != nil {
				t.Fatalf("published failed read: %v", data)
			}

			if resolver.location != "weights.bin" || resolver.offset != 7 || resolver.length != 4 || resolver.calls != 1 {
				t.Fatalf("resolver request: %+v", resolver)
			}

			if testcase.reader != nil && (testcase.reader.closed != 1 || testcase.reader.consumed > 4) {
				t.Fatalf("reader lifecycle: %+v", testcase.reader)
			}
		})
	}
}

func TestExternalRangeChunksAndCancellation(t *testing.T) {
	for _, length := range []int64{0, 1, 32 << 10, (32 << 10) + 1, 100000} {
		t.Run(strconv.FormatInt(length, 10), func(t *testing.T) {
			reader := &rangeReader{data: make([]byte, int(length)+1)}
			resolver := &rangeResolver{reader: reader}
			value := externalRange{location: "a", length: length}
			data, err := value.read(context.Background(), resolver)
			if err != nil || int64(len(data)) != length || reader.consumed != int(length) || reader.largest > 32<<10 || reader.closed != 1 {
				t.Fatalf("bounded read: %d, %v, %+v", len(data), err, reader)
			}

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			reader = &rangeReader{data: make([]byte, int(length)+1), cancel: cancel}
			resolver.reader = reader
			if length == 0 {
				cancel()
			}

			data, err = value.read(ctx, resolver)
			if data != nil || !errors.Is(err, context.Canceled) || reader.closed != 1 || reader.reads > 1 {
				t.Fatalf("cancelled read: %v, %v, %+v", data, err, reader)
			}
		})
	}
}
