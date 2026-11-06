package decode

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestTensorLimits(t *testing.T) {
	limits, err := (TensorLimits{}).normalized()
	if err != nil || limits.MaxLoadBytes != 1<<30 || limits.MaxInitializerBytes != 256<<20 || limits.MaxTensorBytes != 64<<20 || limits.MaxRank != 32 || limits.MaxDimension != 1<<31-1 || limits.MaxRecursion != 64 {
		t.Fatalf("defaults: %+v, %v", limits, err)
	}

	tests := []TensorLimits{
		{MaxLoadBytes: -1}, {MaxInitializerBytes: -1}, {MaxTensorBytes: -1},
		{MaxRank: -1}, {MaxDimension: -1}, {MaxRecursion: -1},
	}

	for _, invalid := range tests {
		if _, err = invalid.normalized(); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("invalid limits %+v: %v", invalid, err)
		}
	}

	raised := TensorLimits{MaxLoadBytes: math.MaxInt64, MaxInitializerBytes: math.MaxInt64, MaxTensorBytes: math.MaxInt64, MaxRank: 128, MaxDimension: math.MaxInt64, MaxRecursion: 128}
	if limits, err = raised.normalized(); err != nil || limits != raised {
		t.Fatalf("raised limits: %+v, %v", limits, err)
	}
}

func TestTensorDecoderAccounting(t *testing.T) {
	// FLOAT's type takes 4 decoded bytes; its scalar payload takes 4 more.
	// Typed: 8 encoded + 8 decoded + 4 owned = 20 bytes.
	// Raw: the same amounts plus 4 conversion staging bytes = 24 bytes.
	tests := []struct {
		name    string
		value   *onnxpb.TensorProto
		maximum int64
		want    error
		usage   int64
	}{
		{"typed exact", &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}, 20, nil, 20},
		{"typed owned over", &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}, 19, ErrLimit, 16},
		{"raw exact", &onnxpb.TensorProto{DataType: proto.Int32(1), RawData: []byte{0, 0, 128, 63}}, 24, nil, 24},
		{"raw staging over", &onnxpb.TensorProto{DataType: proto.Int32(1), RawData: []byte{0, 0, 128, 63}}, 23, ErrLimit, 16},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			decoder, err := NewTensorDecoder(context.Background(), testcase.value, 8, TensorLimits{MaxLoadBytes: testcase.maximum})
			if err != nil {
				t.Fatal(err)
			}

			value, err := decoder.Decode(context.Background(), testcase.value)
			if !errors.Is(err, testcase.want) || (err != nil && value != nil) {
				t.Fatalf("conversion: %v, %v", value, err)
			}

			load, owned := decoder.Usage()
			if load != testcase.usage || (err == nil && owned != 4) || (err != nil && owned != 0) {
				t.Fatalf("usage %d, %d", load, owned)
			}
		})
	}
}

func TestTensorDecoderAggregateLimits(t *testing.T) {
	first := &onnxpb.TensorProto{DataType: proto.Int32(7), Int64Data: []int64{1}}
	second := &onnxpb.TensorProto{DataType: proto.Int32(7), Int64Data: []int64{2}}
	model := &onnxpb.ModelProto{Graph: &onnxpb.GraphProto{Initializer: []*onnxpb.TensorProto{first, second}}}
	tests := []struct {
		name   string
		limits TensorLimits
	}{
		{"initializer aggregate", TensorLimits{MaxInitializerBytes: 8}},
		{"load aggregate", TensorLimits{MaxLoadBytes: 39}},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			// Both decoded tensors coexist: 24 decoded bytes before any copy.
			decoder, err := NewTensorDecoder(context.Background(), model, 0, testcase.limits)
			if err != nil {
				t.Fatal(err)
			}

			if load, owned := decoder.Usage(); load != 24 || owned != 0 {
				t.Fatalf("initial usage %d, %d", load, owned)
			}

			if _, err = decoder.Decode(context.Background(), first); err != nil {
				t.Fatal(err)
			}

			if value, decodeErr := decoder.Decode(context.Background(), second); value != nil || !errors.Is(decodeErr, ErrLimit) {
				t.Fatalf("aggregate limit: %v, %v", value, decodeErr)
			}

			if load, owned := decoder.Usage(); load != 32 || owned != 8 {
				t.Fatalf("failed allocation changed usage %d, %d", load, owned)
			}
		})
	}

	for _, maximum := range []int64{7, 15} {
		if value, err := NewTensorDecoder(context.Background(), first, 8, TensorLimits{MaxLoadBytes: maximum}); value != nil || !errors.Is(err, ErrLimit) {
			t.Fatalf("encoded/decoded limit %d: %v, %v", maximum, value, err)
		}
	}

	// Checked remaining-budget arithmetic must work at the int64 boundary.
	decoder, err := NewTensorDecoder(context.Background(), first, math.MaxInt64-12, TensorLimits{MaxLoadBytes: math.MaxInt64})
	if err != nil {
		t.Fatal(err)
	}

	if value, decodeErr := decoder.Decode(context.Background(), first); value != nil || !errors.Is(decodeErr, ErrLimit) {
		t.Fatalf("load sum overflow: %v, %v", value, decodeErr)
	}

	for _, size := range []int64{7, 8} {
		decoder, err = NewTensorDecoder(context.Background(), first, 0, TensorLimits{MaxTensorBytes: size, MaxInitializerBytes: size})
		if err != nil {
			t.Fatal(err)
		}

		value, decodeErr := decoder.Decode(context.Background(), first)
		if (size == 7 && (value != nil || !errors.Is(decodeErr, ErrLimit))) || (size == 8 && decodeErr != nil) {
			t.Fatalf("tensor boundary %d: %v, %v", size, value, decodeErr)
		}
	}
}
