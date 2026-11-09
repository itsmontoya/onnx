package onnx_test

import (
	"bytes"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/wrapperspb"
)

// TestDependencyBaseline verifies that the pinned storage and protobuf paths
// work together under each build configuration before runtime adapters exist.
func TestDependencyBaseline(t *testing.T) {
	var (
		value   *tensor.Tensor
		values  []int64
		encoded []byte
		decoded wrapperspb.Int64Value
		err     error
	)

	if value, err = tensor.FromInt64(nil, []int64{16777217}); err != nil {
		t.Fatal(err)
	}

	if values, err = value.Int64Values(); err != nil {
		t.Fatal(err)
	}

	if value.Rank() != 0 || len(values) != 1 || values[0] != 16777217 {
		t.Fatalf("scalar storage: shape %v, values %v", value.Shape(), values)
	}

	if encoded, err = proto.Marshal(wrapperspb.Int64(values[0])); err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(encoded, []byte{0x08, 0x81, 0x80, 0x80, 0x08}) {
		t.Fatalf("protobuf encoding: got %x", encoded)
	}

	if err = proto.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Value != values[0] {
		t.Fatalf("protobuf decoding: got %d, want %d", decoded.Value, values[0])
	}
}
