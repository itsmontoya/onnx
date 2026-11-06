package onnx

import (
	"errors"
	"reflect"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func TestTensorInvalidBackend(t *testing.T) {
	var backend tensor.Tensor
	value := Tensor{value: &backend}
	if err := value.Validate(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid backend validation: %v", err)
	}

	if value.DataType() != Invalid || value.Shape() != nil || value.Rank() != 0 || value.Len() != 0 || value.ByteSize() != 0 {
		t.Fatal("invalid backend returned valid metadata")
	}

	if values, err := value.Float32Values(); values != nil || !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid backend float accessor: %v, %v", values, err)
	}

	if values, err := value.Int64Values(); values != nil || !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid backend integer accessor: %v, %v", values, err)
	}

	if cloned, err := value.Clone(); cloned != nil || !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid backend clone: %v, %v", cloned, err)
	}
}

func TestTensorCloneStorage(t *testing.T) {
	type testcase struct {
		name     string
		dataType DataType
		shape    []int64
	}

	tests := []testcase{
		{name: "float scalar", dataType: Float32},
		{name: "int scalar", dataType: Int64},
		{name: "float empty", dataType: Float32, shape: []int64{0, 3}},
		{name: "int empty", dataType: Int64, shape: []int64{0, 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				value  *Tensor
				cloned *Tensor
				err    error
			)

			if value, err = NewTensor(tt.dataType, tt.shape); err != nil {
				t.Fatal(err)
			}

			if cloned, err = value.Clone(); err != nil {
				t.Fatal(err)
			}

			if cloned.value == value.value || cloned.DataType() != tt.dataType || !reflect.DeepEqual(cloned.Shape(), value.Shape()) {
				t.Fatal("clone does not own a separate backend value")
			}
		})
	}
}

func TestTensorClonePayloadIsolation(t *testing.T) {
	var (
		value  *Tensor
		cloned *Tensor
		addend *tensor.Tensor
		values []float32
		err    error
	)

	if value, err = NewFloat32([]int64{2}, []float32{3, 5}); err != nil {
		t.Fatal(err)
	}

	if cloned, err = value.Clone(); err != nil {
		t.Fatal(err)
	}

	if addend, err = tensor.FromFloat32([]int64{2}, []float32{1, 1}); err != nil {
		t.Fatal(err)
	}

	// Only package-internal code can supply a facade's backend as a destination.
	// A clone must remain disjoint even if later runtime code writes that backend.
	if err = tensor.AddInto(value.value, addend, cloned.value); err != nil {
		t.Fatalf("clone aliases source: %v", err)
	}

	if values, err = value.Float32Values(); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(values, []float32{3, 5}) {
		t.Fatalf("writing clone changed source: %v", values)
	}

	if values, err = cloned.Float32Values(); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(values, []float32{4, 6}) {
		t.Fatalf("clone destination: %v", values)
	}
}
