package onnx_test

import (
	"errors"
	"math"
	"reflect"
	"sync"
	"testing"

	"github.com/itsmontoya/onnx"
)

func TestTensorConstruction(t *testing.T) {
	type testcase struct {
		name     string
		dataType onnx.DataType
		shape    []int64
		elements int64
		bytes    int64
	}

	tests := []testcase{
		{name: "float scalar", dataType: onnx.Float32, elements: 1, bytes: 4},
		{name: "int scalar", dataType: onnx.Int64, shape: []int64{}, elements: 1, bytes: 8},
		{name: "float empty", dataType: onnx.Float32, shape: []int64{0, 3}},
		{name: "int empty", dataType: onnx.Int64, shape: []int64{2, 0, 3}},
		{name: "matrix", dataType: onnx.Float32, shape: []int64{2, 3}, elements: 6, bytes: 24},
		{name: "rank four", dataType: onnx.Int64, shape: []int64{1, 2, 2, 3}, elements: 12, bytes: 96},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				value *onnx.Tensor
				err   error
			)

			if value, err = onnx.NewTensor(tt.dataType, tt.shape); err != nil {
				t.Fatal(err)
			}

			if err = value.Validate(); err != nil {
				t.Fatal(err)
			}

			if value.DataType() != tt.dataType || value.Rank() != len(tt.shape) || value.Len() != tt.elements || value.ByteSize() != tt.bytes {
				t.Fatalf("metadata: dtype %v, rank %d, elements %d, bytes %d", value.DataType(), value.Rank(), value.Len(), value.ByteSize())
			}

			if shape := value.Shape(); len(shape) != len(tt.shape) || (len(shape) > 0 && !reflect.DeepEqual(shape, tt.shape)) {
				t.Fatalf("shape: got %v, want %v", shape, tt.shape)
			}

			if tt.dataType == onnx.Float32 {
				var values []float32
				if values, err = value.Float32Values(); err != nil {
					t.Fatal(err)
				}

				if !reflect.DeepEqual(values, make([]float32, tt.elements)) {
					t.Fatalf("nonzero initial float values: %v", values)
				}

				if _, err = value.Int64Values(); !errors.Is(err, onnx.ErrInvalidArgument) {
					t.Fatalf("integer accessor accepted float storage: %v", err)
				}

				return
			}

			var integers []int64
			if integers, err = value.Int64Values(); err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(integers, make([]int64, tt.elements)) {
				t.Fatalf("nonzero initial integer values: %v", integers)
			}

			if _, err = value.Float32Values(); !errors.Is(err, onnx.ErrInvalidArgument) {
				t.Fatalf("float accessor accepted integer storage: %v", err)
			}
		})
	}
}

func TestTensorTypedScalarAndEmpty(t *testing.T) {
	type testcase struct {
		name     string
		shape    []int64
		floats   []float32
		integers []int64
	}

	tests := []testcase{
		{name: "scalar", floats: []float32{3.5}, integers: []int64{16777217}},
		{name: "empty", shape: []int64{2, 0, 3}, floats: []float32{}, integers: []int64{}},
		{name: "rank four", shape: []int64{1, 1, 1, 2}, floats: []float32{3, 5}, integers: []int64{16777217, -16777217}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				floats   *onnx.Tensor
				integers *onnx.Tensor
				actual   []float32
				exact    []int64
				err      error
			)

			if floats, err = onnx.NewFloat32(tt.shape, tt.floats); err != nil {
				t.Fatal(err)
			}

			if actual, err = floats.Float32Values(); err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(actual, tt.floats) || floats.Rank() != len(tt.shape) {
				t.Fatalf("float values: %v, shape %v", actual, floats.Shape())
			}

			if integers, err = onnx.NewInt64(tt.shape, tt.integers); err != nil {
				t.Fatal(err)
			}

			if exact, err = integers.Int64Values(); err != nil {
				t.Fatal(err)
			}

			if !reflect.DeepEqual(exact, tt.integers) || integers.Rank() != len(tt.shape) {
				t.Fatalf("integer values: %v, shape %v", exact, integers.Shape())
			}
		})
	}
}

func TestTensorTypedValues(t *testing.T) {
	var (
		floats   *onnx.Tensor
		integers *onnx.Tensor
		values   []float32
		actual   []int64
		err      error
	)

	bits := []uint32{0x80000000, 0x7fc12345, 0x7f800000, 0xff800000, 0x3f800000, 0xbf800000}
	input := make([]float32, len(bits))
	for index, pattern := range bits {
		input[index] = math.Float32frombits(pattern)
	}

	if floats, err = onnx.NewFloat32([]int64{2, 3}, input); err != nil {
		t.Fatal(err)
	}

	if values, err = floats.Float32Values(); err != nil {
		t.Fatal(err)
	}

	for index, pattern := range bits {
		if math.Float32bits(values[index]) != pattern {
			t.Fatalf("float[%d]: got %08x, want %08x", index, math.Float32bits(values[index]), pattern)
		}
	}

	expected := []int64{16777217, -16777217, math.MaxInt64, math.MinInt64}
	if integers, err = onnx.NewInt64([]int64{4}, expected); err != nil {
		t.Fatal(err)
	}

	if actual, err = integers.Int64Values(); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("integer values: got %v, want %v", actual, expected)
	}

	if _, err = floats.Int64Values(); !errors.Is(err, onnx.ErrInvalidArgument) {
		t.Fatalf("integer access to float tensor: %v", err)
	}

	if _, err = integers.Float32Values(); !errors.Is(err, onnx.ErrInvalidArgument) {
		t.Fatalf("float access to integer tensor: %v", err)
	}
}

func TestTensorOwnership(t *testing.T) {
	var (
		value   *onnx.Tensor
		cloned  *onnx.Tensor
		values  []float32
		copyOut []float32
		err     error
	)

	shape := []int64{1, 2}
	input := []float32{3, 5}
	if value, err = onnx.NewFloat32(shape, input); err != nil {
		t.Fatal(err)
	}

	shape[0] = 9
	input[0] = 99
	if !reflect.DeepEqual(value.Shape(), []int64{1, 2}) {
		t.Fatalf("constructor borrowed shape: %v", value.Shape())
	}

	if values, err = value.Float32Values(); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(values, []float32{3, 5}) {
		t.Fatalf("constructor borrowed values: %v", values)
	}

	values[0] = 42
	shape = value.Shape()
	shape[0] = 8
	if copyOut, err = value.Float32Values(); err != nil {
		t.Fatal(err)
	}

	if copyOut[0] != 3 || value.Shape()[0] != 1 {
		t.Fatalf("accessor borrowed storage: shape %v, values %v", value.Shape(), copyOut)
	}

	if cloned, err = value.Clone(); err != nil {
		t.Fatal(err)
	}

	if copyOut, err = cloned.Float32Values(); err != nil {
		t.Fatal(err)
	}

	if cloned == value || !reflect.DeepEqual(copyOut, []float32{3, 5}) || !reflect.DeepEqual(cloned.Shape(), []int64{1, 2}) {
		t.Fatalf("clone: shape %v, values %v", cloned.Shape(), copyOut)
	}
}

func TestTensorIntegerOwnership(t *testing.T) {
	var (
		value   *onnx.Tensor
		values  []int64
		copyOut []int64
		err     error
	)

	shape := []int64{2}
	input := []int64{16777217, math.MinInt64}
	if value, err = onnx.NewInt64(shape, input); err != nil {
		t.Fatal(err)
	}

	input[0] = 0
	shape[0] = 0
	if values, err = value.Int64Values(); err != nil {
		t.Fatal(err)
	}

	values[0] = 0
	if copyOut, err = value.Int64Values(); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(copyOut, []int64{16777217, math.MinInt64}) || !reflect.DeepEqual(value.Shape(), []int64{2}) {
		t.Fatalf("integer storage borrowed: shape %v, values %v", value.Shape(), copyOut)
	}
}

func TestTensorInvalidConstruction(t *testing.T) {
	type testcase struct {
		name  string
		build func() (*onnx.Tensor, error)
	}

	tests := []testcase{
		{name: "invalid dtype", build: func() (*onnx.Tensor, error) { return onnx.NewTensor(onnx.Invalid, nil) }},
		{name: "unsupported dtype", build: func() (*onnx.Tensor, error) { return onnx.NewTensor(onnx.DataType(255), nil) }},
		{name: "negative dimension", build: func() (*onnx.Tensor, error) { return onnx.NewTensor(onnx.Float32, []int64{-1}) }},
		{name: "empty negative dimension", build: func() (*onnx.Tensor, error) { return onnx.NewTensor(onnx.Int64, []int64{0, -1}) }},
		{name: "element overflow", build: func() (*onnx.Tensor, error) { return onnx.NewTensor(onnx.Float32, []int64{math.MaxInt64, 2}) }},
		{name: "byte overflow", build: func() (*onnx.Tensor, error) { return onnx.NewTensor(onnx.Int64, []int64{math.MaxInt64/8 + 1}) }},
		{name: "missing scalar float", build: func() (*onnx.Tensor, error) { return onnx.NewFloat32(nil, nil) }},
		{name: "extra empty float", build: func() (*onnx.Tensor, error) { return onnx.NewFloat32([]int64{0}, []float32{1}) }},
		{name: "short float payload", build: func() (*onnx.Tensor, error) { return onnx.NewFloat32([]int64{2}, []float32{1}) }},
		{name: "short integer payload", build: func() (*onnx.Tensor, error) { return onnx.NewInt64([]int64{2}, []int64{1}) }},
		{name: "missing scalar integer", build: func() (*onnx.Tensor, error) { return onnx.NewInt64(nil, nil) }},
		{name: "negative typed float", build: func() (*onnx.Tensor, error) { return onnx.NewFloat32([]int64{-1}, nil) }},
		{name: "negative typed integer", build: func() (*onnx.Tensor, error) { return onnx.NewInt64([]int64{-1}, nil) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			value, err := tt.build()
			if value != nil || !errors.Is(err, onnx.ErrInvalidArgument) {
				t.Fatalf("got value %v, error %v", value, err)
			}
		})
	}
}

func TestTensorInvalidReceivers(t *testing.T) {
	var zero onnx.Tensor
	tests := []*onnx.Tensor{nil, &zero}
	for index, value := range tests {
		if err := value.Validate(); err != onnx.ErrInvalidArgument {
			t.Fatalf("receiver %d Validate: %v", index, err)
		}

		if value.DataType() != onnx.Invalid || value.Shape() != nil || value.Rank() != 0 || value.Len() != 0 || value.ByteSize() != 0 {
			t.Fatalf("receiver %d returned valid metadata", index)
		}

		if values, err := value.Float32Values(); values != nil || err != onnx.ErrInvalidArgument {
			t.Fatalf("receiver %d float access: %v, %v", index, values, err)
		}

		if values, err := value.Int64Values(); values != nil || err != onnx.ErrInvalidArgument {
			t.Fatalf("receiver %d integer access: %v, %v", index, values, err)
		}

		if cloned, err := value.Clone(); cloned != nil || err != onnx.ErrInvalidArgument {
			t.Fatalf("receiver %d clone: %v, %v", index, cloned, err)
		}
	}
}

func TestTensorConcurrentReaders(t *testing.T) {
	var (
		value *onnx.Tensor
		err   error
		group sync.WaitGroup
	)

	if value, err = onnx.NewInt64([]int64{2}, []int64{16777217, math.MinInt64}); err != nil {
		t.Fatal(err)
	}

	for range 8 {
		group.Go(func() {
			for range 20 {
				values, readErr := value.Int64Values()
				if readErr != nil || !reflect.DeepEqual(values, []int64{16777217, math.MinInt64}) || value.Shape()[0] != 2 {
					t.Errorf("shared reader: values %v, error %v", values, readErr)
					return
				}

				values[0] = 0
				shape := value.Shape()
				shape[0] = 0
			}
		})
	}

	group.Wait()
}
