package onnx

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"
)

type conformanceComparison struct {
	Absolute   float64 `json:"absolute"`
	Relative   float64 `json:"relative"`
	SignedZero bool    `json:"signed_zero"`
}

func (c conformanceComparison) compare(role string, actual, expected NamedTensor) (err error) {
	if c.Absolute < 0 || c.Relative < 0 || math.IsNaN(c.Absolute) || math.IsNaN(c.Relative) || math.IsInf(c.Absolute, 0) || math.IsInf(c.Relative, 0) {
		return fmt.Errorf("invalid comparison tolerances")
	}

	if actual.Name != expected.Name {
		return fmt.Errorf("%s name %q, want %q", role, actual.Name, expected.Name)
	}

	if err = actual.Tensor.Validate(); err != nil {
		return fmt.Errorf("%s %q actual tensor: %w", role, expected.Name, err)
	}

	if err = expected.Tensor.Validate(); err != nil {
		return fmt.Errorf("%s %q expected tensor: %w", role, expected.Name, err)
	}

	if actual.Tensor.DataType() != expected.Tensor.DataType() {
		return fmt.Errorf("%s %q type %v, want %v", role, expected.Name, actual.Tensor.DataType(), expected.Tensor.DataType())
	}

	shape := expected.Tensor.Shape()
	if !slices.Equal(actual.Tensor.Shape(), shape) {
		return fmt.Errorf("%s %q shape %v, want %v", role, expected.Name, actual.Tensor.Shape(), shape)
	}

	if expected.Tensor.DataType() == Int64 {
		return c.compareIntegers(role, actual, expected, shape)
	}

	return c.compareFloats(role, actual, expected, shape)
}

func (c conformanceComparison) compareIntegers(role string, actual, expected NamedTensor, shape []int64) (err error) {
	var got, want []int64
	if got, err = actual.Tensor.Int64Values(); err != nil {
		return fmt.Errorf("%s %q actual integers: %w", role, expected.Name, err)
	}

	if want, err = expected.Tensor.Int64Values(); err != nil {
		return fmt.Errorf("%s %q expected integers: %w", role, expected.Name, err)
	}

	for index, value := range want {
		if got[index] != value {
			return fmt.Errorf("%s %q flat[%d] index %v = %d, want %d (exact int64)", role, expected.Name, index, conformancePosition(shape, index), got[index], value)
		}
	}

	return nil
}

func (c conformanceComparison) compareFloats(role string, actual, expected NamedTensor, shape []int64) (err error) {
	var got, want []float32
	if got, err = actual.Tensor.Float32Values(); err != nil {
		return fmt.Errorf("%s %q actual floats: %w", role, expected.Name, err)
	}

	if want, err = expected.Tensor.Float32Values(); err != nil {
		return fmt.Errorf("%s %q expected floats: %w", role, expected.Name, err)
	}

	for index, value := range want {
		if !c.equalFloat(float64(got[index]), float64(value)) {
			return fmt.Errorf("%s %q flat[%d] index %v = %v (0x%08x), want %v (0x%08x); atol=%g rtol=%g signed_zero=%t", role, expected.Name, index, conformancePosition(shape, index), got[index], math.Float32bits(got[index]), value, math.Float32bits(value), c.Absolute, c.Relative, c.SignedZero)
		}
	}

	return nil
}

func (c conformanceComparison) equalFloat(got, want float64) (equal bool) {
	if math.IsNaN(want) {
		return math.IsNaN(got)
	}

	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsInf(want, 0) {
		return got == want
	}

	if got == 0 && want == 0 && c.SignedZero {
		return math.Signbit(got) == math.Signbit(want)
	}

	return math.Abs(got-want) <= c.Absolute+c.Relative*math.Abs(want)
}

func conformancePosition(shape []int64, flat int) (position []int64) {
	position = make([]int64, len(shape))
	remainder := int64(flat)
	for axis := len(shape) - 1; axis >= 0; axis-- {
		position[axis] = remainder % shape[axis]
		remainder /= shape[axis]
	}
	return position
}

func TestConformanceComparison(t *testing.T) {
	var comparison = conformanceComparison{Absolute: 1e-5, Relative: 1e-5, SignedZero: true}
	negativeZero := math.Float32frombits(0x80000000)
	tests := []struct {
		name string
		got  float32
		want float32
		pass bool
	}{
		{"absolute tolerance", 1e-6, 0, true},
		{"relative tolerance", 100000.5, 100000, true},
		{"numeric error", 1.1, 1, false},
		{"NaN classification", math.Float32frombits(0x7fc00001), math.Float32frombits(0xffc00002), true},
		{"missing NaN", 0, float32(math.NaN()), false},
		{"unexpected NaN", float32(math.NaN()), 0, false},
		{"positive infinity", float32(math.Inf(1)), float32(math.Inf(1)), true},
		{"negative infinity", float32(math.Inf(-1)), float32(math.Inf(-1)), true},
		{"opposite infinity", float32(math.Inf(-1)), float32(math.Inf(1)), false},
		{"unexpected infinity", float32(math.Inf(1)), 0, false},
		{"missing infinity", 0, float32(math.Inf(1)), false},
		{"matching zero", negativeZero, negativeZero, true},
		{"signed zero", 0, negativeZero, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := comparison.equalFloat(float64(tt.got), float64(tt.want)); got != tt.pass {
				t.Fatalf("equalFloat(%v, %v) = %t, want %t", tt.got, tt.want, got, tt.pass)
			}
		})
	}

	comparison.SignedZero = false
	if !comparison.equalFloat(0, float64(negativeZero)) {
		t.Fatal("explicit nonempty-product zero-sign exception rejected")
	}
}

func TestConformanceComparisonDiagnostics(t *testing.T) {
	var comparison = conformanceComparison{Absolute: 1e-5, Relative: 1e-5, SignedZero: true}
	tests := []struct {
		name     string
		role     string
		actual   NamedTensor
		expected NamedTensor
		contains string
	}{
		{"float mismatch", "output", NamedTensor{"scores", floatTensor(t, []int64{2, 2}, []float32{1, 2, 99, 4})}, NamedTensor{"scores", floatTensor(t, []int64{2, 2}, []float32{1, 2, 3, 4})}, `output "scores" flat[2] index [1 0]`},
		{"integer exactness", "output", NamedTensor{"shape", intTensor(t, []int64{1}, []int64{16777216})}, NamedTensor{"shape", intTensor(t, []int64{1}, []int64{16777217})}, `output "shape" flat[0] index [0]`},
		{"input mismatch", "input", NamedTensor{"features", floatTensor(t, nil, []float32{2})}, NamedTensor{"features", floatTensor(t, nil, []float32{1})}, `input "features" flat[0] index []`},
		{"shape exactness", "output", NamedTensor{"scores", floatTensor(t, []int64{1, 2}, []float32{1, 2})}, NamedTensor{"scores", floatTensor(t, []int64{2, 1}, []float32{1, 2})}, `output "scores" shape [1 2], want [2 1]`},
		{"type exactness", "output", NamedTensor{"scores", intTensor(t, nil, []int64{1})}, NamedTensor{"scores", floatTensor(t, nil, []float32{1})}, `output "scores" type`},
		{"name exactness", "output", NamedTensor{"wrong", floatTensor(t, nil, []float32{1})}, NamedTensor{"scores", floatTensor(t, nil, []float32{1})}, `output name "wrong", want "scores"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := comparison.compare(tt.role, tt.actual, tt.expected)
			if err == nil || !strings.Contains(err.Error(), tt.contains) {
				t.Fatalf("diagnostic = %v, want %q", err, tt.contains)
			}
		})
	}
}
