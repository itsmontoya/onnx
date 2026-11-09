package main

import (
	"fmt"
	"math"

	"github.com/itsmontoya/onnx"
)

// tensorData preserves float32 bits and exact int64 values across JSON.
type tensorData struct {
	Name     string   `json:"name"`
	DType    string   `json:"dtype"`
	Shape    []int64  `json:"shape"`
	Bits     []uint32 `json:"bits,omitempty"`
	Integers []int64  `json:"integers,omitempty"`
}

func (d tensorData) tensor() (out *onnx.Tensor, err error) {
	if d.Name == "" {
		return nil, fmt.Errorf("input name is required")
	}

	switch d.DType {
	case "float32":
		values := make([]float32, len(d.Bits))
		for index, bits := range d.Bits {
			values[index] = math.Float32frombits(bits)
		}
		if out, err = onnx.NewFloat32(d.Shape, values); err != nil {
			return nil, fmt.Errorf("input %q: %w", d.Name, err)
		}
	case "int64":
		if out, err = onnx.NewInt64(d.Shape, d.Integers); err != nil {
			return nil, fmt.Errorf("input %q: %w", d.Name, err)
		}
	default:
		return nil, fmt.Errorf("input %q: unsupported dtype %q", d.Name, d.DType)
	}
	return out, nil
}

func encodeTensor(value onnx.NamedTensor) (out tensorData, err error) {
	out.Name = value.Name
	out.Shape = append([]int64{}, value.Tensor.Shape()...)
	switch value.Tensor.DataType() {
	case onnx.Float32:
		var values []float32
		if values, err = value.Tensor.Float32Values(); err != nil {
			return out, fmt.Errorf("output %q: %w", value.Name, err)
		}

		out.DType = "float32"
		out.Bits = make([]uint32, len(values))
		for index, number := range values {
			out.Bits[index] = math.Float32bits(number)
		}
	case onnx.Int64:
		out.DType = "int64"
		if out.Integers, err = value.Tensor.Int64Values(); err != nil {
			return out, fmt.Errorf("output %q: %w", value.Name, err)
		}
	default:
		return out, fmt.Errorf("output %q: unsupported dtype %v", value.Name, value.Tensor.DataType())
	}
	return out, nil
}
