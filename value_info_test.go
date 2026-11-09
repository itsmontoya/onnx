package onnx_test

import (
	"fmt"

	"github.com/itsmontoya/onnx"
)

func ExampleValueInfo() {
	unknownRank := onnx.ValueInfo{Name: "unknown", DataType: onnx.Float32}
	scalar := onnx.ValueInfo{Name: "scalar", DataType: onnx.Float32, ShapeKnown: true}
	empty := onnx.ValueInfo{
		Name:       "empty",
		DataType:   onnx.Float32,
		ShapeKnown: true,
		Dimensions: []onnx.Dimension{{Kind: onnx.FixedDimension, Value: 0}},
	}
	batch := onnx.ValueInfo{
		Name:       "batch",
		DataType:   onnx.Float32,
		ShapeKnown: true,
		Dimensions: []onnx.Dimension{{Kind: onnx.SymbolicDimension, Symbol: "N"}, {Kind: onnx.FixedDimension, Value: 3}},
	}
	unknownBatch := onnx.ValueInfo{
		Name:       "unknown_batch",
		DataType:   onnx.Float32,
		ShapeKnown: true,
		Dimensions: []onnx.Dimension{{Kind: onnx.UnknownDimension}, {Kind: onnx.FixedDimension, Value: 3}},
		Defaulted:  true,
	}

	fmt.Println(unknownRank.ShapeKnown, scalar.ShapeKnown, len(scalar.Dimensions))
	fmt.Println(empty.Dimensions[0].Kind == onnx.FixedDimension, empty.Dimensions[0].Value)
	fmt.Println(batch.Dimensions[0].Kind == onnx.SymbolicDimension, batch.Dimensions[0].Symbol)
	fmt.Println(unknownBatch.Dimensions[0].Kind == onnx.UnknownDimension, unknownBatch.Defaulted)
	// Output:
	// false true 0
	// true 0
	// true N
	// true true
}
