package operators

import (
	"fmt"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

func inferNumerical(node graph.Node, step *Step, inputs []graph.ValueInfo) (info graph.ValueInfo, err error) {
	step.Gemm = tensor.GemmConfig{Alpha: 1, Beta: 1}
	step.Axis = -1
	for _, attribute := range node.Attributes {
		switch attribute.Name {
		case "alpha":
			step.Gemm.Alpha = attribute.Float
		case "beta":
			step.Gemm.Beta = attribute.Float
		case "axis":
			step.Axis = attribute.Int
		case "transA", "transB":
			if attribute.Int != 0 && attribute.Int != 1 {
				return info, fmt.Errorf("%w: %s must be 0 or 1", graph.ErrInvalidModel, attribute.Name)
			}

			if attribute.Name == "transA" {
				step.Gemm.TransposeA = attribute.Int == 1
			} else {
				step.Gemm.TransposeB = attribute.Int == 1
			}
		}
	}
	info.DataType = tensor.Float32
	switch node.Operator {
	case "Relu", "Sigmoid", "Tanh", "Softmax":
		info.ShapeKnown, info.Dimensions = inputs[0].ShapeKnown, inputs[0].Dimensions
		if node.Operator == "Softmax" && info.ShapeKnown {
			if _, err = tensor.NormalizeAxis(step.Axis, len(info.Dimensions)); err != nil {
				return info, fmt.Errorf("%w: Softmax axis: %w", graph.ErrInvalidModel, err)
			}
		}

		return info, nil
	case "Add", "Sub", "Mul", "Div":
		if !inputs[0].ShapeKnown || !inputs[1].ShapeKnown {
			return info, nil
		}

		info.ShapeKnown = true
		info.Dimensions, err = broadcastDimensions(inputs[0].Dimensions, inputs[1].Dimensions)
		return info, err
	case "MatMul", "Gemm":
		return inferProduct(*step, inputs)
	default:
		return info, fmt.Errorf("%w: no numerical inference handler", graph.ErrUnsupported)
	}
}

func broadcastDimensions(left, right []graph.Dimension) (dimensions []graph.Dimension, err error) {
	dimensions = make([]graph.Dimension, max(len(left), len(right)))
	for offset := 1; offset <= len(dimensions); offset++ {
		a, b := graph.Dimension{Kind: graph.FixedDimension, Value: 1}, graph.Dimension{Kind: graph.FixedDimension, Value: 1}
		if offset <= len(left) {
			a = left[len(left)-offset]
		}

		if offset <= len(right) {
			b = right[len(right)-offset]
		}

		switch {
		case a.Kind == graph.FixedDimension && a.Value == 1:
			dimensions[len(dimensions)-offset] = b
		case b.Kind == graph.FixedDimension && b.Value == 1:
			dimensions[len(dimensions)-offset] = a
		case a.Kind == graph.FixedDimension && b.Kind == graph.FixedDimension:
			if a.Value != b.Value {
				return nil, fmt.Errorf("%w: incompatible broadcast dimensions %d and %d", graph.ErrInvalidModel, a.Value, b.Value)
			}

			dimensions[len(dimensions)-offset] = a
		case a.Kind == graph.FixedDimension:
			dimensions[len(dimensions)-offset] = a
		case b.Kind == graph.FixedDimension:
			dimensions[len(dimensions)-offset] = b
		case a.Kind == graph.SymbolicDimension && b.Kind == graph.SymbolicDimension && a.Symbol == b.Symbol:
			dimensions[len(dimensions)-offset] = a
		}
	}
	return dimensions, nil
}

func inferProduct(step Step, inputs []graph.ValueInfo) (info graph.ValueInfo, err error) {
	info.DataType = tensor.Float32
	for slot := 0; slot < 2; slot++ {
		if !inputs[slot].ShapeKnown {
			continue
		}

		rank := len(inputs[slot].Dimensions)
		if rank == 0 || step.Operator == "Gemm" && rank != 2 {
			return info, fmt.Errorf("%w: %s input %d rank %d", graph.ErrInvalidModel, step.Operator, slot, rank)
		}
	}
	if step.Operator == "Gemm" && len(inputs) > 2 && inputs[2].ShapeKnown && len(inputs[2].Dimensions) > 2 {
		return info, fmt.Errorf("%w: Gemm bias rank exceeds two", graph.ErrInvalidModel)
	}

	if !inputs[0].ShapeKnown || !inputs[1].ShapeKnown {
		return info, nil
	}

	left, right := slices.Clone(inputs[0].Dimensions), slices.Clone(inputs[1].Dimensions)
	leftVector, rightVector := len(left) == 1, len(right) == 1
	if leftVector {
		left = append([]graph.Dimension{{Kind: graph.FixedDimension, Value: 1}}, left...)
	}

	if rightVector {
		right = append(right, graph.Dimension{Kind: graph.FixedDimension, Value: 1})
	}

	if step.Gemm.TransposeA {
		left[0], left[1] = left[1], left[0]
	}

	if step.Gemm.TransposeB {
		right[0], right[1] = right[1], right[0]
	}

	a, b := left[len(left)-1], right[len(right)-2]
	if a.Kind == graph.FixedDimension && b.Kind == graph.FixedDimension && a.Value != b.Value {
		return info, fmt.Errorf("%w: contraction dimensions %d and %d", graph.ErrInvalidModel, a.Value, b.Value)
	}

	if info.Dimensions, err = broadcastDimensions(left[:len(left)-2], right[:len(right)-2]); err != nil {
		return info, err
	}

	if !leftVector {
		info.Dimensions = append(info.Dimensions, left[len(left)-2])
	}

	if !rightVector {
		info.Dimensions = append(info.Dimensions, right[len(right)-1])
	}

	info.ShapeKnown = true
	if step.Operator == "Gemm" && len(inputs) > 2 && inputs[2].ShapeKnown {
		if err = validateBiasDimensions(inputs[2].Dimensions, info.Dimensions); err != nil {
			return info, err
		}
	}

	return info, nil
}

func validateBiasDimensions(bias, output []graph.Dimension) (err error) {
	if len(bias) > len(output) {
		return fmt.Errorf("%w: Gemm bias rank", graph.ErrInvalidModel)
	}

	for axis, size := range bias {
		target := output[axis+len(output)-len(bias)]
		if size.Kind == graph.FixedDimension && target.Kind == graph.FixedDimension && size.Value != 1 && size.Value != target.Value {
			return fmt.Errorf("%w: Gemm bias dimension %d cannot broadcast to %d", graph.ErrInvalidModel, size.Value, target.Value)
		}
	}
	return nil
}
