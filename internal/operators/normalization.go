package operators

import (
	"fmt"
	"math"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

func inferNormalization(node graph.Node, step *Step, inputs []graph.ValueInfo) (info graph.ValueInfo, err error) {
	step.Axis, step.Epsilon = -1, 1e-5
	for _, attribute := range node.Attributes {
		switch attribute.Name {
		case "axis":
			step.Axis = attribute.Int
		case "epsilon":
			step.Epsilon = attribute.Float
		case "stash_type":
			if attribute.Int != 1 {
				return info, fmt.Errorf("%w: LayerNormalization requires stash_type=1", graph.ErrUnsupported)
			}
		}
	}
	if math.IsNaN(float64(step.Epsilon)) || math.IsInf(float64(step.Epsilon), 0) || step.Epsilon < 0 {
		return info, fmt.Errorf("%w: LayerNormalization epsilon must be finite and nonnegative", graph.ErrUnsupported)
	}

	info = inputs[0]
	info.DataType = tensor.Float32
	if !info.ShapeKnown {
		return info, nil
	}

	var axis int
	if axis, err = tensor.NormalizeAxis(step.Axis, len(info.Dimensions)); err != nil {
		return info, fmt.Errorf("%w: LayerNormalization axis: %w", graph.ErrInvalidModel, err)
	}

	for _, dimension := range info.Dimensions[axis:] {
		if dimension.Kind == graph.FixedDimension && dimension.Value == 0 {
			return info, fmt.Errorf("%w: LayerNormalization empty normalized dimension", graph.ErrUnsupported)
		}
	}
	for slot := 1; slot < len(inputs); slot++ {
		if step.Inputs[slot] >= 0 && inputs[slot].ShapeKnown {
			if err = validateBiasDimensions(inputs[slot].Dimensions, info.Dimensions); err != nil {
				return info, fmt.Errorf("LayerNormalization input %d: %w", slot, err)
			}
		}
	}
	return info, nil
}

func normalizationStatistics(input graph.ValueInfo, axis int64) (info graph.ValueInfo) {
	info = input
	if !input.ShapeKnown {
		return info
	}

	normalized, _ := tensor.NormalizeAxis(axis, len(input.Dimensions))
	info.Dimensions = slices.Clone(input.Dimensions)
	for index := normalized; index < len(info.Dimensions); index++ {
		info.Dimensions[index] = graph.Dimension{Kind: graph.FixedDimension, Value: 1}
	}
	return info
}
