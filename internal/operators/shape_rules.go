package operators

import (
	"context"
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

// IsShapeOperator identifies the implemented type-preserving shape handlers.
func IsShapeOperator(operator string) (shape bool) {
	switch operator {
	case "Shape", "Flatten", "Transpose", "Reshape", "Squeeze", "Unsqueeze", "Gather", "Concat":
		return true
	}
	return false
}

// ReversePermutation resolves Transpose's omitted permutation.
func ReversePermutation(rank int) (permutation []int64) {
	permutation = make([]int64, rank)
	for axis := range permutation {
		permutation[axis] = int64(rank - axis - 1)
	}
	return permutation
}

// ValidatePermutation checks ONNX's full nonnegative permutation.
func ValidatePermutation(permutation []int64, rank int) (err error) {
	if len(permutation) != rank {
		return fmt.Errorf("%w: permutation length differs from rank", graph.ErrInvalidModel)
	}

	seen := make([]bool, rank)
	for _, axis := range permutation {
		if axis < 0 || axis >= int64(rank) || seen[int(axis)] {
			return fmt.Errorf("%w: invalid or duplicate permutation axis %d", graph.ErrInvalidModel, axis)
		}

		seen[int(axis)] = true
	}
	return nil
}

// NormalizeAxes checks duplicates after normalizing against the applicable rank.
func NormalizeAxes(axes []int64, rank int) (normalized []int, err error) {
	normalized = make([]int, len(axes))
	seen := make([]bool, rank)
	for index, axis := range axes {
		if normalized[index], err = tensor.NormalizeAxis(axis, rank); err != nil {
			return nil, fmt.Errorf("%w: axes: %w", graph.ErrInvalidModel, err)
		}

		if seen[normalized[index]] {
			return nil, fmt.Errorf("%w: duplicate normalized axis", graph.ErrInvalidModel)
		}

		seen[normalized[index]] = true
	}
	return normalized, nil
}

// ResolveReshape applies the original zero-copy and inferred-dimension rules.
func ResolveReshape(input, target []int64) (shape []int64, err error) {
	return ResolveReshapeMode(input, target, false)
}

// ResolveReshapeMode also applies the schema-14 literal-zero mode.
func ResolveReshapeMode(input, target []int64, allowZero bool) (shape []int64, err error) {
	inputs := []graph.ValueInfo{{ShapeKnown: true, Dimensions: make([]graph.Dimension, len(input))}, {}}
	for axis, value := range input {
		inputs[0].Dimensions[axis] = fixed(value)
	}
	step := Step{Operator: "Reshape", AllowZero: allowZero}
	planner := newShapePlanner(context.Background(), &step, inputs, target, true)
	var info graph.ValueInfo
	if info, err = planner.reshape(graph.ValueInfo{}); err != nil {
		return nil, err
	}

	shape = make([]int64, len(info.Dimensions))
	for axis, dimension := range info.Dimensions {
		shape[axis] = dimension.Value
	}
	return shape, nil
}

func fixed(value int64) (dimension graph.Dimension) {
	return graph.Dimension{Kind: graph.FixedDimension, Value: value}
}

func dimensionProduct(dimensions []graph.Dimension) (product graph.Dimension, err error) {
	product = fixed(1)
	// A known zero determines the count even with other unknown dimensions.
	for _, dimension := range dimensions {
		if dimension.Kind == graph.FixedDimension && dimension.Value == 0 {
			return fixed(0), nil
		}
	}
	for _, dimension := range dimensions {
		if dimension.Kind != graph.FixedDimension {
			return graph.Dimension{}, nil
		}

		if dimension.Value > math.MaxInt64/product.Value {
			return product, fmt.Errorf("%w: shape product overflow", graph.ErrInvalidModel)
		}

		product.Value *= dimension.Value
	}
	return product, nil
}

func validateRawAxes(axes []int64) (err error) {
	seen := make(map[int64]bool)
	for _, axis := range axes {
		if seen[axis] {
			return fmt.Errorf("%w: duplicate axis", graph.ErrInvalidModel)
		}

		seen[axis] = true
	}
	return nil
}

// ShapeRange resolves schema-15 slicing with clamping and overflow-safe indexing.
func ShapeRange(step Step, rank int) (start, end int) {
	limit := int64(rank)
	left, right := step.ShapeStart, step.ShapeEnd
	if !step.ShapeEndSet {
		right = limit
	}

	if left < 0 {
		left += limit
	}

	if right < 0 {
		right += limit
	}

	left, right = min(max(left, 0), limit), min(max(right, 0), limit)
	return int(left), int(max(left, right))
}
