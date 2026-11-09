package operators

import (
	"fmt"
	"math"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

func inferCNN(node graph.Node, step *Step, inputs []graph.ValueInfo) (info graph.ValueInfo, err error) {
	info.DataType = tensor.Float32
	if err = step.CNN.parse(node); err != nil {
		return info, err
	}

	if node.Operator == "MaxPool" && len(step.Outputs) > 1 && step.Outputs[1] >= 0 {
		return info, fmt.Errorf("%w: MaxPool indices output", graph.ErrUnsupported)
	}

	for slot, input := range inputs {
		if !input.ShapeKnown {
			continue
		}

		rank := len(input.Dimensions)
		if slot == 0 && (node.Operator == "BatchNormalization" && rank != 2 && rank != 4 || node.Operator != "BatchNormalization" && rank != 4) {
			return info, fmt.Errorf("%w: %s input rank %d", graph.ErrUnsupported, node.Operator, rank)
		}

		if slot > 0 && step.Inputs[slot] >= 0 && (node.Operator == "BatchNormalization" && rank != 1 || node.Operator == "Conv" && (slot == 1 && rank != 4 || slot == 2 && rank != 1)) {
			return info, fmt.Errorf("%w: %s parameter %d rank %d", graph.ErrInvalidModel, node.Operator, slot, rank)
		}
	}
	for axis := 0; axis < 2; axis++ {
		kernel := step.CNN.Pool.Kernel[axis]
		if node.Operator == "Conv" && !step.CNN.KernelSet && inputs[1].ShapeKnown && inputs[1].Dimensions[axis+2].Kind == graph.FixedDimension {
			kernel = inputs[1].Dimensions[axis+2].Value
		}

		if kernel > 0 && kernel-1 > (math.MaxInt64-1)/step.CNN.Conv.Dilations[axis] {
			return info, fmt.Errorf("%w: effective kernel overflow", graph.ErrInvalidModel)
		}
	}
	if err = validateCNNDimensions(node.Operator, step.CNN, inputs); err != nil {
		return info, err
	}

	if !inputs[0].ShapeKnown {
		return info, nil
	}

	info.ShapeKnown, info.Dimensions = true, slices.Clone(inputs[0].Dimensions)
	if node.Operator == "BatchNormalization" {
		return info, nil
	}

	if node.Operator == "GlobalAveragePool" {
		info.Dimensions[2], info.Dimensions[3] = fixedCNN(1), fixedCNN(1)
		return info, nil
	}

	kernel := [2]graph.Dimension{fixedCNN(step.CNN.Pool.Kernel[0]), fixedCNN(step.CNN.Pool.Kernel[1])}
	if node.Operator == "Conv" {
		info.Dimensions[1] = graph.Dimension{}
		kernel = [2]graph.Dimension{}
		if inputs[1].ShapeKnown {
			info.Dimensions[1] = inputs[1].Dimensions[0]
			kernel = [2]graph.Dimension{inputs[1].Dimensions[2], inputs[1].Dimensions[3]}
		} else if step.CNN.KernelSet {
			kernel = [2]graph.Dimension{fixedCNN(step.CNN.Pool.Kernel[0]), fixedCNN(step.CNN.Pool.Kernel[1])}
		}
	}

	for axis := 0; axis < 2; axis++ {
		input := inputs[0].Dimensions[axis+2]
		info.Dimensions[axis+2] = graph.Dimension{}
		if input.Kind != graph.FixedDimension || kernel[axis].Kind != graph.FixedDimension {
			continue
		}

		config := step.CNN
		config.Conv.Groups = 1
		config.Conv.Strides = [2]int64{step.CNN.Conv.Strides[axis], 1}
		config.Conv.Dilations = [2]int64{step.CNN.Conv.Dilations[axis], 1}
		config.Conv.Pads = [4]int64{step.CNN.Conv.Pads[axis], 0, step.CNN.Conv.Pads[axis+2], 0}
		config.Pool.Strides, config.Pool.Dilations, config.Pool.Pads = config.Conv.Strides, config.Conv.Dilations, config.Conv.Pads
		config.Pool.Kernel, config.KernelSet = [2]int64{kernel[axis].Value, 1}, true
		shapes := [][]int64{{1, 1, input.Value, 1}, {1, 1, kernel[axis].Value, 1}}
		var output []int64
		if output, _, _, err = config.Shape(node.Operator, shapes); err != nil {
			return info, fmt.Errorf("%w: CNN spatial geometry: %w", graph.ErrInvalidModel, err)
		}

		info.Dimensions[axis+2] = fixedCNN(output[2])
	}
	if node.Operator == "AveragePool" && step.CNN.Pool.Kernel[0] > math.MaxInt64/step.CNN.Pool.Kernel[1] && concrete(info) {
		nonempty := true
		for _, dimension := range info.Dimensions {
			nonempty = nonempty && dimension.Value != 0
		}
		if nonempty {
			return info, fmt.Errorf("%w: average kernel area overflow", graph.ErrInvalidModel)
		}
	}

	return info, nil
}

func validateCNNDimensions(operator string, config CNNConfig, inputs []graph.ValueInfo) (err error) {
	if operator == "Conv" {
		if inputs[0].ShapeKnown {
			channels := inputs[0].Dimensions[1]
			if channels.Kind == graph.FixedDimension && (channels.Value <= 0 || channels.Value%config.Conv.Groups != 0) {
				return fmt.Errorf("%w: Conv input channels/group", graph.ErrInvalidModel)
			}
		}

		if !inputs[1].ShapeKnown {
			return nil
		}

		weights := inputs[1].Dimensions
		for axis, dimension := range weights {
			if dimension.Kind == graph.FixedDimension && (dimension.Value <= 0 || axis == 0 && dimension.Value%config.Conv.Groups != 0) {
				return fmt.Errorf("%w: Conv weights/group", graph.ErrInvalidModel)
			}
		}
		for axis := 0; axis < 2; axis++ {
			if config.KernelSet && weights[axis+2].Kind == graph.FixedDimension && weights[axis+2].Value != config.Pool.Kernel[axis] {
				return fmt.Errorf("%w: kernel_shape differs from weights", graph.ErrInvalidModel)
			}
		}
		if inputs[0].ShapeKnown && inputs[0].Dimensions[1].Kind == graph.FixedDimension && weights[1].Kind == graph.FixedDimension && inputs[0].Dimensions[1].Value/config.Conv.Groups != weights[1].Value {
			return fmt.Errorf("%w: Conv weights channels", graph.ErrInvalidModel)
		}

		if len(inputs) > 2 && inputs[2].ShapeKnown && !equalCNNDimension(inputs[2].Dimensions[0], weights[0]) {
			return fmt.Errorf("%w: Conv bias must be [M]", graph.ErrInvalidModel)
		}

		return nil
	}

	if operator == "BatchNormalization" {
		var channels graph.Dimension
		if inputs[0].ShapeKnown {
			channels = inputs[0].Dimensions[1]
		}

		for _, input := range inputs[1:] {
			if !input.ShapeKnown {
				continue
			}

			if !equalCNNDimension(channels, input.Dimensions[0]) {
				return fmt.Errorf("%w: BatchNormalization parameters must be [C]", graph.ErrInvalidModel)
			}

			if channels.Kind != graph.FixedDimension {
				channels = input.Dimensions[0]
			}
		}
	}

	return nil
}

func fixedCNN(value int64) (dimension graph.Dimension) {
	return graph.Dimension{Kind: graph.FixedDimension, Value: value}
}

func equalCNNDimension(left, right graph.Dimension) (equal bool) {
	return left.Kind != graph.FixedDimension || right.Kind != graph.FixedDimension || left.Value == right.Value
}

// IsCNNOperator reports the bounded RT-13 family.
func IsCNNOperator(operator string) (supported bool) {
	switch operator {
	case "Conv", "MaxPool", "AveragePool", "GlobalAveragePool", "BatchNormalization":
		return true
	}
	return false
}
