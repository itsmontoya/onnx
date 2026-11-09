package operators

import (
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/graph"
)

// CNNConfig retains schema defaults and fixed-size spatial attributes. Automatic
// padding is resolved separately for each concrete invocation, including weights.
type CNNConfig struct {
	Conv      tensor.Conv2DConfig
	Pool      tensor.Pool2DConfig
	AutoPad   string
	KernelSet bool
	Epsilon   float32
	Momentum  float32
}

func (c *CNNConfig) parse(node graph.Node) (err error) {
	c.Conv = tensor.Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
	c.Pool = tensor.Pool2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}
	c.AutoPad, c.Epsilon, c.Momentum = "NOTSET", 1e-5, 0.9
	var padsSet bool
	for _, attribute := range node.Attributes {
		if err = c.attribute(attribute); err != nil {
			return err
		}

		padsSet = padsSet || attribute.Name == "pads"
	}

	if c.AutoPad != "NOTSET" && c.AutoPad != "VALID" && c.AutoPad != "SAME_UPPER" && c.AutoPad != "SAME_LOWER" {
		return fmt.Errorf("%w: auto_pad %q", graph.ErrInvalidModel, c.AutoPad)
	}

	for axis := 0; axis < 2; axis++ {
		if c.Conv.Pads[axis] > math.MaxInt64-c.Conv.Pads[axis+2] {
			return fmt.Errorf("%w: padding extent overflow", graph.ErrInvalidModel)
		}
	}
	if padsSet && c.AutoPad != "NOTSET" {
		return fmt.Errorf("%w: pads conflicts with auto_pad", graph.ErrInvalidModel)
	}

	if c.Conv.Groups <= 0 {
		return fmt.Errorf("%w: group must be positive", graph.ErrInvalidModel)
	}

	if c.Epsilon < 0 || math.IsNaN(float64(c.Epsilon)) || math.IsInf(float64(c.Epsilon), 0) {
		return fmt.Errorf("%w: epsilon must be finite and nonnegative", graph.ErrInvalidModel)
	}

	if c.Momentum < 0 || c.Momentum > 1 || math.IsNaN(float64(c.Momentum)) {
		return fmt.Errorf("%w: momentum must be finite in [0,1]", graph.ErrInvalidModel)
	}

	return nil
}

func (c *CNNConfig) attribute(attribute graph.Attribute) (err error) {
	switch attribute.Name {
	case "auto_pad":
		c.AutoPad = string(attribute.String)
	case "group":
		c.Conv.Groups = attribute.Int
	case "kernel_shape", "strides", "dilations":
		if len(attribute.Ints) != 2 || attribute.Ints[0] <= 0 || attribute.Ints[1] <= 0 {
			return fmt.Errorf("%w: %s requires two positive values", graph.ErrInvalidModel, attribute.Name)
		}

		pair := [2]int64{attribute.Ints[0], attribute.Ints[1]}
		switch attribute.Name {
		case "kernel_shape":
			c.Pool.Kernel, c.KernelSet = pair, true
		case "strides":
			c.Conv.Strides, c.Pool.Strides = pair, pair
		case "dilations":
			c.Conv.Dilations, c.Pool.Dilations = pair, pair
		}
	case "pads":
		if len(attribute.Ints) != 4 {
			return fmt.Errorf("%w: pads requires four values", graph.ErrInvalidModel)
		}

		for axis, value := range attribute.Ints {
			if value < 0 {
				return fmt.Errorf("%w: pads must be nonnegative", graph.ErrInvalidModel)
			}

			c.Conv.Pads[axis], c.Pool.Pads[axis] = value, value
		}
	case "ceil_mode", "count_include_pad":
		if attribute.Int != 0 && attribute.Int != 1 {
			return fmt.Errorf("%w: %s must be 0 or 1", graph.ErrInvalidModel, attribute.Name)
		}

		if attribute.Name == "ceil_mode" {
			c.Pool.CeilMode = attribute.Int == 1
		} else {
			c.Pool.CountIncludePad = attribute.Int == 1
		}
	case "storage_order":
		if attribute.Int != 0 {
			return fmt.Errorf("%w: MaxPool storage_order must be 0", graph.ErrUnsupported)
		}
	case "training_mode":
		if attribute.Int != 0 {
			return fmt.Errorf("%w: BatchNormalization training_mode must be 0", graph.ErrUnsupported)
		}
	case "epsilon":
		c.Epsilon = attribute.Float
	case "momentum":
		c.Momentum = attribute.Float
	}
	return nil
}

// Pads resolves SAME with checked integer arithmetic; no payload is inspected.
func (c CNNConfig) Pads(input, kernel, strides, dilations [2]int64) (pads [4]int64, err error) {
	if c.AutoPad == "NOTSET" {
		return c.Conv.Pads, nil
	}

	if c.AutoPad == "VALID" {
		return pads, nil
	}

	for axis, size := range input {
		if size < 0 || kernel[axis] <= 0 || strides[axis] <= 0 || dilations[axis] <= 0 {
			return pads, fmt.Errorf("%w: invalid automatic padding geometry", graph.ErrInvalidModel)
		}

		if kernel[axis]-1 > (math.MaxInt64-1)/dilations[axis] {
			return pads, fmt.Errorf("%w: effective kernel overflow", graph.ErrInvalidModel)
		}

		effective := (kernel[axis]-1)*dilations[axis] + 1
		// ceil(size/stride) avoids size+stride-1 overflow. Empty inputs
		// have zero windows and need no automatic padding.
		if size == 0 {
			continue
		}

		last := (size - 1) / strides[axis]
		if last > (math.MaxInt64-effective)/strides[axis] {
			return pads, fmt.Errorf("%w: automatic padding extent overflow", graph.ErrInvalidModel)
		}

		total := max(int64(0), last*strides[axis]+effective-size)
		pads[axis], pads[axis+2] = total/2, total-total/2
		if c.AutoPad == "SAME_LOWER" {
			pads[axis], pads[axis+2] = pads[axis+2], pads[axis]
		}
	}
	return pads, nil
}

// Shape validates concrete CNN metadata and returns configs for direct kernels.
// The caller reserves the returned destination before numerical allocation.
func (c CNNConfig) Shape(operator string, shapes [][]int64) (shape []int64, conv tensor.Conv2DConfig, pool tensor.Pool2DConfig, err error) {
	conv, pool = c.Conv, c.Pool
	left := shapes[0]
	if operator == "BatchNormalization" {
		if len(left) != 2 && len(left) != 4 {
			return nil, conv, pool, fmt.Errorf("%w: BatchNormalization requires rank two or four", graph.ErrUnsupported)
		}

		for _, parameter := range shapes[1:] {
			if len(parameter) != 1 || parameter[0] != left[1] {
				return nil, conv, pool, fmt.Errorf("%w: BatchNormalization parameters must be [C]", graph.ErrInvalidModel)
			}
		}
		return left, conv, pool, nil
	}

	if len(left) != 4 {
		return nil, conv, pool, fmt.Errorf("%w: CNN input must be NCHW rank four", graph.ErrUnsupported)
	}

	if operator == "GlobalAveragePool" {
		shape, err = tensor.GlobalAveragePoolShape(left)
		return shape, conv, pool, err
	}

	kernel := pool.Kernel
	if operator == "Conv" {
		weights := shapes[1]
		if len(weights) != 4 {
			return nil, conv, pool, fmt.Errorf("%w: Conv weights require rank four", graph.ErrInvalidModel)
		}

		kernel = [2]int64{weights[2], weights[3]}
		if c.KernelSet && kernel != pool.Kernel {
			return nil, conv, pool, fmt.Errorf("%w: kernel_shape differs from weights", graph.ErrInvalidModel)
		}

		if len(shapes) > 2 && shapes[2] != nil && (len(shapes[2]) != 1 || shapes[2][0] != weights[0]) {
			return nil, conv, pool, fmt.Errorf("%w: Conv bias must be [M]", graph.ErrInvalidModel)
		}
	}

	if conv.Pads, err = c.Pads([2]int64{left[2], left[3]}, kernel, conv.Strides, conv.Dilations); err != nil {
		return nil, conv, pool, err
	}

	pool.Pads = conv.Pads
	if operator == "Conv" {
		shape, err = tensor.Conv2DShape(left, shapes[1], conv)
	} else {
		shape, err = tensor.Pool2DShape(left, pool)
		if err == nil && operator == "AveragePool" && shape[0] != 0 && shape[1] != 0 && shape[2] != 0 && shape[3] != 0 && pool.Kernel[0] > math.MaxInt64/pool.Kernel[1] {
			return nil, conv, pool, fmt.Errorf("%w: average kernel area overflow", graph.ErrInvalidModel)
		}
	}

	return shape, conv, pool, err
}
