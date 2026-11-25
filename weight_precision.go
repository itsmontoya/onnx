package onnx

import "github.com/itsmontoya/neuralnetwork/gpu"

// WeightPrecision selects the device storage precision for eligible constant
// float32 weights when executing on Metal. The zero value selects Float32.
type WeightPrecision uint8

const (
	// WeightFloat32 stores resident Metal weights in IEEE-754 binary32.
	WeightFloat32 WeightPrecision = iota
	// WeightFloat16 converts eligible resident Metal weights to IEEE-754 binary16
	// (round-to-nearest-even) with FP32 accumulation on device.
	WeightFloat16
	// WeightBFloat16 converts eligible resident Metal weights to BFloat16
	// (round-to-nearest-even) with FP32 accumulation on device.
	WeightBFloat16
)

// Valid reports whether p is a recognized weight storage precision.
func (p WeightPrecision) Valid() (ok bool) {
	return p == WeightFloat32 || p == WeightFloat16 || p == WeightBFloat16
}

// String returns the canonical lowercase name of p.
func (p WeightPrecision) String() (name string) {
	switch p {
	case WeightFloat32:
		return "float32"
	case WeightFloat16:
		return "float16"
	case WeightBFloat16:
		return "bfloat16"
	default:
		return "invalid"
	}
}

func (p WeightPrecision) gpuDType() (dtype gpu.DType) {
	switch p {
	case WeightFloat16:
		return gpu.Float16
	case WeightBFloat16:
		return gpu.BFloat16
	default:
		return gpu.Float32
	}
}
