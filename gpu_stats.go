package onnx

// GPUStats reports immutable model-level GPU residency established at load time.
// CPU models return the zero value.
type GPUStats struct {
	Active                    bool
	WeightPrecision           WeightPrecision
	WeightTensors             int
	ReducedPrecisionTensors   int
	WeightBytes               uint64
	FP32EquivalentWeightBytes uint64
}
