package onnx

// Device selects the execution backend for a loaded Model.
// The zero value selects CPU execution.
type Device uint8

const (
	// DeviceCPU executes graphs on the host CPU (default).
	DeviceCPU Device = iota
	// DeviceMetal opts into Metal GPU execution through neuralnetwork/gpu.
	// Loading fails with ErrUnavailable when Metal cannot be opened.
	DeviceMetal
)

// Valid reports whether d is a recognized execution device.
func (d Device) Valid() (ok bool) {
	return d == DeviceCPU || d == DeviceMetal
}

// String returns the canonical lowercase name of d.
func (d Device) String() (name string) {
	switch d {
	case DeviceCPU:
		return "cpu"
	case DeviceMetal:
		return "metal"
	default:
		return "invalid"
	}
}
