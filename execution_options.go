package onnx

import "fmt"

// ExecutionOptions selects the execution backend and device weight precision.
// Its zero value selects default CPU execution with no behavioral change.
type ExecutionOptions struct {
	Device          Device
	WeightPrecision WeightPrecision
}

func (o ExecutionOptions) normalized() (out ExecutionOptions, err error) {
	if !o.Device.Valid() {
		return out, fmt.Errorf("%w: execution device %d", ErrInvalidArgument, o.Device)
	}

	if !o.WeightPrecision.Valid() {
		return out, fmt.Errorf("%w: weight precision %d", ErrInvalidArgument, o.WeightPrecision)
	}

	if o.Device == DeviceCPU && o.WeightPrecision != WeightFloat32 {
		return out, fmt.Errorf("%w: weight precision %s requires DeviceMetal", ErrInvalidArgument, o.WeightPrecision)
	}

	return o, nil
}
