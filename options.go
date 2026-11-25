package onnx

// Options configures one load. Zero limits select finite defaults.
// ExternalData is required for external tensors and remains caller-owned.
// Zero Execution selects default CPU execution.
type Options struct {
	Limits       Limits
	ExternalData ExternalDataResolver
	Execution    ExecutionOptions
}

func (o Options) normalized() (out Options, err error) {
	if o.Limits, err = o.Limits.normalized(); err != nil {
		return out, err
	}

	if o.Execution, err = o.Execution.normalized(); err != nil {
		return out, err
	}

	return o, nil
}
