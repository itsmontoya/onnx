package onnx

// Options configures one load. Zero limits select finite defaults.
// ExternalData is required for external tensors and remains caller-owned.
type Options struct {
	Limits       Limits
	ExternalData ExternalDataResolver
}

func (o Options) normalized() (out Options, err error) {
	if o.Limits, err = o.Limits.normalized(); err != nil {
		return out, err
	}

	return o, nil
}
