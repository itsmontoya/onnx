package onnx

import "fmt"

// Options configures one load. Zero limits select finite defaults.
// ExternalData remains unsupported until external-data loading is implemented.
type Options struct {
	Limits       Limits
	ExternalData ExternalDataResolver
}

func (o Options) normalized() (out Options, err error) {
	if o.Limits, err = o.Limits.normalized(); err != nil {
		return out, err
	}

	if o.ExternalData != nil {
		return out, fmt.Errorf("%w: external-data resolver requires RT-14", ErrUnsupported)
	}

	return o, nil
}
