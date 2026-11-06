package graph

import (
	"fmt"

	"github.com/itsmontoya/onnx/internal/decode"
)

// Limits bounds structural records, strings and inline conversion for one build.
// Tensor limits retain decode.TensorLimits defaults. Zero fields select finite
// defaults; byte limits measure logical payload rather than Go object overhead.
type Limits struct {
	Tensor        decode.TensorLimits
	MaxNodes      int
	MaxValues     int
	MaxAttributes int
	MaxNameBytes  int
}

func (l Limits) normalized() (out Limits, err error) {
	if l.MaxNodes < 0 || l.MaxValues < 0 || l.MaxAttributes < 0 || l.MaxNameBytes < 0 {
		return out, fmt.Errorf("%w: negative graph limit", ErrInvalidArgument)
	}

	if l.MaxNodes == 0 {
		l.MaxNodes = 100000
	}

	if l.MaxValues == 0 {
		l.MaxValues = 200000
	}

	if l.MaxAttributes == 0 {
		l.MaxAttributes = 400000
	}

	if l.MaxNameBytes == 0 {
		l.MaxNameBytes = 4096
	}

	if l.Tensor.MaxRank == 0 {
		l.Tensor.MaxRank = 32
	}

	if l.Tensor.MaxDimension == 0 {
		l.Tensor.MaxDimension = 1<<31 - 1
	}

	return l, nil
}
