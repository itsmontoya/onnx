package operators

import (
	"fmt"

	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

// Limits continues graph load accounting and bounds known tensor metadata.
// Zero fields select finite contract defaults. Compilation allocates no numeric output tensors.
type Limits struct {
	MaxLoadBytes   int64
	MaxTensorBytes int64
	MaxRank        int
	MaxDimension   int64
}

func (l Limits) normalized() (out Limits, err error) {
	if l.MaxLoadBytes < 0 || l.MaxTensorBytes < 0 || l.MaxRank < 0 || l.MaxDimension < 0 {
		return out, fmt.Errorf("%w: negative compilation limit", runtimeerror.ErrInvalidArgument)
	}

	if l.MaxLoadBytes == 0 {
		l.MaxLoadBytes = 1 << 30
	}

	if l.MaxTensorBytes == 0 {
		l.MaxTensorBytes = 64 << 20
	}

	if l.MaxRank == 0 {
		l.MaxRank = 32
	}

	if l.MaxDimension == 0 {
		l.MaxDimension = 1<<31 - 1
	}

	return l, nil
}
