package decode

import "fmt"

// TensorLimits bounds decoded payloads and dense tensor conversion. Zero fields
// select the runtime contract's finite defaults; negative fields are invalid.
// Logical bytes exclude Go object headers, slice capacity and allocator overhead.
type TensorLimits struct {
	MaxLoadBytes        int64
	MaxInitializerBytes int64
	MaxTensorBytes      int64
	MaxRank             int
	MaxDimension        int64
	MaxRecursion        int
}

func (l TensorLimits) normalized() (out TensorLimits, err error) {
	if l.MaxLoadBytes < 0 || l.MaxInitializerBytes < 0 || l.MaxTensorBytes < 0 || l.MaxRank < 0 || l.MaxDimension < 0 || l.MaxRecursion < 0 {
		return out, fmt.Errorf("%w: negative tensor limit", ErrInvalidArgument)
	}

	if l.MaxLoadBytes == 0 {
		l.MaxLoadBytes = 1 << 30
	}

	if l.MaxInitializerBytes == 0 {
		l.MaxInitializerBytes = 256 << 20
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

	if l.MaxRecursion == 0 {
		l.MaxRecursion = defaultMaxRecursion
	}

	return l, nil
}
