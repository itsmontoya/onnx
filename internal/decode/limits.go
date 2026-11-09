package decode

import "fmt"

const (
	defaultMaxBytes     int64 = 64 << 20
	defaultMaxRecursion       = 64
)

// Limits bounds one serialized model or tensor. Zero fields select finite defaults.
// Byte limits count wire bytes, not protobuf object overhead or owned tensors.
// MaxRecursion bounds known messages; unknown groups use protobuf's fixed cap.
type Limits struct {
	MaxBytes     int64
	MaxRecursion int
}

func (l Limits) normalized() (out Limits, err error) {
	if l.MaxBytes < 0 || l.MaxRecursion < 0 {
		return out, fmt.Errorf("%w: negative byte or recursion limit", ErrInvalidArgument)
	}

	if l.MaxBytes == 0 {
		l.MaxBytes = defaultMaxBytes
	}

	if l.MaxRecursion == 0 {
		l.MaxRecursion = defaultMaxRecursion
	}

	// Readers consume one extra byte to distinguish exact-size input from oversize.
	if l.MaxBytes >= int64(^uint(0)>>1) {
		return out, fmt.Errorf("%w: byte limit plus one must fit a host index", ErrInvalidArgument)
	}

	return l, nil
}
