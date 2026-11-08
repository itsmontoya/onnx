package onnx

import (
	"context"
	"io"
)

// ExternalDataResolver opens an explicitly permitted external tensor range.
// Resolve receives a nonempty location, nonnegative offset and exact byte length,
// even when the model omits length. The returned reader starts at offset and is
// closed by the loader on every path, including when Resolve also returns an error.
// The loader reads at most length bytes and owns the resulting tensor values.
// Implementations must support concurrent loads if shared between goroutines.
// Cancellation of a blocked Resolve or Read is the implementation's responsibility.
type ExternalDataResolver interface {
	Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error)
}
