package onnx

import (
	"context"
	"io"
)

// ExternalDataResolver reserves the explicit external-storage contract for RT-14.
// Current loaders reject a nonnil resolver and all external tensors as unsupported.
type ExternalDataResolver interface {
	Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error)
}
