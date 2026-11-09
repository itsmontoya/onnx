package decode

import (
	"context"
	"io"
)

// ExternalDataResolver is the consumer's minimal external range contract.
// The caller owns the resolver; the decoder owns every returned reader.
type ExternalDataResolver interface {
	Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error)
}
