package decode

import (
	"context"
	"io"
)

type rangeResolver struct {
	reader   io.ReadCloser
	failure  error
	calls    int
	location string
	offset   int64
	length   int64
}

func (r *rangeResolver) Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error) {
	r.calls++
	r.location, r.offset, r.length = location, offset, length
	return r.reader, r.failure
}
