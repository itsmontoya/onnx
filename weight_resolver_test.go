package onnx

import (
	"bytes"
	"context"
	"fmt"
	"io"
)

type weightResolver struct {
	values map[string][]byte
}

func (r *weightResolver) Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error) {
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	data, exists := r.values[location]
	if !exists || offset < 0 || offset > int64(len(data)) || length < 0 || length > int64(len(data))-offset {
		return nil, fmt.Errorf("test weight location %q: %w", location, ErrInvalidArgument)
	}

	return io.NopCloser(bytes.NewReader(data[offset : offset+length])), nil
}
