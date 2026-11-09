package main

import (
	"bytes"
	"context"
	"fmt"
	"io"

	"github.com/itsmontoya/onnx"
)

func newMemoryResolver(weights []byte) (out *memoryResolver) {
	var r memoryResolver
	r.weights = bytes.Clone(weights)
	return &r
}

// memoryResolver permits one explicit location without filesystem assumptions.
// Its immutable bytes support concurrent loads on all Go platforms.
type memoryResolver struct {
	weights []byte
}

func (r *memoryResolver) Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error) {
	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("resolve memory weights: %w", err)
	}

	if location != "weights.bin" || offset < 0 || offset > int64(len(r.weights)) || length < 0 || length > int64(len(r.weights))-offset {
		return nil, fmt.Errorf("resolve memory weights %q: %w", location, onnx.ErrInvalidArgument)
	}

	return io.NopCloser(bytes.NewReader(r.weights[offset : offset+length])), nil
}
