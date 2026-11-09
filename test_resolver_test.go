package onnx

import (
	"context"
	"io"
	"testing"
)

type testResolver struct{ called bool }

func (r *testResolver) Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error) {
	r.called = true
	return nil, ErrUnsupported
}

func TestResolverRejectedWithoutUse(t *testing.T) {
	resolver := &testResolver{}
	model, err := LoadBytes(context.Background(), modelFixture(t, "profile/identity"), Options{ExternalData: resolver})
	if model != nil || err == nil || resolver.called {
		t.Fatalf("resolver was used or silently ignored: %v, %v, %v", model, err, resolver.called)
	}
}
