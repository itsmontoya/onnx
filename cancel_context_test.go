package onnx

import (
	"context"
	"errors"
	"testing"
)

func newCancelContext(cancelAfter int) (out *cancelContext) {
	var c cancelContext
	c.Context, c.cancel = context.WithCancel(context.Background())
	c.cancelAfter = cancelAfter
	return &c
}

type cancelContext struct {
	context.Context
	cancel      context.CancelFunc
	checks      int
	cancelAfter int
}

func (c *cancelContext) Err() (err error) {
	c.checks++
	if c.checks == c.cancelAfter {
		c.cancel()
	}

	return c.Context.Err()
}

func TestCancelContext(t *testing.T) {
	ctx := newCancelContext(2)
	defer ctx.cancel()
	if err := ctx.Err(); err != nil {
		t.Fatal(err)
	}

	if err := ctx.Err(); !errors.Is(err, context.Canceled) {
		t.Fatalf("context did not cancel: %v", err)
	}
}
