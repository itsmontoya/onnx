package onnx

import (
	"context"
	"sync"
	"testing"
)

func newBlockingContext() (out *blockingContext) {
	var c blockingContext
	c.Context = context.Background()
	c.entered, c.release = make(chan struct{}), make(chan struct{})
	return &c
}

type blockingContext struct {
	context.Context
	once        sync.Once
	releaseOnce sync.Once
	entered     chan struct{}
	release     chan struct{}
}

func (c *blockingContext) Err() (err error) {
	c.once.Do(func() { close(c.entered); <-c.release })
	return c.Context.Err()
}

func (c *blockingContext) releaseRun() { c.releaseOnce.Do(func() { close(c.release) }) }

func TestBlockingContext(t *testing.T) {
	ctx := newBlockingContext()
	ctx.releaseRun()
	ctx.releaseRun()
	if err := ctx.Err(); err != nil {
		t.Fatal(err)
	}

	select {
	case <-ctx.entered:
	default:
		t.Fatal("context did not mark entry")
	}
}
