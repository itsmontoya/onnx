package onnx

import (
	"context"
	"testing"
)

func newDeadlineContext(cancellation *cancelContext) (out *deadlineContext) {
	var c deadlineContext
	c.cancelContext = cancellation
	return &c
}

// deadlineContext exercises a deadline error at deterministic polling boundaries.
type deadlineContext struct {
	*cancelContext
}

func (c *deadlineContext) Err() (err error) {
	if err = c.cancelContext.Err(); err != nil {
		return context.DeadlineExceeded
	}

	return nil
}

func TestDeadlineContext(t *testing.T) {
	cancellation := newCancelContext(2)
	defer cancellation.cancel()
	ctx := newDeadlineContext(cancellation)
	if err := ctx.Err(); err != nil {
		t.Fatal(err)
	}

	if err := ctx.Err(); err != context.DeadlineExceeded {
		t.Fatalf("deadline did not expire: %v", err)
	}
}
