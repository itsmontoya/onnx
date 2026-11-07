package operators

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
		t.Fatalf("poll did not cancel: %v", err)
	}
}

func TestPlanCancellationBoundaries(t *testing.T) {
	registry := planRegistry(t)
	source := builtGraph(t, planFixture())
	baseline := newCancelContext(0)
	defer baseline.cancel()
	if _, err := Compile(baseline, source, registry, Limits{}); err != nil {
		t.Fatal(err)
	}

	for boundary := 1; boundary <= baseline.checks; boundary++ {
		ctx := newCancelContext(boundary)
		plan, err := Compile(ctx, source, registry, Limits{})
		ctx.cancel()
		if plan != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("boundary %d: %v, %v", boundary, plan, err)
		}
	}

	if plan, err := Compile(context.Background(), source, registry, Limits{}); plan == nil || err != nil {
		t.Fatalf("cancelled compilation changed retry: %v, %v", plan, err)
	}
}
