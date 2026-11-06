package graph

import (
	"context"
	"errors"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
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

func TestGraphCancellation(t *testing.T) {
	model := graphFixture()
	model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(1), FloatData: []float32{1}}}
	model.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{{Name: proto.String("tensor"), Type: onnxpb.AttributeProto_TENSOR.Enum(), T: &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{2048}, RawData: make([]byte, 2048*4)}}}
	encoded := encodedLength(t, model)
	baseline := newCancelContext(0)
	defer baseline.cancel()
	if _, err := Build(baseline, model, encoded, Limits{}); err != nil {
		t.Fatal(err)
	}

	// Exercise cancellation at every observed boundary through traversal,
	// metadata construction, conversion and final publication.
	for boundary := 1; boundary <= baseline.checks; boundary++ {
		ctx := newCancelContext(boundary)
		compiled, err := Build(ctx, model, encoded, Limits{})
		ctx.cancel()
		if compiled != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("boundary %d: %v, %v", boundary, compiled, err)
		}
	}

	if compiled, err := Build(context.Background(), model, encoded, Limits{}); compiled == nil || err != nil {
		t.Fatalf("cancelled build changed source/retry: %v, %v", compiled, err)
	}
}
