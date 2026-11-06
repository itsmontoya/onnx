package decode

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

// cancelContext triggers a real cancellation on a chosen polling boundary.
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
		t.Fatalf("polling boundary did not cancel: %v", err)
	}
}

func TestTensorDecoderCancellationBoundaries(t *testing.T) {
	tests := []struct {
		name        string
		wire        *onnxpb.TensorProto
		cancelAfter int
	}{
		{"float staging loop", &onnxpb.TensorProto{DataType: proto.Int32(1), Dims: []int64{2048}, RawData: make([]byte, 2048*4)}, 3},
		{"integer staging loop", &onnxpb.TensorProto{DataType: proto.Int32(7), Dims: []int64{2048}, RawData: make([]byte, 2048*8)}, 3},
		{"after synchronous backend copy", &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}, 3},
	}

	for _, testcase := range tests {
		t.Run(testcase.name, func(t *testing.T) {
			decoder, err := NewTensorDecoder(context.Background(), testcase.wire, 0, TensorLimits{})
			if err != nil {
				t.Fatal(err)
			}

			beforeLoad, beforeOwned := decoder.Usage()
			ctx := newCancelContext(testcase.cancelAfter)
			defer ctx.cancel()
			value, err := decoder.Decode(ctx, testcase.wire)
			if value != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation: %v, %v", value, err)
			}

			if load, owned := decoder.Usage(); load != beforeLoad || owned != beforeOwned {
				t.Fatalf("failed conversion retained logical reservations: %d, %d", load, owned)
			}
		})
	}

	ctx := newCancelContext(2)
	defer ctx.cancel()
	wire := &onnxpb.TensorProto{DataType: proto.Int32(1), FloatData: []float32{1}}
	if value, err := NewTensorDecoder(ctx, wire, 0, TensorLimits{}); value != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("payload traversal cancellation: %v, %v", value, err)
	}
}
