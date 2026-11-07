package graph

import (
	"context"
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
)

func newConstant(ctx context.Context, decoder *decode.TensorDecoder, wire *onnxpb.TensorProto) (out *Constant, err error) {
	var c Constant
	if c.metadata, err = properties(wire.GetMetadataProps()); err != nil {
		return nil, err
	}

	if c.value, err = decoder.Decode(ctx, wire); err != nil {
		return nil, err
	}

	c.name = wire.GetName()
	c.docString = wire.GetDocString()
	return &c, nil
}

// Constant owns a decoded tensor without exposing a writable backend pointer.
// Clone allocates for a future consumer, which must reserve that copy in its own
// budget before calling. Compilation uses no additional numerical copies.
type Constant struct {
	name      string
	docString string
	metadata  []Property
	value     *tensor.Tensor
}

// Name returns the initializer or optional attribute tensor label.
func (c *Constant) Name() (name string) { return c.name }

// DocString returns preserved tensor documentation.
func (c *Constant) DocString() (doc string) { return c.docString }

// Metadata returns an independent metadata snapshot.
func (c *Constant) Metadata() (metadata []Property) { return slices.Clone(c.metadata) }

// DataType returns the exact numerical storage type.
func (c *Constant) DataType() (dataType tensor.DataType) { return c.value.DataType() }

// Rank returns the concrete rank without allocating a shape snapshot.
func (c *Constant) Rank() (rank int) { return c.value.Rank() }

// Shape returns an independent concrete shape.
func (c *Constant) Shape() (shape []int64) { return c.value.Shape() }

// ByteSize returns owned numerical bytes, excluding metadata.
func (c *Constant) ByteSize() (bytes int64) { return c.value.ByteSize() }

// Float32Values returns an owned float32 snapshot, preserving float bits.
func (c *Constant) Float32Values() (values []float32, err error) { return c.value.Float32Values() }

// Int64Values returns an owned int64 snapshot without floating conversion.
func (c *Constant) Int64Values() (values []int64, err error) { return c.value.Int64Values() }

// Clone returns independent backend storage for a budgeted consumer.
func (c *Constant) Clone() (out *tensor.Tensor, err error) { return c.value.Clone() }
