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

func newAttributeConstant(ctx context.Context, decoder *decode.TensorDecoder, attribute Attribute) (out *Constant, err error) {
	var (
		c        Constant
		shape    []int64
		floats   []float32
		integers []int64
		dataType tensor.DataType
	)

	switch attribute.Name {
	case "value_float":
		dataType, floats = tensor.Float32, []float32{attribute.Float}
	case "value_floats":
		dataType, floats, shape = tensor.Float32, attribute.Floats, []int64{int64(len(attribute.Floats))}
	case "value_int":
		dataType, integers = tensor.Int64, []int64{attribute.Int}
	case "value_ints":
		dataType, integers, shape = tensor.Int64, attribute.Ints, []int64{int64(len(attribute.Ints))}
	}
	if c.value, err = decoder.Materialize(ctx, shape, floats, integers, dataType); err != nil {
		return nil, err
	}

	return &c, nil
}

// Constant owns an immutable decoded tensor. Execution may borrow it only as a
// read-only source; writable consumers must Clone and budget the copy first.
// Typed accessors also copy and require budgeted snapshots.
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

// Borrow returns the retained tensor for read-only execution. Consumers must
// never write to it, use it as a destination, or place it in reusable storage.
// Keeping the returned pointer retains its storage independently of the graph.
func (c *Constant) Borrow() (out *tensor.Tensor) { return c.value }
