package onnx

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

// NewTensor constructs a zero-filled Tensor with an owned concrete shape.
// Scalar shapes have one element; a zero dimension produces an empty tensor.
// Size validation checks representability, rather than imposing a memory quota.
func NewTensor(dataType DataType, shape []int64) (out *Tensor, err error) {
	var t Tensor
	if t.value, err = tensor.New(dataType, shape); err != nil {
		return nil, fmt.Errorf("%w: create tensor: %w", ErrInvalidArgument, err)
	}

	return &t, nil
}

// NewFloat32 constructs a Tensor by copying shape and row-major values.
// The value count must exactly match the concrete shape, including empty shapes.
func NewFloat32(shape []int64, values []float32) (out *Tensor, err error) {
	var t Tensor
	if t.value, err = tensor.FromFloat32(shape, values); err != nil {
		return nil, fmt.Errorf("%w: create float32 tensor: %w", ErrInvalidArgument, err)
	}

	return &t, nil
}

// NewInt64 constructs a Tensor by copying shape and exact signed integer values.
// Integer payloads never pass through floating-point storage.
func NewInt64(shape []int64, values []int64) (out *Tensor, err error) {
	var t Tensor
	if t.value, err = tensor.FromInt64(shape, values); err != nil {
		return nil, fmt.Errorf("%w: create int64 tensor: %w", ErrInvalidArgument, err)
	}

	return &t, nil
}

// Tensor is an immutable facade over owned, contiguous backend tensor storage.
// Constructors, shape/value accessors, and Clone copy their arguments or results.
// Values may be shared across sessions and concurrent readers. Runtime outputs
// remain valid after subsequent runs or closure; no method exposes a writable
// backend destination. Nil and zero Tensor values are invalid, not scalar values.
type Tensor struct {
	value *tensor.Tensor
}

// Validate reports invalid receivers or backend storage.
func (t *Tensor) Validate() (err error) {
	if t == nil || t.value == nil {
		return ErrInvalidArgument
	}

	if err = t.value.Validate(); err != nil {
		return fmt.Errorf("%w: validate tensor: %w", ErrInvalidArgument, err)
	}

	return nil
}

// DataType returns Invalid for a nil or invalid receiver.
func (t *Tensor) DataType() (dataType DataType) {
	if t == nil {
		return Invalid
	}

	return t.value.DataType()
}

// Shape returns an owned copy, or nil for a nil or invalid receiver.
func (t *Tensor) Shape() (shape []int64) {
	if t == nil {
		return nil
	}

	return t.value.Shape()
}

// Rank returns the dimension count, or zero for a nil or invalid receiver.
func (t *Tensor) Rank() (rank int) {
	if t == nil {
		return 0
	}

	return t.value.Rank()
}

// Len returns the element count, or zero for a nil or invalid receiver.
func (t *Tensor) Len() (elements int64) {
	if t == nil {
		return 0
	}

	return t.value.Len()
}

// ByteSize returns numeric payload bytes, excluding shape/allocation overhead.
// A nil or invalid receiver has zero payload bytes.
func (t *Tensor) ByteSize() (bytes int64) {
	if t == nil {
		return 0
	}

	return t.value.ByteSize()
}

// Float32Values returns an owned copy, rejecting invalid or non-float32 tensors.
func (t *Tensor) Float32Values() (values []float32, err error) {
	if err = t.Validate(); err != nil {
		return nil, err
	}

	if values, err = t.value.Float32Values(); err != nil {
		return nil, fmt.Errorf("%w: read float32 tensor: %w", ErrInvalidArgument, err)
	}

	return values, nil
}

// Int64Values returns an owned exact copy, rejecting invalid or non-int64 tensors.
func (t *Tensor) Int64Values() (values []int64, err error) {
	if err = t.Validate(); err != nil {
		return nil, err
	}

	if values, err = t.value.Int64Values(); err != nil {
		return nil, fmt.Errorf("%w: read int64 tensor: %w", ErrInvalidArgument, err)
	}

	return values, nil
}

// Clone returns independent copies of the shape and numeric storage.
func (t *Tensor) Clone() (out *Tensor, err error) {
	if err = t.Validate(); err != nil {
		return nil, err
	}

	var cloned Tensor
	if cloned.value, err = t.value.Clone(); err != nil {
		return nil, fmt.Errorf("%w: clone tensor: %w", ErrInvalidArgument, err)
	}

	return &cloned, nil
}
