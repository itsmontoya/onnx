package decode

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

// NewTensorDecoder accounts encoded bytes and all decoded payloads in wire before
// numeric conversion. Pass the complete model, attribute or test tensor message
// and its original encoded length. Do not mutate wire during the decoder's life.
// The decoder retains tensor messages, but never borrows their numeric storage.
func NewTensorDecoder(ctx context.Context, wire proto.Message, encodedBytes int64, limits TensorLimits) (out *TensorDecoder, err error) {
	var d TensorDecoder
	if err = d.initialize(ctx, wire, encodedBytes, limits, nil); err != nil {
		return nil, err
	}

	return &d, nil
}

// NewTensorDecoderWithResolver also permits explicitly resolved external ranges.
// The resolver remains caller-owned; each returned reader is closed by Decode.
func NewTensorDecoderWithResolver(ctx context.Context, wire proto.Message, encodedBytes int64, limits TensorLimits, resolver ExternalDataResolver) (out *TensorDecoder, err error) {
	var d TensorDecoder
	if err = d.initialize(ctx, wire, encodedBytes, limits, resolver); err != nil {
		return nil, err
	}

	return &d, nil
}

// TensorDecoder converts registered dense TensorProto values to owned backend
// tensors. Use one decoder per load to aggregate initializer/Constant copies.
// It is not safe for concurrent calls or mutation of registered wire messages.
// Raw conversion reserves staging plus an owned copy; typed conversion reserves
// its owned copy. External conversion also reserves its read buffer. Successful
// staging charges remain reserved conservatively for the load's lifetime.
// Accounting excludes Go object/allocation overhead.
type TensorDecoder struct {
	resolver         ExternalDataResolver
	limits           TensorLimits
	tensors          map[*onnxpb.TensorProto]bool
	loadBytes        int64
	initializerBytes int64
}

func (d *TensorDecoder) initialize(ctx context.Context, wire proto.Message, encodedBytes int64, limits TensorLimits, resolver ExternalDataResolver) (err error) {
	var decodedBytes int64

	if d.limits, err = limits.normalized(); err != nil {
		return err
	}

	if ctx == nil || wire == nil || !wire.ProtoReflect().IsValid() || encodedBytes < 0 {
		return fmt.Errorf("%w: context, wire message or encoded byte count", ErrInvalidArgument)
	}

	if wire.ProtoReflect().Descriptor().ParentFile() != onnxpb.File_onnx_proto {
		return fmt.Errorf("%w: message is not from the pinned ONNX schema", ErrInvalidArgument)
	}

	if encodedBytes > d.limits.MaxLoadBytes {
		return fmt.Errorf("%w: encoded payload", ErrLimit)
	}

	d.resolver = resolver
	d.tensors = make(map[*onnxpb.TensorProto]bool)
	if decodedBytes, err = countPayload(ctx, wire.ProtoReflect(), d.limits.MaxRecursion, d.limits.MaxLoadBytes-encodedBytes, d.tensors); err != nil {
		return err
	}

	d.loadBytes = encodedBytes + decodedBytes
	return nil
}

// Decode validates a dense tensor before allocating numerical storage.
// Both wire parsing entry points and all tensor-bearing model fields share it.
func (d *TensorDecoder) Decode(ctx context.Context, value *onnxpb.TensorProto) (out *tensor.Tensor, err error) {
	if d == nil || d.tensors == nil || ctx == nil || value == nil || !d.tensors[value] {
		return nil, fmt.Errorf("%w: unregistered tensor or invalid decoder/context", ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("decode tensor %q: %w", value.GetName(), err)
	}

	var (
		dataType                tensor.DataType
		elements, bytes, charge int64
		external                externalRange
		copies                  int64 = 1
	)

	if dataType, elements, bytes, err = d.validate(value); err != nil {
		return nil, fmt.Errorf("decode tensor %q: %w", value.GetName(), err)
	}

	if value.GetDataLocation() == onnxpb.TensorProto_EXTERNAL {
		if external, err = newExternalRange(value, bytes); err != nil {
			return nil, fmt.Errorf("decode tensor %q location %q: %w", value.GetName(), external.location, err)
		}

		if d.resolver == nil {
			return nil, fmt.Errorf("decode tensor %q location %q: %w: explicit external-data resolver required", value.GetName(), external.location, ErrUnsupported)
		}
	}

	if bytes > d.limits.MaxInitializerBytes-d.initializerBytes {
		return nil, fmt.Errorf("decode tensor %q: %w: aggregate initializer bytes", value.GetName(), ErrLimit)
	}

	// Every external range coexists with raw bytes, typed staging and ownership.
	if value.RawData != nil {
		copies = 2
	}

	if value.GetDataLocation() == onnxpb.TensorProto_EXTERNAL {
		copies = 3
	}

	if bytes > (d.limits.MaxLoadBytes-d.loadBytes)/copies {
		return nil, fmt.Errorf("decode tensor %q location %q: %w: conversion buffers and owned copy", value.GetName(), external.location, ErrLimit)
	}

	charge = bytes * copies
	conversion := value
	if value.GetDataLocation() == onnxpb.TensorProto_EXTERNAL {
		var raw []byte
		if raw, err = external.read(ctx, d.resolver); err != nil {
			return nil, fmt.Errorf("decode tensor %q location %q: %w", value.GetName(), external.location, err)
		}

		// Construct only the conversion fields; generated messages contain locks.
		conversion = &onnxpb.TensorProto{Dims: value.Dims, RawData: raw}
	}

	if out, err = convertTensor(ctx, conversion, dataType, elements); err != nil {
		return nil, fmt.Errorf("decode tensor %q location %q: %w", value.GetName(), external.location, err)
	}

	d.loadBytes += charge
	d.initializerBytes += bytes
	return out, nil
}

// Usage returns conservative load payload bytes and owned initializer bytes.
func (d *TensorDecoder) Usage() (loadBytes, initializerBytes int64) {
	if d == nil {
		return 0, 0
	}

	return d.loadBytes, d.initializerBytes
}

// ReservePayload charges an additional owned metadata/attribute snapshot before
// allocating it. It does not charge initializer storage or reserve numeric tensors;
// those must go through Decode. Failed reservations leave usage unchanged.
func (d *TensorDecoder) ReservePayload(ctx context.Context, bytes int64) (err error) {
	if d == nil || d.tensors == nil || ctx == nil || bytes < 0 {
		return fmt.Errorf("%w: invalid payload reservation", ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return fmt.Errorf("reserve snapshot payload: %w", err)
	}

	if bytes > d.limits.MaxLoadBytes-d.loadBytes {
		return fmt.Errorf("%w: owned snapshot payload", ErrLimit)
	}

	d.loadBytes += bytes
	return nil
}

func (d *TensorDecoder) validate(value *onnxpb.TensorProto) (dataType tensor.DataType, elements, bytes int64, err error) {
	if value.Segment != nil {
		return dataType, 0, 0, fmt.Errorf("%w: tensor segments", ErrUnsupported)
	}

	if value.GetDataLocation() != onnxpb.TensorProto_DEFAULT && value.GetDataLocation() != onnxpb.TensorProto_EXTERNAL {
		return dataType, 0, 0, fmt.Errorf("%w: data location %d", ErrUnsupported, value.GetDataLocation())
	}

	if value.GetDataLocation() == onnxpb.TensorProto_DEFAULT && len(value.ExternalData) != 0 {
		return dataType, 0, 0, fmt.Errorf("%w: external metadata requires EXTERNAL data location", ErrUnsupported)
	}

	switch value.GetDataType() {
	case int32(onnxpb.TensorProto_FLOAT):
		dataType = tensor.Float32
	case int32(onnxpb.TensorProto_INT64):
		dataType = tensor.Int64
	default:
		return dataType, 0, 0, fmt.Errorf("%w: tensor data type %d", ErrUnsupported, value.GetDataType())
	}

	if len(value.Dims) > d.limits.MaxRank {
		return dataType, 0, 0, fmt.Errorf("%w: tensor rank %d", ErrLimit, len(value.Dims))
	}

	for axis, dimension := range value.Dims {
		if dimension < 0 {
			return dataType, 0, 0, fmt.Errorf("%w: negative dimension %d at axis %d", ErrInvalidTensor, dimension, axis)
		}

		if dimension > d.limits.MaxDimension {
			return dataType, 0, 0, fmt.Errorf("%w: dimension %d at axis %d", ErrLimit, dimension, axis)
		}
	}

	if elements, bytes, err = tensor.ValidateShape(value.Dims, dataType); err != nil {
		return dataType, 0, 0, fmt.Errorf("%w: shape: %w", ErrInvalidTensor, err)
	}

	if bytes > d.limits.MaxTensorBytes {
		return dataType, 0, 0, fmt.Errorf("%w: tensor bytes %d", ErrLimit, bytes)
	}

	if value.GetDataLocation() == onnxpb.TensorProto_EXTERNAL {
		return dataType, elements, bytes, nil
	}

	if err = validateEncoding(value, dataType, elements, bytes); err != nil {
		return dataType, 0, 0, err
	}

	return dataType, elements, bytes, nil
}

func validateEncoding(value *onnxpb.TensorProto, dataType tensor.DataType, elements, bytes int64) (err error) {
	if len(value.Int32Data) != 0 || len(value.StringData) != 0 || len(value.DoubleData) != 0 || len(value.Uint64Data) != 0 || (dataType == tensor.Float32 && len(value.Int64Data) != 0) || (dataType == tensor.Int64 && len(value.FloatData) != 0) {
		return fmt.Errorf("%w: incompatible typed payload for data type %d", ErrInvalidTensor, value.GetDataType())
	}

	count := int64(len(value.Int64Data))
	if dataType == tensor.Float32 {
		count = int64(len(value.FloatData))
	}

	if value.RawData != nil {
		if count != 0 {
			return fmt.Errorf("%w: conflicting raw and typed encodings", ErrInvalidTensor)
		}

		if int64(len(value.RawData)) != bytes {
			return fmt.Errorf("%w: raw byte count %d, want %d", ErrInvalidTensor, len(value.RawData), bytes)
		}

		return nil
	}

	if count != elements {
		return fmt.Errorf("%w: typed element count %d, want %d", ErrInvalidTensor, count, elements)
	}

	return nil
}

func convertTensor(ctx context.Context, value *onnxpb.TensorProto, dataType tensor.DataType, elements int64) (out *tensor.Tensor, err error) {
	if dataType == tensor.Float32 {
		values := value.FloatData
		if value.RawData != nil {
			values = make([]float32, int(elements))
			for index := range values {
				if index%1024 == 0 {
					if err = ctx.Err(); err != nil {
						return nil, err
					}
				}

				values[index] = math.Float32frombits(binary.LittleEndian.Uint32(value.RawData[index*4:]))
			}
		}

		if err = ctx.Err(); err != nil {
			return nil, err
		}

		out, err = tensor.FromFloat32(value.Dims, values)
	} else {
		values := value.Int64Data
		if value.RawData != nil {
			values = make([]int64, int(elements))
			for index := range values {
				if index%1024 == 0 {
					if err = ctx.Err(); err != nil {
						return nil, err
					}
				}

				values[index] = int64(binary.LittleEndian.Uint64(value.RawData[index*8:]))
			}
		}

		if err = ctx.Err(); err != nil {
			return nil, err
		}

		out, err = tensor.FromInt64(value.Dims, values)
	}

	if err != nil {
		return nil, err
	}

	if err = ctx.Err(); err != nil {
		return nil, err
	}

	return out, nil
}

// Materialize converts a supported Constant scalar/list attribute through the
// same aggregate limits as inline tensors. Values are borrowed only for this
// synchronous call; the returned tensor owns its storage.
func (d *TensorDecoder) Materialize(ctx context.Context, shape []int64, floats []float32, integers []int64, dataType tensor.DataType) (out *tensor.Tensor, err error) {
	if d == nil || d.tensors == nil || ctx == nil {
		return nil, fmt.Errorf("%w: materialization decoder/context", ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return nil, err
	}

	wire := onnxpb.TensorProto{Dims: shape, FloatData: floats, Int64Data: integers}
	if dataType == tensor.Float32 {
		wire.DataType = proto.Int32(int32(onnxpb.TensorProto_FLOAT))
	} else if dataType == tensor.Int64 {
		wire.DataType = proto.Int32(int32(onnxpb.TensorProto_INT64))
	} else {
		return nil, fmt.Errorf("%w: Constant materialization type", ErrUnsupported)
	}

	var elements, bytes int64
	if _, elements, bytes, err = d.validate(&wire); err != nil {
		return nil, err
	}

	if bytes > d.limits.MaxInitializerBytes-d.initializerBytes || bytes > d.limits.MaxLoadBytes-d.loadBytes {
		return nil, fmt.Errorf("%w: aggregate Constant materialization bytes", ErrLimit)
	}

	if out, err = convertTensor(ctx, &wire, dataType, elements); err != nil {
		return nil, err
	}

	d.loadBytes += bytes
	d.initializerBytes += bytes
	return out, nil
}
