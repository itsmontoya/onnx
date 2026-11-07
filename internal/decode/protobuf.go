package decode

import (
	"context"
	"fmt"
	"io"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

// ParseModel decodes bounded wire bytes without validating ONNX graph semantics.
// The returned wire object owns its data and preserves protobuf unknown fields.
func ParseModel(ctx context.Context, data []byte, limits Limits) (out *onnxpb.ModelProto, err error) {
	var model onnxpb.ModelProto
	if err = unmarshal(ctx, data, &model, limits, "model"); err != nil {
		return nil, err
	}

	return &model, nil
}

// ReadModel reads at most the byte limit plus one, without closing source.
// Context cancellation is checked between reads and before/after unmarshalling.
func ReadModel(ctx context.Context, source io.Reader, limits Limits) (out *onnxpb.ModelProto, err error) {
	out, _, err = ReadModelWithSize(ctx, source, limits)
	return out, err
}

// ReadModelWithSize also reports the actual encoded length for load accounting.
// The length includes noncanonical encodings and retained unknown fields.
func ReadModelWithSize(ctx context.Context, source io.Reader, limits Limits) (out *onnxpb.ModelProto, encodedBytes int64, err error) {
	var data []byte
	if data, err = readWire(ctx, source, limits, "model"); err != nil {
		return nil, 0, err
	}

	if out, err = ParseModel(ctx, data, limits); err != nil {
		return nil, 0, err
	}

	return out, int64(len(data)), nil
}

// ParseTensor decodes TensorProto wire bytes using the same bounded decoder.
// Storage types, dimensions, and payload encodings are validated separately.
func ParseTensor(ctx context.Context, data []byte, limits Limits) (out *onnxpb.TensorProto, err error) {
	var value onnxpb.TensorProto
	if err = unmarshal(ctx, data, &value, limits, "tensor"); err != nil {
		return nil, err
	}

	return &value, nil
}

// ReadTensor reads a bounded TensorProto without closing source.
func ReadTensor(ctx context.Context, source io.Reader, limits Limits) (out *onnxpb.TensorProto, err error) {
	var data []byte
	if data, err = readWire(ctx, source, limits, "tensor"); err != nil {
		return nil, err
	}

	return ParseTensor(ctx, data, limits)
}

func unmarshal(ctx context.Context, data []byte, destination proto.Message, limits Limits, kind string) (err error) {
	if limits, err = validate(ctx, limits); err != nil {
		return err
	}

	if int64(len(data)) > limits.MaxBytes {
		return fmt.Errorf("decode %s: %w (%d > %d)", kind, ErrLimit, len(data), limits.MaxBytes)
	}

	options := proto.UnmarshalOptions{RecursionLimit: limits.MaxRecursion}
	if err = options.Unmarshal(data, destination); err != nil {
		return fmt.Errorf("decode %s: %w: %w", kind, ErrInvalidWire, err)
	}

	if err = ctx.Err(); err != nil {
		return fmt.Errorf("decode %s: %w", kind, err)
	}

	return nil
}

func validate(ctx context.Context, limits Limits) (out Limits, err error) {
	if out, err = limits.normalized(); err != nil {
		return out, err
	}

	if ctx == nil {
		return out, fmt.Errorf("%w: nil context", ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return out, fmt.Errorf("decode protobuf: %w", err)
	}

	return out, nil
}

func readWire(ctx context.Context, source io.Reader, limits Limits, kind string) (data []byte, err error) {
	if limits, err = validate(ctx, limits); err != nil {
		return nil, err
	}

	if source == nil {
		return nil, fmt.Errorf("%w: nil %s reader", ErrInvalidArgument, kind)
	}

	buffer := make([]byte, min(limits.MaxBytes+1, 32<<10))
	var emptyReads int
	for {
		if err = ctx.Err(); err != nil {
			return nil, fmt.Errorf("read %s: %w", kind, err)
		}

		remaining := limits.MaxBytes + 1 - int64(len(data))
		request := buffer[:min(int64(len(buffer)), remaining)]
		var count int
		count, err = source.Read(request)
		if count < 0 || count > len(request) {
			return nil, fmt.Errorf("read %s: %w: reader returned invalid count %d", kind, ErrInvalidArgument, count)
		}

		data = append(data, request[:count]...)
		if int64(len(data)) > limits.MaxBytes {
			return nil, fmt.Errorf("read %s: %w (%d)", kind, ErrLimit, limits.MaxBytes)
		}

		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("read %s: %w", kind, err)
		}

		if err == io.EOF {
			return data, nil
		}

		if count > 0 {
			emptyReads = 0
			continue
		}

		emptyReads++
		if emptyReads == 100 {
			return nil, fmt.Errorf("read %s: %w", kind, io.ErrNoProgress)
		}
	}
}
