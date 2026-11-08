package decode

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"

	"github.com/itsmontoya/onnx/internal/onnxpb"
)

func newExternalRange(value *onnxpb.TensorProto, bytes int64) (out externalRange, err error) {
	out.length = bytes
	for _, entry := range value.ExternalData {
		if entry.GetKey() == "location" {
			out.location = entry.GetValue()
		}
	}
	seen := make(map[string]bool)
	for _, entry := range value.ExternalData {
		key := entry.GetKey()

		if entry == nil || seen[key] {
			return out, fmt.Errorf("%w: nil or duplicate external key %q", ErrInvalidTensor, key)
		}

		seen[key] = true
		switch key {
		case "location":
		case "offset":
			if out.offset, err = externalDecimal(entry.GetValue()); err != nil {
				return out, fmt.Errorf("offset: %w", err)
			}

		case "length":
			var length int64
			if length, err = externalDecimal(entry.GetValue()); err != nil {
				return out, fmt.Errorf("length: %w", err)
			}

			if length != bytes {
				return out, fmt.Errorf("%w: external length %d, want %d", ErrInvalidTensor, length, bytes)
			}

		case "checksum":
			return out, fmt.Errorf("%w: external checksum verification", ErrUnsupported)
		default:
			return out, fmt.Errorf("%w: unknown external key %q", ErrUnsupported, key)
		}
	}
	if out.location == "" {
		return out, fmt.Errorf("%w: external location required", ErrInvalidTensor)
	}

	if out.offset > math.MaxInt64-bytes {
		return out, fmt.Errorf("%w: external range overflow", ErrInvalidTensor)
	}

	if value.RawData != nil || len(value.FloatData) != 0 || len(value.Int64Data) != 0 || len(value.Int32Data) != 0 || len(value.StringData) != 0 || len(value.DoubleData) != 0 || len(value.Uint64Data) != 0 {
		return out, fmt.Errorf("%w: external and inline encodings conflict", ErrInvalidTensor)
	}

	return out, nil
}

type externalRange struct {
	location string
	offset   int64
	length   int64
}

func (r externalRange) read(ctx context.Context, resolver ExternalDataResolver) (data []byte, err error) {
	var source io.ReadCloser
	source, err = resolver.Resolve(ctx, r.location, r.offset, r.length)
	if source != nil {
		defer func() {
			if closeErr := source.Close(); closeErr != nil {
				err = errors.Join(err, fmt.Errorf("close external range: %w", closeErr))
				data = nil
			}
		}()
	}

	if err != nil {
		return nil, fmt.Errorf("resolve external range: %w", err)
	}

	if source == nil {
		return nil, fmt.Errorf("%w: resolver returned nil reader", ErrInvalidArgument)
	}

	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("read external range: %w", err)
	}

	data = make([]byte, int(r.length))
	var position, emptyReads int
	for position < len(data) {
		if err = ctx.Err(); err != nil {
			return nil, fmt.Errorf("read external range: %w", err)
		}

		request := data[position : position+min(len(data)-position, 32<<10)]
		var count int
		count, err = source.Read(request)
		if count < 0 || count > len(request) {
			return nil, fmt.Errorf("%w: external reader count %d", ErrInvalidArgument, count)
		}

		position += count
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("read external range: %w", err)
		}

		if err == io.EOF && position != len(data) {
			return nil, fmt.Errorf("read external range: %w", io.ErrUnexpectedEOF)
		}

		if count > 0 {
			emptyReads = 0
			continue
		}

		emptyReads++
		if emptyReads == 100 {
			return nil, fmt.Errorf("read external range: %w", io.ErrNoProgress)
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, fmt.Errorf("read external range: %w", err)
	}

	return data, nil
}

func externalDecimal(value string) (number int64, err error) {
	if value == "" {
		return 0, fmt.Errorf("%w: empty decimal", ErrInvalidTensor)
	}

	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return 0, fmt.Errorf("%w: require nonnegative decimal int64", ErrInvalidTensor)
		}
	}
	if number, err = strconv.ParseInt(value, 10, 64); err != nil {
		return 0, fmt.Errorf("%w: decimal int64: %w", ErrInvalidTensor, err)
	}

	return number, nil
}
