package decode

import (
	"context"
	"fmt"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// countPayload walks decoded messages, not wire bytes. It charges actual field
// widths, including metadata, unknown bytes and unsupported payload fields.
func countPayload(ctx context.Context, message protoreflect.Message, depth int, limit int64, tensors map[*onnxpb.TensorProto]bool) (bytes int64, err error) {
	if depth <= 0 {
		return 0, fmt.Errorf("%w: decoded payload recursion limit", ErrLimit)
	}

	if err = ctx.Err(); err != nil {
		return 0, fmt.Errorf("count decoded payload: %w", err)
	}

	if _, sparse := message.Interface().(*onnxpb.SparseTensorProto); sparse {
		return 0, fmt.Errorf("%w: sparse tensor storage", ErrUnsupported)
	}

	if value, dense := message.Interface().(*onnxpb.TensorProto); dense {
		tensors[value] = true
	}

	bytes = int64(len(message.GetUnknown()))
	if bytes > limit {
		return 0, fmt.Errorf("%w: decoded unknown payload", ErrLimit)
	}

	message.Range(func(field protoreflect.FieldDescriptor, value protoreflect.Value) (more bool) {
		if field.IsList() {
			list := value.List()
			for index := 0; index < list.Len(); index++ {
				var size int64
				if size, err = countField(ctx, field, list.Get(index), depth, limit-bytes, tensors); err != nil {
					return false
				}

				bytes += size
			}
		} else {
			var size int64
			if size, err = countField(ctx, field, value, depth, limit-bytes, tensors); err != nil {
				return false
			}

			bytes += size
		}

		return true
	})
	return bytes, err
}

func countField(ctx context.Context, field protoreflect.FieldDescriptor, value protoreflect.Value, depth int, limit int64, tensors map[*onnxpb.TensorProto]bool) (bytes int64, err error) {
	if err = ctx.Err(); err != nil {
		return 0, fmt.Errorf("count decoded field %s: %w", field.FullName(), err)
	}

	switch field.Kind() {
	case protoreflect.MessageKind, protoreflect.GroupKind:
		return countPayload(ctx, value.Message(), depth-1, limit, tensors)
	case protoreflect.DoubleKind, protoreflect.Int64Kind, protoreflect.Uint64Kind, protoreflect.Sint64Kind, protoreflect.Fixed64Kind, protoreflect.Sfixed64Kind:
		bytes = 8
	case protoreflect.FloatKind, protoreflect.Int32Kind, protoreflect.Uint32Kind, protoreflect.Sint32Kind, protoreflect.Fixed32Kind, protoreflect.Sfixed32Kind, protoreflect.EnumKind:
		bytes = 4
	case protoreflect.BoolKind:
		bytes = 1
	case protoreflect.StringKind:
		bytes = int64(len(value.String()))
	case protoreflect.BytesKind:
		bytes = int64(len(value.Bytes()))
	}

	if bytes > limit {
		return 0, fmt.Errorf("%w: decoded field %s", ErrLimit, field.FullName())
	}

	return bytes, nil
}
