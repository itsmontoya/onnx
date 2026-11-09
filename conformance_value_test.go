package onnx

import (
	"context"
	"fmt"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
)

type conformanceValue struct {
	Name   string              `json:"name"`
	Tensor conformanceArtifact `json:"tensor"`
}

func (v conformanceValue) read() (out NamedTensor, err error) {
	var (
		data    []byte
		wire    *onnxpb.TensorProto
		decoder *decode.TensorDecoder
		value   Tensor
	)

	if data, err = v.Tensor.read(); err != nil {
		return out, fmt.Errorf("value %q: %w", v.Name, err)
	}

	if wire, err = decode.ParseTensor(context.Background(), data, decode.Limits{}); err != nil {
		return out, fmt.Errorf("parse value %q: %w", v.Name, err)
	}

	if v.Name == "" || wire.GetName() != v.Name {
		return out, fmt.Errorf("value name %q, TensorProto name %q", v.Name, wire.GetName())
	}

	if decoder, err = decode.NewTensorDecoder(context.Background(), wire, int64(len(data)), decode.TensorLimits{}); err != nil {
		return out, fmt.Errorf("prepare value %q: %w", v.Name, err)
	}

	if value.value, err = decoder.Decode(context.Background(), wire); err != nil {
		return out, fmt.Errorf("decode value %q: %w", v.Name, err)
	}

	return NamedTensor{Name: v.Name, Tensor: &value}, nil
}
