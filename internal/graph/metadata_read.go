package graph

import (
	"fmt"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
)

func properties(wires []*onnxpb.StringStringEntryProto) (out []Property, err error) {
	seen := make(map[string]bool)
	for index, wire := range wires {
		if wire == nil {
			return nil, fmt.Errorf("metadata[%d]: %w: nil entry", index, ErrInvalidModel)
		}

		key := wire.GetKey()
		if seen[key] {
			return nil, fmt.Errorf("metadata key %q: %w: duplicate entry", key, ErrInvalidModel)
		}

		seen[key] = true
		out = append(out, Property{Key: key, Value: wire.GetValue()})
	}
	return out, nil
}

func readMetadata(model *onnxpb.ModelProto) (metadata Metadata, err error) {
	metadata.Name = model.Graph.GetName()
	metadata.IRVersion = model.GetIrVersion()
	metadata.Opset = 13
	metadata.Producer = model.GetProducerName()
	metadata.ProducerVersion = model.GetProducerVersion()
	metadata.ModelDomain = model.GetDomain()
	metadata.ModelVersion = model.GetModelVersion()
	metadata.DocString = model.GetDocString()
	metadata.GraphDocString = model.Graph.GetDocString()
	if metadata.Properties, err = properties(model.MetadataProps); err != nil {
		return metadata, fmt.Errorf("model: %w", err)
	}

	if metadata.GraphProperties, err = properties(model.Graph.MetadataProps); err != nil {
		return metadata, fmt.Errorf("graph %q: %w", metadata.Name, err)
	}

	for index, wire := range model.Graph.QuantizationAnnotation {
		if wire == nil {
			return metadata, fmt.Errorf("graph annotation[%d]: %w: nil annotation", index, ErrInvalidModel)
		}

		var annotation Annotation
		annotation.Tensor = wire.GetTensorName()
		if annotation.Parameters, err = properties(wire.QuantParameterTensorNames); err != nil {
			return metadata, fmt.Errorf("graph annotation value %q: %w", annotation.Tensor, err)
		}

		metadata.Annotations = append(metadata.Annotations, annotation)
	}
	return metadata, nil
}

func readInfo(wire *onnxpb.ValueInfoProto, required, input bool, limits Limits) (info ValueInfo, err error) {
	info.Name = wire.GetName()
	info.DocString = wire.GetDocString()
	if info.Metadata, err = properties(wire.MetadataProps); err != nil {
		return info, err
	}

	if wire.Type == nil {
		if required {
			return info, fmt.Errorf("%w: graph input/output type required", ErrInvalidModel)
		}

		return info, nil
	}

	info.Denotation = wire.Type.GetDenotation()
	typed, dense := wire.Type.Value.(*onnxpb.TypeProto_TensorType)
	if !dense || typed == nil || typed.TensorType == nil {
		return info, fmt.Errorf("%w: dense tensor type required", ErrInvalidModel)
	}

	value := typed.TensorType
	switch value.GetElemType() {
	case int32(onnxpb.TensorProto_FLOAT):
		info.DataType = tensor.Float32
	case int32(onnxpb.TensorProto_INT64):
		info.DataType = tensor.Int64
	case int32(onnxpb.TensorProto_UNDEFINED):
		return info, fmt.Errorf("%w: tensor element type required", ErrInvalidModel)
	default:
		return info, fmt.Errorf("%w: tensor element type %d", ErrUnsupported, value.GetElemType())
	}

	if value.Shape == nil {
		if input {
			return info, fmt.Errorf("%w: unknown input rank", ErrUnsupported)
		}

		return info, nil
	}

	if len(value.Shape.Dim) > limits.Tensor.MaxRank {
		return info, fmt.Errorf("%w: metadata rank %d", ErrLimit, len(value.Shape.Dim))
	}

	info.ShapeKnown = true
	for axis, wireDimension := range value.Shape.Dim {
		var dimension Dimension
		if dimension, err = readDimension(wireDimension, limits.Tensor.MaxDimension); err != nil {
			return info, fmt.Errorf("axis %d: %w", axis, err)
		}

		if input && axis > 0 && dimension.Kind != FixedDimension {
			return info, fmt.Errorf("%w: dynamic nonleading input axis %d", ErrUnsupported, axis)
		}

		info.Dimensions = append(info.Dimensions, dimension)
	}
	return info, nil
}

func readDimension(wire *onnxpb.TensorShapeProto_Dimension, maximum int64) (dimension Dimension, err error) {
	if wire == nil {
		return dimension, fmt.Errorf("%w: nil dimension", ErrInvalidModel)
	}

	dimension.Denotation = wire.GetDenotation()
	switch value := wire.Value.(type) {
	case nil:
		dimension.Kind = UnknownDimension
	case *onnxpb.TensorShapeProto_Dimension_DimValue:
		if value == nil || value.DimValue < 0 {
			return dimension, fmt.Errorf("%w: negative/invalid fixed dimension", ErrInvalidModel)
		}

		if value.DimValue > maximum {
			return dimension, fmt.Errorf("%w: fixed dimension %d", ErrLimit, value.DimValue)
		}

		dimension.Kind = FixedDimension
		dimension.Value = value.DimValue
	case *onnxpb.TensorShapeProto_Dimension_DimParam:
		if value == nil || value.DimParam == "" {
			return dimension, fmt.Errorf("%w: nonempty dimension symbol required", ErrInvalidModel)
		}

		dimension.Kind = SymbolicDimension
		dimension.Symbol = value.DimParam
	}
	return dimension, nil
}
