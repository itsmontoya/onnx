package graph

import (
	"context"
	"reflect"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestConstantAndMetadataOwnership(t *testing.T) {
	model := graphFixture()
	model.ProducerName = proto.String("producer")
	model.ProducerVersion = proto.String("1.0")
	model.Domain = proto.String("descriptive.model.domain")
	model.ModelVersion = proto.Int64(42)
	model.DocString = proto.String("model docs")
	model.MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("model"), Value: proto.String("value")}}
	model.Graph.DocString = proto.String("graph docs")
	model.Graph.MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("graph"), Value: proto.String("value")}}
	model.Graph.QuantizationAnnotation = []*onnxpb.TensorAnnotation{{TensorName: proto.String("x"), QuantParameterTensorNames: []*onnxpb.StringStringEntryProto{{Key: proto.String("SCALE_TENSOR"), Value: proto.String("w")}}}}
	model.Graph.Input[0].DocString = proto.String("input docs")
	model.Graph.Input[0].MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("input"), Value: proto.String("value")}}
	model.Graph.Input[0].Type.Denotation = proto.String("TENSOR")
	model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Denotation = proto.String("DATA_BATCH")
	model.Graph.Node[0].MetadataProps = []*onnxpb.StringStringEntryProto{{Key: proto.String("node"), Value: proto.String("value")}}
	model.Graph.Initializer = []*onnxpb.TensorProto{{Name: proto.String("w"), DataType: proto.Int32(1), Dims: []int64{2}, RawData: []byte{0, 0, 0x80, 0x3f, 0, 0, 0, 0x40}, DocString: proto.String("weight docs"), MetadataProps: []*onnxpb.StringStringEntryProto{{Key: proto.String("weight"), Value: proto.String("value")}}}}
	compiled, err := Build(context.Background(), model, encodedLength(t, model), Limits{})
	if err != nil {
		t.Fatal(err)
	}

	model.Graph.Input[0].Type.GetTensorType().Shape.Dim[0].Value.(*onnxpb.TensorShapeProto_Dimension_DimParam).DimParam = "mutated"
	model.Graph.Input[0].MetadataProps[0].Value = proto.String("mutated")
	model.Graph.Node[0].Input[0] = "mutated"
	model.Graph.Node[0].MetadataProps[0].Value = proto.String("mutated")
	model.Graph.Initializer[0].Dims[0] = 1
	clear(model.Graph.Initializer[0].RawData)
	model.Graph.Initializer[0].MetadataProps[0].Value = proto.String("mutated")
	model.Graph.QuantizationAnnotation[0].QuantParameterTensorNames[0].Value = proto.String("mutated")
	model.MetadataProps[0].Value = proto.String("mutated")
	proto.Reset(model)
	metadata := compiled.Metadata()
	if metadata.Name != "test" || metadata.Producer != "producer" || metadata.ProducerVersion != "1.0" || metadata.ModelDomain != "descriptive.model.domain" || metadata.ModelVersion != 42 || metadata.DocString != "model docs" || metadata.GraphDocString != "graph docs" || metadata.Properties[0].Value != "value" || metadata.Annotations[0].Parameters[0].Value != "w" {
		t.Fatalf("model snapshot changed: %v", metadata)
	}

	inputs := compiled.Inputs()
	if inputs[0].Dimensions[0].Symbol != "N" || inputs[0].Dimensions[0].Denotation != "DATA_BATCH" || inputs[0].Denotation != "TENSOR" || inputs[0].DocString != "input docs" || inputs[0].Metadata[0].Value != "value" {
		t.Fatalf("input snapshot changed: %v", inputs)
	}

	metadata.Properties[0].Value = "changed"
	metadata.GraphProperties[0].Value = "changed"
	metadata.Annotations[0].Parameters[0].Value = "changed"
	inputs[0].Dimensions[0].Symbol = "changed"
	inputs[0].Metadata[0].Value = "changed"
	outputs := compiled.Outputs()
	outputs[0].Dimensions[0].Symbol = "changed"
	values := compiled.Values()
	values[0].Declarations[0].Dimensions[0].Symbol = "changed"
	value, found := compiled.Value("w")
	if !found || value.Default == nil {
		t.Fatal("initializer missing")
	}

	constant := value.Default
	if constant.Name() != "w" || constant.DocString() != "weight docs" || constant.DataType() != tensor.Float32 || constant.Rank() != 1 || constant.ByteSize() != 8 || !reflect.DeepEqual(constant.Shape(), []int64{2}) || constant.Metadata()[0].Value != "value" {
		t.Fatal("constant metadata changed")
	}

	floats, err := constant.Float32Values()
	if err != nil || !reflect.DeepEqual(floats, []float32{1, 2}) {
		t.Fatalf("constant data changed: %v, %v", floats, err)
	}

	floats[0] = 99
	shape := constant.Shape()
	shape[0] = 99
	constantMetadata := constant.Metadata()
	constantMetadata[0].Value = "changed"
	cloned, err := constant.Clone()
	if err != nil {
		t.Fatal(err)
	}

	zeros, err := tensor.New(tensor.Float32, []int64{2})
	if err != nil {
		t.Fatal(err)
	}

	if err = tensor.AddInto(zeros, zeros, cloned); err != nil {
		t.Fatal(err)
	}

	if floats, err = constant.Float32Values(); err != nil || !reflect.DeepEqual(floats, []float32{1, 2}) {
		t.Fatalf("clone/accessor mutation changed constant: %v, %v", floats, err)
	}

	if compiled.Metadata().Properties[0].Value != "value" || compiled.Metadata().GraphProperties[0].Value != "value" || compiled.Metadata().Annotations[0].Parameters[0].Value != "w" || compiled.Inputs()[0].Dimensions[0].Symbol != "N" || compiled.Outputs()[0].Dimensions[0].Symbol != "N" || compiled.Values()[0].Declarations[0].Dimensions[0].Symbol != "N" || constant.Metadata()[0].Value != "value" {
		t.Fatal("accessor mutation changed compiled records")
	}
}
