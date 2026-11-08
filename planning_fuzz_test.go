package onnx

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func FuzzShapeAttributePlanning(f *testing.F) {
	for _, path := range []string{
		"testdata/m2/models/reshape_zero_infer_int64.onnx",
		"testdata/m2/models/transpose_explicit_float32.onnx",
		"testdata/m2/models/concat_empty_int64.onnx",
		"testdata/m3/models/average_negative_pads.onnx",
		"testdata/m3/models/conv_grouped_bias.onnx",
		"testdata/m1/constant_ints.onnx",
	} {
		data, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}

		f.Add(data)
	}
	// Malformed permutations, attribute discriminators and huge signed axes
	// start from the same real graph exercised by the deterministic M2 suite.
	data, err := os.ReadFile("testdata/m2/models/transpose_explicit_float32.onnx")
	if err != nil {
		f.Fatal(err)
	}

	for _, attribute := range []*onnxpb.AttributeProto{
		{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 0}},
		{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1<<63 - 1)},
		{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(-1 << 63)},
	} {
		wire, parseErr := decode.ParseModel(context.Background(), data, decode.Limits{MaxBytes: 16 << 10, MaxRecursion: 16})
		if parseErr != nil {
			f.Fatal(parseErr)
		}

		wire.Graph.Node[0].Attribute = []*onnxpb.AttributeProto{attribute}
		encoded, marshalErr := proto.Marshal(wire)
		if marshalErr != nil {
			f.Fatal(marshalErr)
		}

		f.Add(encoded)
	}
	f.Fuzz(func(t *testing.T, wire []byte) {
		if len(wire) > 16<<10 {
			t.Skip()
		}

		limits := Limits{
			MaxModelBytes: 16 << 10, MaxLoadBytes: 512 << 10, MaxInitializerBytes: 32 << 10,
			MaxTensorBytes: 16 << 10, MaxLiveBytes: 128 << 10, MaxWorkspaceBytes: 64 << 10,
			MaxNodes: 32, MaxValues: 64, MaxAttributes: 32, MaxNameBytes: 256,
			MaxRank: 8, MaxDimension: 1024, MaxRecursion: 16, MaxSessions: 1,
		}
		model, loadErr := LoadBytes(context.Background(), wire, Options{Limits: limits})
		if loadErr != nil {
			var diagnostic *Error
			if model != nil || !errors.As(loadErr, &diagnostic) || diagnostic.Stage != "load" {
				t.Fatalf("partial model or missing load diagnostic: %v", loadErr)
			}

			return
		}

		session, sessionErr := model.NewSession()
		if sessionErr != nil {
			t.Fatal(sessionErr)
		}

		outputs, runErr := session.Run(context.Background(), nil, RunOptions{})
		if runErr != nil && outputs != nil {
			t.Fatal("failed run published partial outputs")
		}

		if closeErr := session.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}

		if closeErr := model.Close(); closeErr != nil {
			t.Fatal(closeErr)
		}
	})
}
