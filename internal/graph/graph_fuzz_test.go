package graph

import (
	"context"
	"os"
	"testing"

	"github.com/itsmontoya/onnx/internal/decode"
	"google.golang.org/protobuf/proto"
)

func FuzzGraphReferences(f *testing.F) {
	for _, path := range []string{"../../testdata/profile/identity.onnx", "../../testdata/sessions/branch_defaults.onnx"} {
		data, err := os.ReadFile(path)
		if err != nil {
			f.Fatal(err)
		}

		f.Add(data)
	}
	for _, reference := range []string{"absent", "y", ""} {
		wire := graphFixture()
		wire.Graph.Node[0].Input[0] = reference
		data, err := proto.Marshal(wire)
		if err != nil {
			f.Fatal(err)
		}

		f.Add(data)
	}
	f.Add([]byte{0x3a, 0x02, 0x08})
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 16<<10 {
			t.Skip()
		}

		ctx := context.Background()
		wire, err := decode.ParseModel(ctx, data, decode.Limits{MaxBytes: 16 << 10, MaxRecursion: 16})
		if err != nil {
			return
		}

		limits := Limits{
			Tensor: decode.TensorLimits{
				MaxLoadBytes: 256 << 10, MaxInitializerBytes: 32 << 10, MaxTensorBytes: 16 << 10,
				MaxRank: 8, MaxDimension: 1024, MaxRecursion: 16,
			},
			MaxNodes: 32, MaxValues: 64, MaxAttributes: 32, MaxNameBytes: 256,
		}
		first, firstErr := Build(ctx, wire, int64(len(data)), limits)
		second, secondErr := Build(ctx, wire, int64(len(data)), limits)
		if firstErr != nil || secondErr != nil {
			if first != nil || second != nil || firstErr == nil || secondErr == nil || firstErr.Error() != secondErr.Error() {
				t.Fatalf("nondeterministic failure or partial graph: %v, %v", firstErr, secondErr)
			}

			return
		}

		if load, owned := first.Usage(); load > limits.Tensor.MaxLoadBytes || owned > limits.Tensor.MaxInitializerBytes {
			t.Fatalf("graph exceeded finite payload limits: %d, %d", load, owned)
		}
	})
}
