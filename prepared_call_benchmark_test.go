package onnx

import (
	"context"
	"os"
	"testing"

	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

// BenchmarkPreparedCall compares eligible and deferred calls, including changing
// valid signatures. The control uses the same plan with preparation disabled.
func BenchmarkPreparedCall(b *testing.B) {
	for _, mode := range []string{"Static", "Dynamic", "Alternating"} {
		for _, prepared := range []bool{false, true} {
			name := "Control/" + mode
			if prepared {
				name = "Prepared/" + mode
			}

			b.Run(name, func(call *testing.B) {
				var (
					wire    onnxpb.ModelProto
					data    []byte
					model   *Model
					session *Session
					err     error
				)

				if data, err = os.ReadFile("testdata/m1/add_matching.onnx"); err != nil {
					call.Fatal(err)
				}

				if err = proto.Unmarshal(data, &wire); err != nil {
					call.Fatal(err)
				}

				if mode != "Static" {
					for _, input := range wire.Graph.Input {
						input.Type.GetTensorType().Shape.Dim[0].Value = nil
					}
					wire.Graph.Output[0].Type.GetTensorType().Shape = nil
				}
				if data, err = proto.Marshal(&wire); err != nil {
					call.Fatal(err)
				}

				if model, err = LoadBytes(context.Background(), data, Options{}); err != nil {
					call.Fatal(err)
				}

				call.Cleanup(func() { _ = model.Close() })
				if !prepared {
					model.plan.calls = nil
				}

				if session, err = model.NewSession(); err != nil {
					call.Fatal(err)
				}

				call.Cleanup(func() { _ = session.Close() })
				signatures := make([]map[string]*Tensor, 2)
				for index := range signatures {
					signatures[index] = make(map[string]*Tensor)
					for _, input := range wire.Graph.Input {
						shape := make([]int64, len(input.Type.GetTensorType().Shape.Dim))
						var elements int64 = 1
						for axis, dimension := range input.Type.GetTensorType().Shape.Dim {
							shape[axis] = dimension.GetDimValue()
							if mode != "Static" && axis == 0 {
								shape[axis] = int64(index + 1)
							}

							elements *= shape[axis]
						}
						if signatures[index][input.GetName()], err = NewFloat32(shape, make([]float32, elements)); err != nil {
							call.Fatal(err)
						}
					}
				}
				call.ReportAllocs()
				call.ResetTimer()
				for iteration := 0; iteration < call.N; iteration++ {
					index := 0
					if mode == "Alternating" {
						index = iteration % len(signatures)
					}

					if _, err = session.Run(context.Background(), signatures[index], RunOptions{}); err != nil {
						call.Fatal(err)
					}
				}
			})
		}
	}
}
