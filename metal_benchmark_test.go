package onnx

import (
	"context"
	"fmt"
	"testing"
)

// BenchmarkMetalVsCPU compares warm CPU and opt-in Metal execution across
// FP32, FP16, and BF16 resident weights on checked-in inference fixtures
// (mlp.onnx), a 100%-resident FFN graph, a fully resident MLP graph
// (Gemm -> Relu -> Gemm, formerly hybrid in ONNX-01), and a complete
// resident Transformer encoder block.
func BenchmarkMetalVsCPU(b *testing.B) {
	const (
		vocab  int64 = 4096
		hidden int64 = 1024
		ffn    int64 = 4096
	)

	residentGraph := buildResidentTransformerFFNLikeGraph(b, vocab, hidden, ffn)
	hybridGraph := buildHybridMLPONNXGraph(b, hidden, ffn)

	modes := []struct {
		name      string
		execution ExecutionOptions
	}{
		{"CPU", ExecutionOptions{Device: DeviceCPU}},
		{"Metal_FP32", ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat32}},
		{"Metal_FP16", ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat16}},
		{"Metal_BF16", ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightBFloat16}},
	}

	for _, fixture := range inferenceFixtures(b) {
		if fixture.Model.Path != "inference/models/mlp.onnx" {
			continue
		}

		b.Run("Fixture_"+fixture.Name, func(group *testing.B) {
			modelData, readErr := fixture.Model.read()
			if readErr != nil {
				group.Fatal(readErr)
			}

			inputs, inputErr := fixture.readInputs()
			if inputErr != nil {
				group.Fatal(inputErr)
			}

			for _, mode := range modes {
				group.Run(mode.name, func(sub *testing.B) {
					if mode.execution.Device == DeviceMetal && !metalAvailable() {
						sub.Skip("Metal device unavailable in this build or host")
					}

					benchmarkModelMode(sub, modelData, inputs, mode.execution)
				})
			}
		})
	}

	for _, batch := range []int64{1, 32} {
		b.Run(fmt.Sprintf("ResidentFFN_M%d", batch), func(group *testing.B) {
			rows := make([]int64, batch)
			for i := range rows {
				rows[i] = int64(i*7) % vocab
			}

			indices, err := NewInt64([]int64{batch}, rows)
			if err != nil {
				group.Fatal(err)
			}

			residual, err := NewFloat32([]int64{batch, hidden}, deterministicFloats(int(batch*hidden), 101))
			if err != nil {
				group.Fatal(err)
			}

			scale, err := NewFloat32([]int64{batch, ffn}, deterministicFloats(int(batch*ffn), 202))
			if err != nil {
				group.Fatal(err)
			}

			inputs := map[string]*Tensor{
				"indices":  indices,
				"residual": residual,
				"scale":    scale,
			}

			for _, mode := range modes {
				group.Run(mode.name, func(sub *testing.B) {
					if mode.execution.Device == DeviceMetal && !metalAvailable() {
						sub.Skip("Metal device unavailable in this build or host")
					}

					benchmarkModelMode(sub, residentGraph, inputs, mode.execution)
				})
			}
		})

		b.Run(fmt.Sprintf("HybridMLP_M%d", batch), func(group *testing.B) {
			features, err := NewFloat32([]int64{batch, hidden}, deterministicFloats(int(batch*hidden), 303))
			if err != nil {
				group.Fatal(err)
			}

			inputs := map[string]*Tensor{"features": features}
			for _, mode := range modes {
				group.Run(mode.name, func(sub *testing.B) {
					if mode.execution.Device == DeviceMetal && !metalAvailable() {
						sub.Skip("Metal device unavailable in this build or host")
					}

					benchmarkModelMode(sub, hybridGraph, inputs, mode.execution)
				})
			}
		})
	}

	transformerConfigs := []struct {
		name    string
		batch   int64
		seqLen  int64
		heads   int64
		headDim int64
		ffn     int64
	}{
		{"TransformerBlock_B1_S16", 1, 16, 8, 64, 2048},
		{"TransformerBlock_B4_S32", 4, 32, 8, 64, 2048},
	}

	for _, cfg := range transformerConfigs {
		b.Run(cfg.name, func(group *testing.B) {
			graph := buildTransformerEncoderBlockGraph(group, cfg.batch, cfg.seqLen, cfg.heads, cfg.headDim, cfg.ffn)
			hiddenDim := cfg.heads * cfg.headDim
			x, err := NewFloat32([]int64{cfg.batch, cfg.seqLen, hiddenDim}, deterministicFloats(int(cfg.batch*cfg.seqLen*hiddenDim), 1201))
			if err != nil {
				group.Fatal(err)
			}

			inputs := map[string]*Tensor{"x": x}
			for _, mode := range modes {
				group.Run(mode.name, func(sub *testing.B) {
					if mode.execution.Device == DeviceMetal && !metalAvailable() {
						sub.Skip("Metal device unavailable in this build or host")
					}

					benchmarkModelMode(sub, graph, inputs, mode.execution)
				})
			}
		})
	}
}

func benchmarkModelMode(b *testing.B, modelBytesData []byte, inputs map[string]*Tensor, execution ExecutionOptions) {
	b.Helper()
	ctx := context.Background()
	model, err := LoadBytes(ctx, modelBytesData, Options{
		Limits:    largeModelLimits(),
		Execution: execution,
	})
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := model.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})

	session, err := model.NewSession()
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() {
		if closeErr := session.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	})

	// Warmup outside timed loop.
	if _, err = session.Run(ctx, inputs, RunOptions{}); err != nil {
		b.Fatal(err)
	}

	modelStats := model.GPUStats()
	sessStats, err := session.GPUStats()
	if err != nil {
		b.Fatal(err)
	}

	b.ReportMetric(float64(modelStats.WeightBytes), "gpu_wt_B")
	b.ReportMetric(float64(sessStats.ActivationBytes), "gpu_act_B")
	b.ReportMetric(float64(sessStats.ScopeSubmits), "submits/op")
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err = session.Run(ctx, inputs, RunOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}
