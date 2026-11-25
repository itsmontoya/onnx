package onnx

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/gpu"
	"github.com/itsmontoya/onnx/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func metalAvailable() (ok bool) {
	device, err := gpu.OpenDefault()
	if err != nil {
		return false
	}

	_ = device.Close()
	return true
}

func requireMetal(t *testing.T) {
	t.Helper()
	if !metalAvailable() {
		t.Skip("Metal device unavailable in this build or host")
	}
}

func TestExecutionOptionsValidation(t *testing.T) {
	if !DeviceCPU.Valid() || !DeviceMetal.Valid() || Device(99).Valid() {
		t.Fatal("Device.Valid mismatch")
	}

	if DeviceCPU.String() != "cpu" || DeviceMetal.String() != "metal" || Device(99).String() != "invalid" {
		t.Fatal("Device.String mismatch")
	}

	if !WeightFloat32.Valid() || !WeightFloat16.Valid() || !WeightBFloat16.Valid() || WeightPrecision(99).Valid() {
		t.Fatal("WeightPrecision.Valid mismatch")
	}

	if WeightFloat32.String() != "float32" || WeightFloat16.String() != "float16" || WeightBFloat16.String() != "bfloat16" || WeightPrecision(99).String() != "invalid" {
		t.Fatal("WeightPrecision.String mismatch")
	}

	data := modelFixture(t, "inference/models/mlp")
	invalidOptions := []struct {
		name    string
		options ExecutionOptions
	}{
		{"invalid device", ExecutionOptions{Device: Device(42)}},
		{"invalid precision", ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightPrecision(42)}},
		{"cpu with fp16 weights", ExecutionOptions{Device: DeviceCPU, WeightPrecision: WeightFloat16}},
		{"cpu with bf16 weights", ExecutionOptions{Device: DeviceCPU, WeightPrecision: WeightBFloat16}},
	}

	for _, tt := range invalidOptions {
		t.Run(tt.name, func(t *testing.T) {
			model, err := LoadBytes(context.Background(), data, Options{Execution: tt.options})
			if model != nil || !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("expected ErrInvalidArgument, got model=%v err=%v", model, err)
			}
		})
	}
}

func TestMetalUnavailableReturnsErrUnavailable(t *testing.T) {
	if metalAvailable() {
		t.Skip("Metal is available in this build; unavailable path tested on non-metal builds")
	}

	data := modelFixture(t, "inference/models/mlp")
	model, err := LoadBytes(context.Background(), data, Options{
		Execution: ExecutionOptions{Device: DeviceMetal},
	})
	if model != nil || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("expected ErrUnavailable, got model=%v err=%v", model, err)
	}

	var diagnostic *Error
	if !errors.As(err, &diagnostic) || diagnostic.Kind != ErrUnavailable || diagnostic.Stage != "load" {
		t.Fatalf("expected load diagnostic with ErrUnavailable, got %+v", diagnostic)
	}
}

func TestCPUModelGPUStatsZero(t *testing.T) {
	var (
		nilModel   *Model
		zeroModel  Model
		nilSession *Session
		zeroSess   Session
	)

	if nilModel.GPUStats().Active || zeroModel.GPUStats().Active {
		t.Fatal("nil/zero model reported active GPU stats")
	}

	if _, err := nilSession.GPUStats(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil session GPUStats: %v", err)
	}

	if _, err := zeroSess.GPUStats(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("zero session GPUStats: %v", err)
	}

	model := loadedModel(t, modelFixture(t, "inference/models/mlp"), Options{})
	if stats := model.GPUStats(); stats.Active || stats.WeightTensors != 0 || stats.WeightBytes != 0 {
		t.Fatalf("CPU model reported GPU stats: %+v", stats)
	}

	session := modelSession(t, model)
	stats, err := session.GPUStats()
	if err != nil || stats.Active || stats.DeviceOps != 0 {
		t.Fatalf("CPU session reported GPU stats: %+v, %v", stats, err)
	}
}

func TestMetalInferenceFixturesParity(t *testing.T) {
	requireMetal(t)
	cases := readInferenceCases(t)
	precisions := []struct {
		precision WeightPrecision
		atol      float64
		rtol      float64
	}{
		{WeightFloat32, 1e-5, 1e-5},
		{WeightFloat16, 2e-3, 2e-3},
		{WeightBFloat16, 1.5e-2, 1.5e-2},
	}

	for _, prec := range precisions {
		t.Run(prec.precision.String(), func(t *testing.T) {
			for _, fixture := range cases {
				if fixture.Model.Path == "inference/models/shape.onnx" {
					continue
				}

				t.Run(fixture.Name, func(t *testing.T) {
					data, err := fixture.Model.read()
					if err != nil {
						t.Fatal(err)
					}

					inputs, err := fixture.readInputs()
					if err != nil {
						t.Fatal(err)
					}

					cpuModel := loadedModel(t, data, Options{})
					cpuSession := modelSession(t, cpuModel)
					cpuOutputs, err := cpuSession.Run(context.Background(), inputs, RunOptions{})
					if err != nil {
						t.Fatal(err)
					}

					gpuModel := loadedModel(t, data, Options{
						Execution: ExecutionOptions{
							Device:          DeviceMetal,
							WeightPrecision: prec.precision,
						},
					})
					gpuStats := gpuModel.GPUStats()
					if !gpuStats.Active || gpuStats.WeightTensors == 0 || gpuStats.WeightBytes == 0 {
						t.Fatalf("expected active GPU weight residency, got %+v", gpuStats)
					}

					if prec.precision != WeightFloat32 && gpuStats.ReducedPrecisionTensors == 0 {
						t.Fatalf("expected reduced-precision weights for %s, got %+v", prec.precision, gpuStats)
					}

					gpuSession := modelSession(t, gpuModel)
					gpuOutputs, err := gpuSession.Run(context.Background(), inputs, RunOptions{})
					if err != nil {
						t.Fatal(err)
					}

					sessStats, err := gpuSession.GPUStats()
					if err != nil {
						t.Fatal(err)
					}

					if !sessStats.Active || sessStats.DeviceOps == 0 || sessStats.ScopeSubmits == 0 {
						t.Fatalf("expected Metal device execution, got %+v", sessStats)
					}

					if prec.precision == WeightFloat32 {
						if err = fixture.compareOutputs(gpuOutputs); err != nil {
							t.Fatalf("Metal FP32 vs golden reference: %v", err)
						}
					}

					dist := compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, prec.atol, prec.rtol)
					t.Logf("%s (%s): deviceOps=%d fallbackOps=%d submits=%d weightBytes=%d/%d maxAbs=%.3e p99=%.3e",
						fixture.Name, prec.precision, sessStats.DeviceOps, sessStats.HostFallbackOps,
						sessStats.ScopeSubmits, gpuStats.WeightBytes, gpuStats.FP32EquivalentWeightBytes,
						dist.maxAbs, dist.p99)
				})
			}
		})
	}
}

func TestMetalPyTorchExportAndExternalWeightsParity(t *testing.T) {
	requireMetal(t)
	t.Run("pytorch_mlp_opset17", func(t *testing.T) {
		modelBytesData, err := loadFixtureFile("testdata/profiles/exports/pytorch_mlp.onnx")
		if err != nil {
			t.Fatal(err)
		}

		input := floatTensor(t, []int64{3, 2}, []float32{0.5, -1.25, 2.0, 0.75, -0.5, 1.5})
		inputs := map[string]*Tensor{"features": input}

		cpuModel := loadedModel(t, modelBytesData, Options{})
		cpuOutputs, err := modelSession(t, cpuModel).Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		for _, prec := range []struct {
			p    WeightPrecision
			atol float64
		}{
			{WeightFloat32, 1e-5},
			{WeightFloat16, 3e-3},
			{WeightBFloat16, 2e-2},
		} {
			gpuModel := loadedModel(t, modelBytesData, Options{
				Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: prec.p},
			})
			gpuSess := modelSession(t, gpuModel)
			gpuOutputs, runErr := gpuSess.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatal(runErr)
			}

			sessStats, statsErr := gpuSess.GPUStats()
			if statsErr != nil || sessStats.DeviceOps == 0 {
				t.Fatalf("expected Metal ops on PyTorch export: stats=%+v err=%v", sessStats, statsErr)
			}

			compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, prec.atol, prec.atol)
		}
	})

	t.Run("external_mlp", func(t *testing.T) {
		resolver, err := NewDirectoryResolver("testdata/external/models")
		if err != nil {
			t.Fatal(err)
		}

		defer func() {
			if closeErr := resolver.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}()

		input := floatTensor(t, []int64{2, 2}, []float32{1.0, 2.0, -0.5, 1.5})
		inputs := map[string]*Tensor{"features": input}

		cpuModel, err := LoadFile(context.Background(), "testdata/external/models/mlp.onnx", Options{ExternalData: resolver})
		if err != nil {
			t.Fatal(err)
		}

		defer func() { _ = cpuModel.Close() }()
		cpuOutputs, err := modelSession(t, cpuModel).Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		gpuModel, err := LoadFile(context.Background(), "testdata/external/models/mlp.onnx", Options{
			ExternalData: resolver,
			Execution:    ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat32},
		})
		if err != nil {
			t.Fatal(err)
		}

		defer func() { _ = gpuModel.Close() }()
		gpuOutputs, err := modelSession(t, gpuModel).Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, 1e-5, 1e-5)
	})
}

func TestMetalOperatorsAndResidentChainParity(t *testing.T) {
	requireMetal(t)
	graphBytes := buildResidentTransformerFFNLikeGraph(t, 64, 128, 256)
	indices := intTensor(t, []int64{4}, []int64{0, 17, -1, 63})
	residualVals := deterministicFloats(4*128, 101)
	gateVals := deterministicFloats(4*256, 202)
	inputs := map[string]*Tensor{
		"indices":  indices,
		"residual": floatTensor(t, []int64{4, 128}, residualVals),
		"scale":    floatTensor(t, []int64{4, 256}, gateVals),
	}

	cpuModel := loadedModel(t, graphBytes, Options{})
	cpuOutputs, err := modelSession(t, cpuModel).Run(context.Background(), inputs, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	precisions := []struct {
		precision WeightPrecision
		atol      float64
		rtol      float64
	}{
		{WeightFloat32, 1e-4, 1e-4},
		{WeightFloat16, 1e-2, 1e-2},
		{WeightBFloat16, 5e-2, 5e-2},
	}

	for _, prec := range precisions {
		t.Run(prec.precision.String(), func(t *testing.T) {
			gpuModel := loadedModel(t, graphBytes, Options{
				Execution: ExecutionOptions{
					Device:          DeviceMetal,
					WeightPrecision: prec.precision,
				},
			})
			modelStats := gpuModel.GPUStats()
			if !modelStats.Active || modelStats.WeightTensors != 5 {
				t.Fatalf("unexpected model GPUStats: %+v", modelStats)
			}

			if prec.precision != WeightFloat32 {
				if modelStats.ReducedPrecisionTensors != 3 {
					t.Fatalf("expected 3 reduced-precision weights (embed, w1, w2), got %+v", modelStats)
				}

				if modelStats.WeightBytes >= modelStats.FP32EquivalentWeightBytes {
					t.Fatalf("expected reduced weight bytes, got %+v", modelStats)
				}
			}

			gpuSess := modelSession(t, gpuModel)
			gpuOutputs, runErr := gpuSess.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatal(runErr)
			}

			sessStats, statsErr := gpuSess.GPUStats()
			if statsErr != nil {
				t.Fatal(statsErr)
			}

			if sessStats.DeviceOps != 6 || sessStats.HostFallbackOps != 0 || sessStats.ScopeSubmits != 1 || sessStats.HostDownloads != 1 {
				t.Fatalf("expected single-scope resident chain (6 device ops, 0 fallback, 1 submit, 1 download), got %+v", sessStats)
			}

			dist := compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, prec.atol, prec.rtol)
			t.Logf("resident_chain (%s): weightBytes=%d (fp32=%d) actBufs=%d actBytes=%d p50=%.3e p95=%.3e p99=%.3e max=%.3e",
				prec.precision, modelStats.WeightBytes, modelStats.FP32EquivalentWeightBytes,
				sessStats.ActivationBuffers, sessStats.ActivationBytes,
				dist.p50, dist.p95, dist.p99, dist.maxAbs)
		})
	}
}

func TestMetalLifecycleConcurrencyAndDynamicBatch(t *testing.T) {
	requireMetal(t)
	graphBytes := buildResidentTransformerFFNLikeGraph(t, 32, 64, 128)
	model, err := LoadBytes(context.Background(), graphBytes, Options{
		Limits:    Limits{MaxSessions: 4},
		Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat16},
	})
	if err != nil {
		t.Fatal(err)
	}

	// Create sessions, then close Model BEFORE sessions finish (ownership contract).
	sessions := make([]*Session, 3)
	for i := range sessions {
		if sessions[i], err = model.NewSession(); err != nil {
			t.Fatal(err)
		}
	}

	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	// Model.Close must be idempotent.
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}

	cpuModel := loadedModel(t, graphBytes, Options{})
	cpuSess := modelSession(t, cpuModel)

	errs := make(chan error, len(sessions))
	for i, sess := range sessions {
		go func(worker int, s *Session) {
			defer func() {
				if closeErr := s.Close(); closeErr != nil {
					errs <- closeErr
				}
			}()

			// Dynamic batch sizes exercise metalCache reallocation across runs.
			for _, batch := range []int64{1, 5, 2} {
				rows := make([]int64, batch)
				for r := int64(0); r < batch; r++ {
					rows[r] = (r + int64(worker)*3) % 32
				}

				inputs := map[string]*Tensor{
					"indices":  intTensor(t, []int64{batch}, rows),
					"residual": floatTensor(t, []int64{batch, 64}, deterministicFloats(int(batch*64), worker+1)),
					"scale":    floatTensor(t, []int64{batch, 128}, deterministicFloats(int(batch*128), worker+10)),
				}

				if _, runErr := s.Run(context.Background(), inputs, RunOptions{}); runErr != nil {
					errs <- runErr
					return
				}
			}

			errs <- nil
		}(i, sess)
	}

	for range sessions {
		if workerErr := <-errs; workerErr != nil {
			t.Fatal(workerErr)
		}
	}

	// Verify CPU reference still works independently.
	checkInputs := map[string]*Tensor{
		"indices":  intTensor(t, []int64{1}, []int64{0}),
		"residual": floatTensor(t, []int64{1, 64}, deterministicFloats(64, 1)),
		"scale":    floatTensor(t, []int64{1, 128}, deterministicFloats(128, 10)),
	}
	if _, err = cpuSess.Run(context.Background(), checkInputs, RunOptions{}); err != nil {
		t.Fatal(err)
	}
}

func TestMetalCancellationAndOutOfBoundsGatherRecovery(t *testing.T) {
	requireMetal(t)
	graphBytes := buildResidentTransformerFFNLikeGraph(t, 16, 32, 64)
	model := loadedModel(t, graphBytes, Options{
		Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat32},
	})
	sess := modelSession(t, model)

	validInputs := map[string]*Tensor{
		"indices":  intTensor(t, []int64{2}, []int64{0, -1}),
		"residual": floatTensor(t, []int64{2, 32}, deterministicFloats(64, 7)),
		"scale":    floatTensor(t, []int64{2, 64}, deterministicFloats(128, 9)),
	}

	// Canceled context before Run.
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	if outputs, err := sess.Run(canceledCtx, validInputs, RunOptions{}); outputs != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got outputs=%v err=%v", outputs, err)
	}

	// Out-of-bounds Gather index must fail with ErrInvalidModel.
	badInputs := map[string]*Tensor{
		"indices":  intTensor(t, []int64{2}, []int64{0, 16}),
		"residual": validInputs["residual"],
		"scale":    validInputs["scale"],
	}
	if outputs, err := sess.Run(context.Background(), badInputs, RunOptions{}); outputs != nil || !errors.Is(err, ErrInvalidModel) {
		t.Fatalf("expected ErrInvalidModel for OOB Gather, got outputs=%v err=%v", outputs, err)
	}

	// Session must recover cleanly after both failures.
	if outputs, err := sess.Run(context.Background(), validInputs, RunOptions{}); err != nil || len(outputs) != 1 {
		t.Fatalf("session recovery failed: outputs=%v err=%v", outputs, err)
	}
}

type errorDistribution struct {
	count  int
	p50    float64
	p95    float64
	p99    float64
	maxAbs float64
}

func compareOutputsAgainstCPU(t *testing.T, cpuOutputs, gpuOutputs []NamedTensor, atol, rtol float64) (dist errorDistribution) {
	t.Helper()
	if len(cpuOutputs) != len(gpuOutputs) {
		t.Fatalf("output count mismatch: cpu=%d gpu=%d", len(cpuOutputs), len(gpuOutputs))
	}

	var diffs []float64
	for index := range cpuOutputs {
		if cpuOutputs[index].Name != gpuOutputs[index].Name {
			t.Fatalf("output[%d] name mismatch: cpu=%q gpu=%q", index, cpuOutputs[index].Name, gpuOutputs[index].Name)
		}

		if !slices.Equal(cpuOutputs[index].Tensor.Shape(), gpuOutputs[index].Tensor.Shape()) {
			t.Fatalf("output[%d] shape mismatch: cpu=%v gpu=%v", index, cpuOutputs[index].Tensor.Shape(), gpuOutputs[index].Tensor.Shape())
		}

		cpuVals, err := cpuOutputs[index].Tensor.Float32Values()
		if err != nil {
			t.Fatal(err)
		}

		gpuVals, err := gpuOutputs[index].Tensor.Float32Values()
		if err != nil {
			t.Fatal(err)
		}

		for elem := range cpuVals {
			diff := math.Abs(float64(gpuVals[elem]) - float64(cpuVals[elem]))
			diffs = append(diffs, diff)
			limit := atol + rtol*math.Abs(float64(cpuVals[elem]))
			if math.IsNaN(diff) || diff > limit {
				t.Fatalf("output %q[%d] mismatch: cpu=%g gpu=%g absDiff=%g limit=%g",
					cpuOutputs[index].Name, elem, cpuVals[elem], gpuVals[elem], diff, limit)
			}
		}
	}

	return summarizeErrors(diffs)
}

func summarizeErrors(diffs []float64) (dist errorDistribution) {
	if len(diffs) == 0 {
		return dist
	}

	sorted := slices.Clone(diffs)
	slices.Sort(sorted)
	dist.count = len(sorted)
	dist.p50 = sorted[(len(sorted)-1)*50/100]
	dist.p95 = sorted[(len(sorted)-1)*95/100]
	dist.p99 = sorted[(len(sorted)-1)*99/100]
	dist.maxAbs = sorted[len(sorted)-1]
	return dist
}

func deterministicFloats(count, seed int) (out []float32) {
	out = make([]float32, count)
	state := uint64(seed*6364136223846793005 + 1442695040888963407)
	for i := range out {
		state = state*6364136223846793005 + 1442695040888963407
		unit := float64(state>>40) / float64(uint64(1)<<24)
		out[i] = float32((unit*2.0 - 1.0) * 0.25)
	}

	return out
}

func float32RawBytes(values []float32) (raw []byte) {
	raw = make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}

	return raw
}

func makeFloatInitializer(name string, shape []int64, values []float32) (out *onnxpb.TensorProto) {
	return &onnxpb.TensorProto{
		Name:     proto.String(name),
		DataType: proto.Int32(int32(onnxpb.TensorProto_FLOAT)),
		Dims:     shape,
		RawData:  float32RawBytes(values),
	}
}

func makeDynamicFloatValueInfo(name, batchSymbol string, cols int64) (out *onnxpb.ValueInfoProto) {
	return &onnxpb.ValueInfoProto{
		Name: proto.String(name),
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: proto.Int32(int32(onnxpb.TensorProto_FLOAT)),
					Shape: &onnxpb.TensorShapeProto{
						Dim: []*onnxpb.TensorShapeProto_Dimension{
							{Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: batchSymbol}},
							{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: cols}},
						},
					},
				},
			},
		},
	}
}

func makeDynamicInt64VectorValueInfo(name, batchSymbol string) (out *onnxpb.ValueInfoProto) {
	return &onnxpb.ValueInfoProto{
		Name: proto.String(name),
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: proto.Int32(int32(onnxpb.TensorProto_INT64)),
					Shape: &onnxpb.TensorShapeProto{
						Dim: []*onnxpb.TensorShapeProto_Dimension{
							{Value: &onnxpb.TensorShapeProto_Dimension_DimParam{DimParam: batchSymbol}},
						},
					},
				},
			},
		},
	}
}

// buildResidentTransformerFFNLikeGraph constructs a valid IR-8 opset-13 ONNX
// model exercising Gather -> Gemm(transB=1, bias) -> Mul -> Gemm(transB=1, bias) -> Add -> Identity.
func buildResidentTransformerFFNLikeGraph(tb testing.TB, vocab, hidden, ffn int64) (data []byte) {
	tb.Helper()
	model := &onnxpb.ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("onnx-gpu-test"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(13)},
		},
		Graph: &onnxpb.GraphProto{
			Name: proto.String("resident_ffn_like"),
			Input: []*onnxpb.ValueInfoProto{
				makeDynamicInt64VectorValueInfo("indices", "batch"),
				makeDynamicFloatValueInfo("residual", "batch", hidden),
				makeDynamicFloatValueInfo("scale", "batch", ffn),
			},
			Output: []*onnxpb.ValueInfoProto{
				makeDynamicFloatValueInfo("output", "batch", hidden),
			},
			Initializer: []*onnxpb.TensorProto{
				makeFloatInitializer("embed_table", []int64{vocab, hidden}, deterministicFloats(int(vocab*hidden), 11)),
				makeFloatInitializer("w1", []int64{ffn, hidden}, deterministicFloats(int(ffn*hidden), 22)),
				makeFloatInitializer("b1", []int64{ffn}, deterministicFloats(int(ffn), 33)),
				makeFloatInitializer("w2", []int64{hidden, ffn}, deterministicFloats(int(hidden*ffn), 44)),
				makeFloatInitializer("b2", []int64{hidden}, deterministicFloats(int(hidden), 55)),
			},
			Node: []*onnxpb.NodeProto{
				{
					Name:   proto.String("gather_embed"),
					OpType: proto.String("Gather"),
					Input:  []string{"embed_table", "indices"},
					Output: []string{"embedded"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(0)},
					},
				},
				{
					Name:   proto.String("up_proj"),
					OpType: proto.String("Gemm"),
					Input:  []string{"embedded", "w1", "b1"},
					Output: []string{"up"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("transB"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)},
					},
				},
				{
					Name:   proto.String("gate_mul"),
					OpType: proto.String("Mul"),
					Input:  []string{"up", "scale"},
					Output: []string{"gated"},
				},
				{
					Name:   proto.String("down_proj"),
					OpType: proto.String("Gemm"),
					Input:  []string{"gated", "w2", "b2"},
					Output: []string{"projected"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("transB"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)},
					},
				},
				{
					Name:   proto.String("residual_add"),
					OpType: proto.String("Add"),
					Input:  []string{"projected", "residual"},
					Output: []string{"summed"},
				},
				{
					Name:   proto.String("out_identity"),
					OpType: proto.String("Identity"),
					Input:  []string{"summed"},
					Output: []string{"output"},
				},
			},
		},
	}

	var err error
	if data, err = proto.Marshal(model); err != nil {
		tb.Fatal(err)
	}

	return data
}

// buildHybridMLPONNXGraph constructs a valid IR-8 opset-13 ONNX model with
// Gemm(transB=1) -> Relu (CPU fallback) -> Gemm(transB=1) to measure hybrid boundary cost.
func buildHybridMLPONNXGraph(tb testing.TB, hidden, ffn int64) (data []byte) {
	tb.Helper()
	model := &onnxpb.ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("onnx-gpu-test"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(13)},
		},
		Graph: &onnxpb.GraphProto{
			Name: proto.String("hybrid_mlp"),
			Input: []*onnxpb.ValueInfoProto{
				makeDynamicFloatValueInfo("features", "batch", hidden),
			},
			Output: []*onnxpb.ValueInfoProto{
				makeDynamicFloatValueInfo("output", "batch", hidden),
			},
			Initializer: []*onnxpb.TensorProto{
				makeFloatInitializer("w1", []int64{ffn, hidden}, deterministicFloats(int(ffn*hidden), 61)),
				makeFloatInitializer("b1", []int64{ffn}, deterministicFloats(int(ffn), 62)),
				makeFloatInitializer("w2", []int64{hidden, ffn}, deterministicFloats(int(hidden*ffn), 63)),
				makeFloatInitializer("b2", []int64{hidden}, deterministicFloats(int(hidden), 64)),
			},
			Node: []*onnxpb.NodeProto{
				{
					Name:   proto.String("fc1"),
					OpType: proto.String("Gemm"),
					Input:  []string{"features", "w1", "b1"},
					Output: []string{"h1"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("transB"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)},
					},
				},
				{
					Name:   proto.String("relu"),
					OpType: proto.String("Relu"),
					Input:  []string{"h1"},
					Output: []string{"a1"},
				},
				{
					Name:   proto.String("fc2"),
					OpType: proto.String("Gemm"),
					Input:  []string{"a1", "w2", "b2"},
					Output: []string{"output"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("transB"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)},
					},
				},
			},
		},
	}

	var err error
	if data, err = proto.Marshal(model); err != nil {
		tb.Fatal(err)
	}

	return data
}

func loadFixtureFile(path string) (data []byte, err error) {
	if data, err = os.ReadFile(path); err != nil {
		return nil, fmt.Errorf("read %q: %w", path, err)
	}

	return data, nil
}

func largeModelLimits() (out Limits) {
	return Limits{
		MaxModelBytes:       128 << 20,
		MaxLoadBytes:        2 << 30,
		MaxInitializerBytes: 256 << 20,
		MaxTensorBytes:      128 << 20,
		MaxLiveBytes:        512 << 20,
		MaxWorkspaceBytes:   256 << 20,
	}
}

func TestMetalUnitOperatorsParity(t *testing.T) {
	requireMetal(t)
	tests := []struct {
		name      string
		fixture   string
		deviceOps int
	}{
		// Non-unit alpha/beta Gemm falls back to CPU because neuralnetwork/gpu lacks scalar scale.
		{"gemm_00_full_fallback", "m1/gemm_00_full", 0},
		{"gemm_01_row_fallback", "m1/gemm_01_row", 0},
		{"gemm_beta_zero", "m1/gemm_beta_zero", 1},
		{"gemm_defaults", "m1/gemm_defaults", 1},
		{"add_matching", "m1/add_matching", 1},
		{"mul_matching", "m1/mul_matching", 1},
		{"mlp_initializers", "m1/mlp_initializers", 1},
		{"residual", "m1/residual", 2},
	}

	for index, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := modelFixture(t, tt.fixture)
			cpuModel := loadedModel(t, data, Options{})
			inputs := makeInputsForMetadata(t, cpuModel.Metadata().Inputs, index+1)
			cpuOutputs, err := modelSession(t, cpuModel).Run(context.Background(), inputs, RunOptions{})
			if err != nil {
				t.Fatal(err)
			}

			gpuModel := loadedModel(t, data, Options{
				Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat32},
			})
			gpuSess := modelSession(t, gpuModel)
			gpuOutputs, err := gpuSess.Run(context.Background(), inputs, RunOptions{})
			if err != nil {
				t.Fatal(err)
			}

			sessStats, err := gpuSess.GPUStats()
			if err != nil {
				t.Fatal(err)
			}

			if tt.deviceOps > 0 && sessStats.DeviceOps < tt.deviceOps {
				t.Fatalf("expected at least %d device ops, got %+v", tt.deviceOps, sessStats)
			}

			if tt.deviceOps == 0 && sessStats.HostFallbackOps == 0 {
				t.Fatalf("expected CPU fallback for non-unit alpha/beta Gemm, got %+v", sessStats)
			}

			compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, 1e-5, 1e-5)
		})
	}
}

func TestMetalLargeGraphErrorDistributions(t *testing.T) {
	requireMetal(t)
	const (
		vocab  int64 = 4096
		hidden int64 = 1024
		ffn    int64 = 4096
	)

	residentGraph := buildResidentTransformerFFNLikeGraph(t, vocab, hidden, ffn)
	hybridGraph := buildHybridMLPONNXGraph(t, hidden, ffn)

	precisions := []struct {
		precision WeightPrecision
		atol      float64
		rtol      float64
	}{
		{WeightFloat32, 1e-4, 1e-4},
		{WeightFloat16, 1.5e-2, 1.5e-2},
		{WeightBFloat16, 1e-1, 1e-1},
	}

	for _, batch := range []int64{1, 32} {
		rows := make([]int64, batch)
		for i := range rows {
			rows[i] = int64(i*7) % vocab
		}

		resInputs := map[string]*Tensor{
			"indices":  intTensor(t, []int64{batch}, rows),
			"residual": floatTensor(t, []int64{batch, hidden}, deterministicFloats(int(batch*hidden), 101)),
			"scale":    floatTensor(t, []int64{batch, ffn}, deterministicFloats(int(batch*ffn), 202)),
		}
		resCPUModel := loadedModel(t, residentGraph, Options{Limits: largeModelLimits()})
		resCPUOut, err := modelSession(t, resCPUModel).Run(context.Background(), resInputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		hybInputs := map[string]*Tensor{
			"features": floatTensor(t, []int64{batch, hidden}, deterministicFloats(int(batch*hidden), 303)),
		}
		hybCPUModel := loadedModel(t, hybridGraph, Options{Limits: largeModelLimits()})
		hybCPUOut, err := modelSession(t, hybCPUModel).Run(context.Background(), hybInputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		for _, prec := range precisions {
			t.Run(fmt.Sprintf("ResidentFFN_M%d_%s", batch, prec.precision), func(t *testing.T) {
				gpuModel := loadedModel(t, residentGraph, Options{
					Limits:    largeModelLimits(),
					Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: prec.precision},
				})
				gpuSess := modelSession(t, gpuModel)
				gpuOut, runErr := gpuSess.Run(context.Background(), resInputs, RunOptions{})
				if runErr != nil {
					t.Fatal(runErr)
				}

				dist := compareOutputsAgainstCPU(t, resCPUOut, gpuOut, prec.atol, prec.rtol)
				mStats := gpuModel.GPUStats()
				sStats, _ := gpuSess.GPUStats()
				t.Logf("ResidentFFN M=%d %s: wtBytes=%d/%d actBytes=%d submits=%d n=%d p50=%.3e p95=%.3e p99=%.3e max=%.3e",
					batch, prec.precision, mStats.WeightBytes, mStats.FP32EquivalentWeightBytes,
					sStats.ActivationBytes, sStats.ScopeSubmits, dist.count, dist.p50, dist.p95, dist.p99, dist.maxAbs)
			})

			t.Run(fmt.Sprintf("HybridMLP_M%d_%s", batch, prec.precision), func(t *testing.T) {
				gpuModel := loadedModel(t, hybridGraph, Options{
					Limits:    largeModelLimits(),
					Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: prec.precision},
				})
				gpuSess := modelSession(t, gpuModel)
				gpuOut, runErr := gpuSess.Run(context.Background(), hybInputs, RunOptions{})
				if runErr != nil {
					t.Fatal(runErr)
				}

				dist := compareOutputsAgainstCPU(t, hybCPUOut, gpuOut, prec.atol, prec.rtol)
				mStats := gpuModel.GPUStats()
				sStats, _ := gpuSess.GPUStats()
				t.Logf("HybridMLP M=%d %s: wtBytes=%d/%d actBytes=%d submits=%d n=%d p50=%.3e p95=%.3e p99=%.3e max=%.3e",
					batch, prec.precision, mStats.WeightBytes, mStats.FP32EquivalentWeightBytes,
					sStats.ActivationBytes, sStats.ScopeSubmits, dist.count, dist.p50, dist.p95, dist.p99, dist.maxAbs)
			})
		}
	}
}

func makeInputsForMetadata(t *testing.T, infos []ValueInfo, seed int) (inputs map[string]*Tensor) {
	t.Helper()
	inputs = make(map[string]*Tensor, len(infos))
	for i, info := range infos {
		if info.Defaulted {
			continue
		}

		shape := make([]int64, len(info.Dimensions))
		elements := int64(1)
		for axis, dim := range info.Dimensions {
			size := dim.Value
			if dim.Kind != FixedDimension {
				size = 2
			}

			shape[axis] = size
			elements *= size
		}

		if info.DataType == Float32 {
			inputs[info.Name] = floatTensor(t, shape, deterministicFloats(int(elements), seed*31+i))
		} else {
			vals := make([]int64, elements)
			inputs[info.Name] = intTensor(t, shape, vals)
		}
	}

	return inputs
}
