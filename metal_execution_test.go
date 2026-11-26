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

					if !sessStats.Active || sessStats.DeviceOps == 0 || sessStats.HostFallbackOps != 0 || sessStats.ScopeSubmits != 1 {
						t.Fatalf("expected fully resident single-scope Metal execution, got %+v", sessStats)
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
			if statsErr != nil || sessStats.DeviceOps == 0 || sessStats.HostFallbackOps != 0 || sessStats.ScopeSubmits != 1 {
				t.Fatalf("expected fully resident single-scope Metal execution on PyTorch export: stats=%+v err=%v", sessStats, statsErr)
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
		gpuSess := modelSession(t, gpuModel)
		gpuOutputs, err := gpuSess.Run(context.Background(), inputs, RunOptions{})
		if err != nil {
			t.Fatal(err)
		}

		sessStats, err := gpuSess.GPUStats()
		if err != nil || sessStats.DeviceOps == 0 || sessStats.HostFallbackOps != 0 || sessStats.ScopeSubmits != 1 {
			t.Fatalf("expected fully resident single-scope Metal execution on external mlp: stats=%+v err=%v", sessStats, err)
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
		{"gemm_00_full", "m1/gemm_00_full", 1},
		{"gemm_01_row", "m1/gemm_01_row", 1},
		{"gemm_10_full", "m1/gemm_10_full", 1},
		{"gemm_11_full", "m1/gemm_11_full", 1},
		{"gemm_00_scalar", "m1/gemm_00_scalar", 1},
		{"gemm_00_column", "m1/gemm_00_column", 1},
		{"gemm_beta_zero", "m1/gemm_beta_zero", 1},
		{"gemm_defaults", "m1/gemm_defaults", 1},
		{"add_matching", "m1/add_matching", 1},
		{"add_broadcast", "m1/add_broadcast", 1},
		{"sub_matching", "m1/sub_matching", 1},
		{"sub_broadcast", "m1/sub_broadcast", 1},
		{"mul_matching", "m1/mul_matching", 1},
		{"mul_broadcast", "m1/mul_broadcast", 1},
		{"div_matching", "m1/div_matching", 1},
		{"div_broadcast", "m1/div_broadcast", 1},
		{"relu_finite", "m1/relu_finite", 1},
		{"sigmoid_finite", "m1/sigmoid_finite", 1},
		{"tanh_finite", "m1/tanh_finite", 1},
		{"softmax_default", "m1/softmax_default", 1},
		{"softmax_axis_neg1", "m1/softmax_axis_-1", 1},
		{"softmax_axis_neg2", "m1/softmax_axis_-2", 1},
		{"softmax_axis_0", "m1/softmax_axis_0", 1},
		{"softmax_axis_1", "m1/softmax_axis_1", 1},
		{"matmul_batch", "m1/matmul_batch", 1},
		{"mlp_initializers", "m1/mlp_initializers", 3},
		{"residual", "m1/residual", 3},
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

			if sessStats.DeviceOps < tt.deviceOps || sessStats.HostFallbackOps != 0 || sessStats.ScopeSubmits != 1 {
				t.Fatalf("expected fully resident single-scope execution (>=%d device ops, 0 fallback, 1 submit), got %+v", tt.deviceOps, sessStats)
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
				if sStats.HostFallbackOps != 0 || sStats.ScopeSubmits != 1 {
					t.Fatalf("expected fully resident single-scope ResidentFFN, got %+v", sStats)
				}

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
				if sStats.HostFallbackOps != 0 || sStats.ScopeSubmits != 1 {
					t.Fatalf("expected fully resident single-scope HybridMLP, got %+v", sStats)
				}

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

func TestMetalM2M3AndNormalizationFixturesParity(t *testing.T) {
	requireMetal(t)
	runSuite := func(t *testing.T, cases []conformanceCase, requireResident map[string]bool) {
		t.Helper()
		for _, fixture := range cases {
			t.Run(fixture.Name, func(t *testing.T) {
				data, err := fixture.Model.read()
				if err != nil {
					t.Fatal(err)
				}

				inputs, err := fixture.readInputs()
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

				if err = fixture.compareOutputs(gpuOutputs); err != nil {
					t.Fatalf("Metal vs reference: %v", err)
				}

				if requireResident[fixture.Name] {
					sessStats, statsErr := gpuSess.GPUStats()
					if statsErr != nil || sessStats.DeviceOps == 0 || sessStats.HostFallbackOps != 0 || sessStats.ScopeSubmits != 1 {
						t.Fatalf("expected fully resident single-scope execution for %s, got stats=%+v err=%v", fixture.Name, sessStats, statsErr)
					}
				}
			})
		}
	}

	t.Run("m2", func(t *testing.T) {
		runSuite(t, readM2Cases(t), map[string]bool{
			"transpose_default_float32":  true,
			"transpose_explicit_float32": true,
		})
	})

	t.Run("m3", func(t *testing.T) {
		runSuite(t, readM3Cases(t), nil)
	})

	t.Run("normalization", func(t *testing.T) {
		runSuite(t, readNormalizationCases(t), nil)
	})
}

func TestMetalLayerNormalizationParity(t *testing.T) {
	requireMetal(t)
	for _, withBias := range []bool{false, true} {
		name := "no_bias"
		if withBias {
			name = "with_bias"
		}

		t.Run(name, func(t *testing.T) {
			graphBytes := buildLayerNormGraph(t, 256, withBias)
			input := floatTensor(t, []int64{3, 5, 256}, deterministicFloats(3*5*256, 701))
			inputs := map[string]*Tensor{"x": input}

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
				{WeightFloat32, 1e-5, 1e-5},
				{WeightFloat16, 2e-3, 2e-3},
				{WeightBFloat16, 1.5e-2, 1.5e-2},
			}

			for _, prec := range precisions {
				t.Run(prec.precision.String(), func(t *testing.T) {
					gpuModel := loadedModel(t, graphBytes, Options{
						Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: prec.precision},
					})
					mStats := gpuModel.GPUStats()
					expectedReduced := 1
					if withBias {
						expectedReduced = 2
					}

					if prec.precision != WeightFloat32 && mStats.ReducedPrecisionTensors != expectedReduced {
						t.Fatalf("expected %d reduced-precision LayerNorm params, got %+v", expectedReduced, mStats)
					}

					gpuSess := modelSession(t, gpuModel)
					gpuOutputs, runErr := gpuSess.Run(context.Background(), inputs, RunOptions{})
					if runErr != nil {
						t.Fatal(runErr)
					}

					sStats, statsErr := gpuSess.GPUStats()
					if statsErr != nil || sStats.DeviceOps != 1 || sStats.HostFallbackOps != 0 || sStats.ScopeSubmits != 1 {
						t.Fatalf("expected resident single-scope LayerNorm, got stats=%+v err=%v", sStats, statsErr)
					}

					compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, prec.atol, prec.rtol)
				})
			}
		})
	}
}

func TestMetalBatchedMatMulAndTransposeParity(t *testing.T) {
	requireMetal(t)
	graphBytes := buildBatchedMatMulTransposeGraph(t, 2, 4, 8, 16)
	q := floatTensor(t, []int64{2, 8, 4, 16}, deterministicFloats(2*8*4*16, 801))
	k := floatTensor(t, []int64{2, 8, 4, 16}, deterministicFloats(2*8*4*16, 802))
	inputs := map[string]*Tensor{"q": q, "k": k}

	cpuModel := loadedModel(t, graphBytes, Options{})
	cpuOutputs, err := modelSession(t, cpuModel).Run(context.Background(), inputs, RunOptions{})
	if err != nil {
		t.Fatal(err)
	}

	for _, prec := range []struct {
		precision WeightPrecision
		atol      float64
	}{
		{WeightFloat32, 1e-4},
		{WeightFloat16, 5e-3},
		{WeightBFloat16, 3e-2},
	} {
		t.Run(prec.precision.String(), func(t *testing.T) {
			gpuModel := loadedModel(t, graphBytes, Options{
				Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: prec.precision},
			})
			gpuSess := modelSession(t, gpuModel)
			gpuOutputs, runErr := gpuSess.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatal(runErr)
			}

			sStats, statsErr := gpuSess.GPUStats()
			if statsErr != nil || sStats.DeviceOps != 5 || sStats.HostFallbackOps != 0 || sStats.ScopeSubmits != 1 {
				t.Fatalf("expected 5 resident device ops in 1 scope, got stats=%+v err=%v", sStats, statsErr)
			}

			compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, prec.atol, prec.atol)
		})
	}
}

func TestMetalZeroCopyShapeViewsAndLifecycle(t *testing.T) {
	requireMetal(t)
	graphBytes := buildZeroCopyShapeViewChainGraph(t, 16, 32)

	cpuModel := loadedModel(t, graphBytes, Options{})
	cpuSess := modelSession(t, cpuModel)

	gpuModel, err := LoadBytes(context.Background(), graphBytes, Options{
		Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: WeightFloat32},
	})
	if err != nil {
		t.Fatal(err)
	}

	gpuSess, err := gpuModel.NewSession()
	if err != nil {
		_ = gpuModel.Close()
		t.Fatal(err)
	}

	// Run multiple dynamic batch sizes to exercise view creation, reuse, and cleanup.
	for _, batch := range []int64{1, 4, 2, 3} {
		inputs := map[string]*Tensor{
			"x": floatTensor(t, []int64{batch, 16}, deterministicFloats(int(batch*16), int(900+batch))),
		}
		cpuOutputs, cpuErr := cpuSess.Run(context.Background(), inputs, RunOptions{})
		if cpuErr != nil {
			t.Fatal(cpuErr)
		}

		gpuOutputs, gpuErr := gpuSess.Run(context.Background(), inputs, RunOptions{})
		if gpuErr != nil {
			t.Fatal(gpuErr)
		}

		sStats, statsErr := gpuSess.GPUStats()
		if statsErr != nil || sStats.DeviceOps != 6 || sStats.HostFallbackOps != 0 || sStats.ScopeSubmits != 1 || sStats.HostUploads != 1 || sStats.HostDownloads != 1 {
			t.Fatalf("batch=%d: expected 6 device ops (2 Gemm + 4 zero-copy views), 0 fallback, 1 submit, 1 upload, 1 download; got stats=%+v err=%v", batch, sStats, statsErr)
		}

		// Verify zero-copy views allocate no activation buffers of their own:
		// only input upload + Gemm1 (dst, prod) + Gemm2 (dst, prod) = 5 buffers.
		if sStats.ActivationBuffers != 5 {
			t.Fatalf("batch=%d: expected 5 cached activation buffers (0 for views), got %+v", batch, sStats)
		}

		compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, 1e-5, 1e-5)
	}

	// Closing session then model must succeed cleanly (Device.Close returns ErrBusy if any view leaked).
	if err = gpuSess.Close(); err != nil {
		t.Fatalf("Session.Close failed (possible leaked view): %v", err)
	}

	if err = gpuModel.Close(); err != nil {
		t.Fatalf("Model.Close failed (possible leaked view): %v", err)
	}
}

func TestMetalTransformerEncoderBlockParity(t *testing.T) {
	requireMetal(t)
	const (
		batch   int64 = 2
		seqLen  int64 = 16
		heads   int64 = 4
		headDim int64 = 32
		hidden  int64 = heads * headDim
		ffn     int64 = hidden * 4
	)

	graphBytes := buildTransformerEncoderBlockGraph(t, batch, seqLen, heads, headDim, ffn)
	inputs := map[string]*Tensor{
		"x": floatTensor(t, []int64{batch, seqLen, hidden}, deterministicFloats(int(batch*seqLen*hidden), 1201)),
	}

	cpuModel := loadedModel(t, graphBytes, Options{Limits: largeModelLimits()})
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
		{WeightFloat16, 1.5e-2, 1.5e-2},
		{WeightBFloat16, 1e-1, 1e-1},
	}

	for _, prec := range precisions {
		t.Run(prec.precision.String(), func(t *testing.T) {
			gpuModel := loadedModel(t, graphBytes, Options{
				Limits:    largeModelLimits(),
				Execution: ExecutionOptions{Device: DeviceMetal, WeightPrecision: prec.precision},
			})
			gpuSess := modelSession(t, gpuModel)
			gpuOutputs, runErr := gpuSess.Run(context.Background(), inputs, RunOptions{})
			if runErr != nil {
				t.Fatal(runErr)
			}

			mStats := gpuModel.GPUStats()
			sStats, statsErr := gpuSess.GPUStats()
			if statsErr != nil {
				t.Fatal(statsErr)
			}

			if sStats.HostFallbackOps != 0 || sStats.ScopeSubmits != 1 || sStats.HostDownloads != 1 {
				t.Fatalf("expected fully resident single-scope Transformer block, got %+v", sStats)
			}

			dist := compareOutputsAgainstCPU(t, cpuOutputs, gpuOutputs, prec.atol, prec.rtol)
			t.Logf("TransformerBlock (%s): deviceOps=%d wtBytes=%d/%d actBufs=%d actBytes=%d submits=%d p50=%.3e p95=%.3e p99=%.3e max=%.3e",
				prec.precision, sStats.DeviceOps, mStats.WeightBytes, mStats.FP32EquivalentWeightBytes,
				sStats.ActivationBuffers, sStats.ActivationBytes, sStats.ScopeSubmits,
				dist.p50, dist.p95, dist.p99, dist.maxAbs)
		})
	}
}

func makeInt64Initializer(name string, shape []int64, values []int64) (out *onnxpb.TensorProto) {
	return &onnxpb.TensorProto{
		Name:      proto.String(name),
		DataType:  proto.Int32(int32(onnxpb.TensorProto_INT64)),
		Dims:      shape,
		Int64Data: values,
	}
}

func makeFixedFloatValueInfo(name string, dims []int64) (out *onnxpb.ValueInfoProto) {
	protoDims := make([]*onnxpb.TensorShapeProto_Dimension, len(dims))
	for i, d := range dims {
		protoDims[i] = &onnxpb.TensorShapeProto_Dimension{
			Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: d},
		}
	}

	return &onnxpb.ValueInfoProto{
		Name: proto.String(name),
		Type: &onnxpb.TypeProto{
			Value: &onnxpb.TypeProto_TensorType{
				TensorType: &onnxpb.TypeProto_Tensor{
					ElemType: proto.Int32(int32(onnxpb.TensorProto_FLOAT)),
					Shape:    &onnxpb.TensorShapeProto{Dim: protoDims},
				},
			},
		},
	}
}

func buildLayerNormGraph(tb testing.TB, width int64, withBias bool) (data []byte) {
	tb.Helper()
	inits := []*onnxpb.TensorProto{
		makeFloatInitializer("scale", []int64{width}, deterministicFloats(int(width), 711)),
	}
	nodeInputs := []string{"x", "scale"}
	if withBias {
		inits = append(inits, makeFloatInitializer("bias", []int64{width}, deterministicFloats(int(width), 712)))
		nodeInputs = append(nodeInputs, "bias")
	}

	model := &onnxpb.ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("onnx-gpu-test"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(17)},
		},
		Graph: &onnxpb.GraphProto{
			Name:        proto.String("layernorm_test"),
			Input:       []*onnxpb.ValueInfoProto{makeFixedFloatValueInfo("x", []int64{3, 5, width})},
			Output:      []*onnxpb.ValueInfoProto{makeFixedFloatValueInfo("y", []int64{3, 5, width})},
			Initializer: inits,
			Node: []*onnxpb.NodeProto{
				{
					Name:   proto.String("ln"),
					OpType: proto.String("LayerNormalization"),
					Input:  nodeInputs,
					Output: []string{"y"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(-1)},
						{Name: proto.String("epsilon"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(1e-5)},
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

func buildBatchedMatMulTransposeGraph(tb testing.TB, batch, heads, seqLen, headDim int64) (data []byte) {
	tb.Helper()
	model := &onnxpb.ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("onnx-gpu-test"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(13)},
		},
		Graph: &onnxpb.GraphProto{
			Name: proto.String("batched_matmul_transpose"),
			Input: []*onnxpb.ValueInfoProto{
				makeFixedFloatValueInfo("q", []int64{batch, seqLen, heads, headDim}),
				makeFixedFloatValueInfo("k", []int64{batch, seqLen, heads, headDim}),
			},
			Output: []*onnxpb.ValueInfoProto{
				makeFixedFloatValueInfo("out", []int64{batch, heads, seqLen, headDim}),
			},
			Initializer: []*onnxpb.TensorProto{
				// Shared [seqLen, headDim] weight broadcast across [batch, heads].
				makeFloatInitializer("proj", []int64{seqLen, headDim}, deterministicFloats(int(seqLen*headDim), 811)),
			},
			Node: []*onnxpb.NodeProto{
				{
					Name:   proto.String("q_t"),
					OpType: proto.String("Transpose"),
					Input:  []string{"q"},
					Output: []string{"qt"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 2, 1, 3}},
					},
				},
				{
					Name:   proto.String("k_t"),
					OpType: proto.String("Transpose"),
					Input:  []string{"k"},
					Output: []string{"kt"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 2, 3, 1}},
					},
				},
				{
					Name:   proto.String("scores_mm"),
					OpType: proto.String("MatMul"),
					Input:  []string{"qt", "kt"},
					Output: []string{"scores"},
				},
				{
					Name:   proto.String("scores_sm"),
					OpType: proto.String("Softmax"),
					Input:  []string{"scores"},
					Output: []string{"probs"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(-1)},
					},
				},
				{
					Name:   proto.String("proj_mm"),
					OpType: proto.String("MatMul"),
					Input:  []string{"probs", "proj"},
					Output: []string{"out"},
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

func buildZeroCopyShapeViewChainGraph(tb testing.TB, inDim, hiddenDim int64) (data []byte) {
	tb.Helper()
	model := &onnxpb.ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("onnx-gpu-test"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(13)},
		},
		Graph: &onnxpb.GraphProto{
			Name:   proto.String("zero_copy_shape_views"),
			Input:  []*onnxpb.ValueInfoProto{makeDynamicFloatValueInfo("x", "batch", inDim)},
			Output: []*onnxpb.ValueInfoProto{makeDynamicFloatValueInfo("y", "batch", inDim)},
			Initializer: []*onnxpb.TensorProto{
				makeFloatInitializer("w1", []int64{hiddenDim, inDim}, deterministicFloats(int(hiddenDim*inDim), 911)),
				makeFloatInitializer("b1", []int64{hiddenDim}, deterministicFloats(int(hiddenDim), 912)),
				makeFloatInitializer("w2", []int64{inDim, hiddenDim}, deterministicFloats(int(inDim*hiddenDim), 913)),
				makeFloatInitializer("b2", []int64{inDim}, deterministicFloats(int(inDim), 914)),
				makeInt64Initializer("unsq_axes", []int64{1}, []int64{1}),
				makeInt64Initializer("reshape_3d", []int64{3}, []int64{0, 1, hiddenDim}),
				makeInt64Initializer("sq_axes", []int64{1}, []int64{1}),
			},
			Node: []*onnxpb.NodeProto{
				{
					Name:   proto.String("fc1"),
					OpType: proto.String("Gemm"),
					Input:  []string{"x", "w1", "b1"},
					Output: []string{"h1"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("transB"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)},
					},
				},
				{
					Name:   proto.String("unsq"),
					OpType: proto.String("Unsqueeze"),
					Input:  []string{"h1", "unsq_axes"},
					Output: []string{"h2"},
				},
				{
					Name:   proto.String("resh"),
					OpType: proto.String("Reshape"),
					Input:  []string{"h2", "reshape_3d"},
					Output: []string{"h3"},
				},
				{
					Name:   proto.String("sq"),
					OpType: proto.String("Squeeze"),
					Input:  []string{"h3", "sq_axes"},
					Output: []string{"h4"},
				},
				{
					Name:   proto.String("flat"),
					OpType: proto.String("Flatten"),
					Input:  []string{"h4"},
					Output: []string{"h5"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(1)},
					},
				},
				{
					Name:   proto.String("fc2"),
					OpType: proto.String("Gemm"),
					Input:  []string{"h5", "w2", "b2"},
					Output: []string{"y"},
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

// buildTransformerEncoderBlockGraph constructs a complete opset-17 ONNX
// Transformer encoder block (Pre-LN Multi-Head Self-Attention + Pre-LN FFN)
// that executes 100% GPU-resident in one gpu.Scope.
func buildTransformerEncoderBlockGraph(tb testing.TB, batch, seqLen, heads, headDim, ffn int64) (data []byte) {
	tb.Helper()
	hidden := heads * headDim
	invSqrtD := float32(1.0 / math.Sqrt(float64(headDim)))

	model := &onnxpb.ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("onnx-gpu-test"),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{Domain: proto.String(""), Version: proto.Int64(17)},
		},
		Graph: &onnxpb.GraphProto{
			Name:   proto.String("transformer_encoder_block"),
			Input:  []*onnxpb.ValueInfoProto{makeFixedFloatValueInfo("x", []int64{batch, seqLen, hidden})},
			Output: []*onnxpb.ValueInfoProto{makeFixedFloatValueInfo("y", []int64{batch, seqLen, hidden})},
			Initializer: []*onnxpb.TensorProto{
				makeFloatInitializer("ln1_w", []int64{hidden}, deterministicFloats(int(hidden), 1001)),
				makeFloatInitializer("ln1_b", []int64{hidden}, deterministicFloats(int(hidden), 1002)),
				makeFloatInitializer("wq", []int64{hidden, hidden}, deterministicFloats(int(hidden*hidden), 1003)),
				makeFloatInitializer("bq", []int64{hidden}, deterministicFloats(int(hidden), 1004)),
				makeFloatInitializer("wk", []int64{hidden, hidden}, deterministicFloats(int(hidden*hidden), 1005)),
				makeFloatInitializer("bk", []int64{hidden}, deterministicFloats(int(hidden), 1006)),
				makeFloatInitializer("wv", []int64{hidden, hidden}, deterministicFloats(int(hidden*hidden), 1007)),
				makeFloatInitializer("bv", []int64{hidden}, deterministicFloats(int(hidden), 1008)),
				makeFloatInitializer("wo", []int64{hidden, hidden}, deterministicFloats(int(hidden*hidden), 1009)),
				makeFloatInitializer("bo", []int64{hidden}, deterministicFloats(int(hidden), 1010)),
				makeFloatInitializer("attn_scale", nil, []float32{invSqrtD}),
				makeInt64Initializer("split_heads_shape", []int64{4}, []int64{batch, seqLen, heads, headDim}),
				makeInt64Initializer("merge_heads_shape", []int64{3}, []int64{batch, seqLen, hidden}),
				makeFloatInitializer("ln2_w", []int64{hidden}, deterministicFloats(int(hidden), 1011)),
				makeFloatInitializer("ln2_b", []int64{hidden}, deterministicFloats(int(hidden), 1012)),
				makeFloatInitializer("w1", []int64{hidden, ffn}, deterministicFloats(int(hidden*ffn), 1013)),
				makeFloatInitializer("b1", []int64{ffn}, deterministicFloats(int(ffn), 1014)),
				makeFloatInitializer("w2", []int64{ffn, hidden}, deterministicFloats(int(ffn*hidden), 1015)),
				makeFloatInitializer("b2", []int64{hidden}, deterministicFloats(int(hidden), 1016)),
			},
			Node: []*onnxpb.NodeProto{
				{
					Name:   proto.String("ln1"),
					OpType: proto.String("LayerNormalization"),
					Input:  []string{"x", "ln1_w", "ln1_b"},
					Output: []string{"n1"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(-1)},
						{Name: proto.String("epsilon"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(1e-5)},
					},
				},
				{Name: proto.String("q_mm"), OpType: proto.String("MatMul"), Input: []string{"n1", "wq"}, Output: []string{"q_lin"}},
				{Name: proto.String("q_add"), OpType: proto.String("Add"), Input: []string{"q_lin", "bq"}, Output: []string{"q_b"}},
				{Name: proto.String("q_resh"), OpType: proto.String("Reshape"), Input: []string{"q_b", "split_heads_shape"}, Output: []string{"q_4d"}},
				{
					Name: proto.String("q_tr"), OpType: proto.String("Transpose"), Input: []string{"q_4d"}, Output: []string{"q_h"},
					Attribute: []*onnxpb.AttributeProto{{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 2, 1, 3}}},
				},
				{Name: proto.String("k_mm"), OpType: proto.String("MatMul"), Input: []string{"n1", "wk"}, Output: []string{"k_lin"}},
				{Name: proto.String("k_add"), OpType: proto.String("Add"), Input: []string{"k_lin", "bk"}, Output: []string{"k_b"}},
				{Name: proto.String("k_resh"), OpType: proto.String("Reshape"), Input: []string{"k_b", "split_heads_shape"}, Output: []string{"k_4d"}},
				{
					Name: proto.String("k_tr"), OpType: proto.String("Transpose"), Input: []string{"k_4d"}, Output: []string{"k_h_t"},
					Attribute: []*onnxpb.AttributeProto{{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 2, 3, 1}}},
				},
				{Name: proto.String("v_mm"), OpType: proto.String("MatMul"), Input: []string{"n1", "wv"}, Output: []string{"v_lin"}},
				{Name: proto.String("v_add"), OpType: proto.String("Add"), Input: []string{"v_lin", "bv"}, Output: []string{"v_b"}},
				{Name: proto.String("v_resh"), OpType: proto.String("Reshape"), Input: []string{"v_b", "split_heads_shape"}, Output: []string{"v_4d"}},
				{
					Name: proto.String("v_tr"), OpType: proto.String("Transpose"), Input: []string{"v_4d"}, Output: []string{"v_h"},
					Attribute: []*onnxpb.AttributeProto{{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 2, 1, 3}}},
				},
				{Name: proto.String("scores_mm"), OpType: proto.String("MatMul"), Input: []string{"q_h", "k_h_t"}, Output: []string{"scores"}},
				{Name: proto.String("scores_scale"), OpType: proto.String("Mul"), Input: []string{"scores", "attn_scale"}, Output: []string{"scaled_scores"}},
				{
					Name: proto.String("scores_sm"), OpType: proto.String("Softmax"), Input: []string{"scaled_scores"}, Output: []string{"probs"},
					Attribute: []*onnxpb.AttributeProto{{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(-1)}},
				},
				{Name: proto.String("ctx_mm"), OpType: proto.String("MatMul"), Input: []string{"probs", "v_h"}, Output: []string{"ctx_h"}},
				{
					Name: proto.String("ctx_tr"), OpType: proto.String("Transpose"), Input: []string{"ctx_h"}, Output: []string{"ctx_t"},
					Attribute: []*onnxpb.AttributeProto{{Name: proto.String("perm"), Type: onnxpb.AttributeProto_INTS.Enum(), Ints: []int64{0, 2, 1, 3}}},
				},
				{Name: proto.String("ctx_resh"), OpType: proto.String("Reshape"), Input: []string{"ctx_t", "merge_heads_shape"}, Output: []string{"ctx"}},
				{Name: proto.String("o_mm"), OpType: proto.String("MatMul"), Input: []string{"ctx", "wo"}, Output: []string{"o_lin"}},
				{Name: proto.String("o_add"), OpType: proto.String("Add"), Input: []string{"o_lin", "bo"}, Output: []string{"attn_out"}},
				{Name: proto.String("res1"), OpType: proto.String("Add"), Input: []string{"x", "attn_out"}, Output: []string{"h1"}},
				{
					Name:   proto.String("ln2"),
					OpType: proto.String("LayerNormalization"),
					Input:  []string{"h1", "ln2_w", "ln2_b"},
					Output: []string{"n2"},
					Attribute: []*onnxpb.AttributeProto{
						{Name: proto.String("axis"), Type: onnxpb.AttributeProto_INT.Enum(), I: proto.Int64(-1)},
						{Name: proto.String("epsilon"), Type: onnxpb.AttributeProto_FLOAT.Enum(), F: proto.Float32(1e-5)},
					},
				},
				{Name: proto.String("ffn1_mm"), OpType: proto.String("MatMul"), Input: []string{"n2", "w1"}, Output: []string{"f1_lin"}},
				{Name: proto.String("ffn1_add"), OpType: proto.String("Add"), Input: []string{"f1_lin", "b1"}, Output: []string{"f1"}},
				{Name: proto.String("ffn_relu"), OpType: proto.String("Relu"), Input: []string{"f1"}, Output: []string{"f1_act"}},
				{Name: proto.String("ffn2_mm"), OpType: proto.String("MatMul"), Input: []string{"f1_act", "w2"}, Output: []string{"f2_lin"}},
				{Name: proto.String("ffn2_add"), OpType: proto.String("Add"), Input: []string{"f2_lin", "b2"}, Output: []string{"ffn_out"}},
				{Name: proto.String("res2"), OpType: proto.String("Add"), Input: []string{"h1", "ffn_out"}, Output: []string{"y"}},
			},
		},
	}

	var err error
	if data, err = proto.Marshal(model); err != nil {
		tb.Fatal(err)
	}

	return data
}
