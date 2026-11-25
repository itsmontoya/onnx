# ONNX-01 Consumer Report: `neuralnetwork/gpu` Integration in `onnx`

**Task:** `ONNX-01` (`github.com/itsmontoya/onnx`)  
**Upstream dependency:** `github.com/itsmontoya/neuralnetwork@v0.12.1`  
**Upstream commit SHA:** `32f83b827469b384e3a2546dc9ef4a5482e2c479`  
**Module checksum:** `h1:y4YSQZ9BaK1tM301y1lqSD+A18j7NAYtYi2i8+4Q0yY=` (`go.mod`: `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=`)  
**Hardware / Environment:** Apple M3 (`10`-core GPU, 24 GiB unified memory, Metal 4), Darwin `25.5.0 arm64`, Go `1.27.4` (`go 1.27.3` in `go.mod`, `GOWORK=off`)  
**Raw artifacts:** `.benchmarks/gpu-onnx-01/2026-11-25-m3/` (gitignored: `test-matrix.log`, `metal-parity.txt`, `bench-cpu-neon.txt`, `bench-metal.txt`)

## 1. Opt-In Metal Execution Architecture

CPU execution remains the zero-value default (`Options{}`) with zero behavioral changes. Callers opt into Metal explicitly at load time:

```go
model, err := onnx.LoadFile(ctx, path, onnx.Options{
    Execution: onnx.ExecutionOptions{
        Device:          onnx.DeviceMetal,
        WeightPrecision: onnx.WeightFloat16, // WeightFloat32 (default), WeightFloat16, WeightBFloat16
    },
})
```

- **Explicit unavailability:** Requesting `DeviceMetal` on non-Metal builds (`CGO_ENABLED=0`, `-tags=purego`, non-Darwin, or hosts without Metal) fails at `Load*` with `onnx.ErrUnavailable` rather than silently falling back to CPU.
- **Model/Session ownership:** `executionPlan` reference-counts `Model` and its live `Session`s. Immutable resident weight tensors (`gpu.Tensor`) and the `gpu.Device` handle are shared across concurrent sessions and released only when the model and all sessions have closed. Each `Session` owns a private `metalCache` of reusable FP32 activation buffers, released on `Session.Close()` or reset on run error.
- **Resident scope batching & hybrid fallback:** Consecutive GPU-eligible operators record into a single `gpu.Scope` with zero intermediate host readbacks. When a CPU-only operator (such as `Relu`, `Softmax`, `Sub`, `Conv`, or `LayerNormalization`) or final output publication requires host data, the open scope submits, waits, and downloads only the required tensor.
- **Observability:** `Model.GPUStats()` reports immutable model-level weight residency (`WeightTensors`, `ReducedPrecisionTensors`, `WeightBytes`, `FP32EquivalentWeightBytes`); `Session.GPUStats()` reports per-run `DeviceOps`, `HostFallbackOps`, `ScopeSubmits`, `HostUploads`, `HostDownloads`, and cached activation residency.

### Operator Mapping (`onnx` Opset 13/17 → `neuralnetwork/gpu` v0.12.1)

| ONNX Operator | Metal GPU Path | Supported Weight DTypes | CPU Fallback Cases |
| --- | --- | --- | --- |
| `MatMul` | `Scope.MatMul` | `Float32`, `Float16`, `BFloat16` | Rank != 2 (vector promotion, 3D+ batched), empty dims |
| `Gemm` | `Scope.MatMul` (`transB=0`), `Scope.MatMulRightTranspose` (`transB=1`), plus resident `Scope.Add` / `GatherRows` row-bias broadcast | `Float32`, `Float16`, `BFloat16` (B weight); `Float32` (C bias) | `transA=1`, `alpha != 1`, `beta` not in `{0,1}`, scalar/column `C` |
| `Add`, `Mul` | `Scope.Add`, `Scope.Mul` (same-shape or 2D+1D/row bias via `GatherRows`) | `Float32` | General multidirectional broadcast, rank-0 scalars, empty dims |
| `Gather` | `Scope.GatherRows` (negative ONNX indices normalized on host) | `Float32`, `Float16`, `BFloat16` (dequantizing to `Float32`) | `axis != 0`, `data` rank != 2, `indices` rank != 1, `int64` data |
| `Identity` | Zero-copy resident handle propagation | Any resident GPU tensor | Host-only tensors |
| `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `Sub`, `Div`, `LayerNormalization`, `Conv`, pooling, `BatchNormalization`, shape ops | — | — | Always CPU fallback (not in `neuralnetwork/gpu`) |

## 2. Numerical Parity & Error Distributions

All Metal parity tests passed against `onnx`'s CPU reference (and checked-in golden outputs for FP32) in `metal-parity.txt`:

- **Checked-in ONNX fixtures (`mlp.onnx` batch 1/3, `branch.onnx` batch 1/3, unmodified PyTorch opset-17 `pytorch_mlp.onnx`, external-weights `external/models/mlp.onnx`, and `m1/*` unit fixtures):**
  - `mlp.onnx` (`Gemm → Relu → Gemm → Softmax`): `2` device ops, `2` fallback ops, `2` scope submits, weight bytes `68 → 44 B` (`FP16`/`BF16`), max abs error `0.000e+00` across `FP32`, `FP16`, and `BF16`.
  - `branch.onnx` (`Gemm → Relu → Gemm → Add → Softmax / Sub`): `3` device ops (`Gemm` + `Add` batched in 1 scope), `3` fallback ops, `2` scope submits, max abs error `0.000e+00`.
- **Large ONNX graphs (`ResidentFFN`: `vocab=4096, hidden=1024, ffn=4096`, 6 device ops / 0 fallback / 1 scope submit; `HybridMLP`: `hidden=1024, ffn=4096`, `Gemm → Relu → Gemm`, 2 device ops / 1 fallback / 2 scope submits):**

| Workload | Precision | Weight Bytes (vs FP32) | Outputs | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResidentFFN M=1` | `float32` | 50,352,128 (100%) | 1,024 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `ResidentFFN M=1` | `float16` | **25,186,304 (50.0%)** | 1,024 | `1.736e-04` | `5.370e-04` | `7.063e-04` | `1.030e-03` |
| `ResidentFFN M=1` | `bfloat16` | **25,186,304 (50.0%)** | 1,024 | `1.419e-03` | `4.283e-03` | `5.560e-03` | `7.872e-03` |
| `ResidentFFN M=32` | `float32` | 50,352,128 (100%) | 32,768 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `ResidentFFN M=32` | `float16` | **25,186,304 (50.0%)** | 32,768 | `1.912e-04` | `5.602e-04` | `7.305e-04` | `1.362e-03` |
| `ResidentFFN M=32` | `bfloat16` | **25,186,304 (50.0%)** | 32,768 | `1.538e-03` | `4.476e-03` | `5.861e-03` | `9.520e-03` |
| `HybridMLP M=1` | `float32` | 33,574,912 (100%) | 1,024 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `HybridMLP M=1` | `float16` | **16,797,696 (50.0%)** | 1,024 | `7.592e-04` | `2.198e-03` | `2.885e-03` | `3.766e-03` |
| `HybridMLP M=1` | `bfloat16` | **16,797,696 (50.0%)** | 1,024 | `6.444e-03` | `1.694e-02` | `2.170e-02` | `2.904e-02` |
| `HybridMLP M=32` | `float32` | 33,574,912 (100%) | 32,768 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `HybridMLP M=32` | `float16` | **16,797,696 (50.0%)** | 32,768 | `7.715e-04` | `2.239e-03` | `2.956e-03` | `4.664e-03` |
| `HybridMLP M=32` | `bfloat16` | **16,797,696 (50.0%)** | 32,768 | `6.148e-03` | `1.786e-02` | `2.365e-02` | `3.499e-02` |

Note: Rank-2 weight matrices (`embed_table`, `w1`, `w2`) are halved exactly (`50.0%`), while 1D FP32 biases (`b1`, `b2`, 20 KB total) stay `Float32`.

## 3. Paired CPU NEON vs. Metal Benchmarks on Apple M3

Medians of 5 runs (`-benchtime=5x -count=5`). Pure ARM64 NEON CPU was measured with `CGO_ENABLED=0` (`bench-cpu-neon.txt`); Metal modes and `-tags=metal` `DeviceCPU` (legacy copying `neuralnetwork/tensor` Metal path) were measured with `CGO_ENABLED=1 -tags=metal` (`bench-metal.txt`).

| Workload | CPU NEON (`CGO=0`) | Legacy `DeviceCPU` (`-tags=metal`) | Metal FP32 | Metal FP16 | Metal BF16 | Best Metal vs CPU NEON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Fixture_mlp_batch_1` | **4.16 µs** (`3.9 KB/op`) | 4.21 µs | 720.0 µs | 639.9 µs | 373.8 µs | 0.01× (submit-bound) |
| `Fixture_mlp_batch_3` | **3.31 µs** (`4.0 KB/op`) | 3.64 µs | 415.8 µs | 429.8 µs | 423.8 µs | 0.01× (submit-bound) |
| `ResidentFFN_M1` (1 submit) | **0.507 ms** (`77.9 KB/op`) | 1.810 ms (`33.7 MB/op`) | 2.055 ms (`35.5 KB/op`) | 0.651 ms (`35.5 KB/op`) | 0.669 ms (`35.5 KB/op`) | 0.78× (2.78× vs legacy) |
| `HybridMLP_M1` (2 submits) | **0.484 ms** (`66.0 KB/op`) | 1.887 ms (`33.7 MB/op`) | 1.473 ms (`85.3 KB/op`) | 0.891 ms (`85.3 KB/op`) | 0.811 ms (`85.3 KB/op`) | 0.60× (2.33× vs legacy) |
| `ResidentFFN_M32` (1 submit) | 5.645 ms (`2.24 MB/op`) | 8.055 ms (`37.1 MB/op`) | **3.686 ms** (`1.05 MB/op`) | **2.493 ms** (`1.05 MB/op`) | **1.565 ms** (`1.05 MB/op`) | **3.61× faster** (BF16) |
| `HybridMLP_M32` (2 submits) | 5.729 ms (`1.97 MB/op`) | 7.950 ms (`36.8 MB/op`) | **3.829 ms** (`2.63 MB/op`) | **1.826 ms** (`2.63 MB/op`) | **1.822 ms** (`2.63 MB/op`) | **3.15× faster** (BF16) |

### Key Observations
- **Batched workloads (`M=32`) show clear Metal speedups:** `1.50×–1.53×` in FP32, `2.26×–3.14×` in FP16, and `3.15×–3.61×` in BF16 over ARM64 NEON CPU, while cutting rank-2 weight residency by 50% and halving host allocations on `ResidentFFN_M32` (`2.24 MB/op → 1.05 MB/op`).
- **Tiny models & small `M=1` workloads remain faster on CPU NEON:** Per-scope submit/wait latency (~180–200 µs) makes opt-in Metal slower than CPU NEON on tiny fixtures (`mlp.onnx`) and small 2-layer `M=1` graphs, validating keeping CPU as default.
- **Hybrid fallback penalty is measurable:** Adding one CPU `Relu` fallback between two `Gemm` layers (`HybridMLP` vs `ResidentFFN`) adds a second `Scope` submit/wait (~200 µs at `M=1`) and increases host allocations at `M=32` from `1.05 MB/op` to `2.63 MB/op` due to downloading and re-uploading `[32, 4096]`.

## 4. `neuralnetwork/gpu` API Generality Assessment (Input for `REVIEW-2`)

### What Worked Cleanly
1. **Zero-build-tag consumer portability:** `neuralnetwork/gpu` compiles everywhere (`CGO_ENABLED=0`, `-tags=purego`, `GOOS=linux GOARCH=amd64`) and returns `gpu.ErrUnavailable` cleanly when disabled. `onnx` required zero platform build-tag stub files.
2. **Ownership and concurrency model:** `gpu.Device` + shared read-only weight `gpu.Tensor`s + per-session activation buffers + per-run `gpu.Scope` mapped cleanly onto `onnx`'s `Model`/`Session` lifecycle, passing `-race -tags=metal` with concurrent sessions and out-of-order `Model.Close()` before `Session.Close()`.
3. **Uniform reduced-precision weight interface:** `NewTensorWithDType`, `UploadFloat32As`, `MatMul`, `MatMulRightTranspose`, and dequantizing `GatherRows` worked seamlessly for FP32, FP16, and BF16 weights with FP32 accumulation.

### Concrete API Friction & Missing Generic Primitives for `REVIEW-2`
1. **Missing standard activations (`Relu`, `Sigmoid`, `Tanh`, `Softmax`):** `gpu.Scope` currently exposes only Llama-specific `SiLU` and `SwiGLU`. Missing `Relu` and `Softmax` prevented even basic ONNX MLPs (`mlp.onnx`, `branch.onnx`, `pytorch_mlp.onnx`) from running in a single resident scope.
2. **Missing elementwise primitives (`Sub`, `Div`, scalar scale) and broadcasting (`AddBias` / broadcast `Add`/`Mul`):** `Scope.Add` and `Scope.Mul` require identical shapes. Supporting ONNX `Gemm` row biases (`[N]` / `[1, N]`) required a `GatherRows(bias[1, N], zeros(M), bcast[M, N])` workaround plus an extra `[M, N]` buffer; non-unit `alpha`/`beta` and general multidirectional broadcasts must fall back to CPU.
3. **Missing `LayerNormalization` (or reduction primitives `ReduceMean` + `Sqrt`):** `gpu.Scope` provides Llama-specific `RMSNorm` only; opset-17 `LayerNormalization` cannot run on GPU.
4. **Rank-2-only `MatMul` and no `Transpose` / `MatMulLeftTranspose`:** `Scope.MatMul` rejects 3D+ batched `MatMul`, rank-1 vector promotion, and `Gemm` with `transA=1`.
5. **No zero-copy `Tensor` reshape/view:** `gpu.Tensor` has no metadata reshape/view, and `Scope.Copy` requires identical shapes. Contiguous row-major ONNX shape ops (`Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`) cannot stay resident without consumer-side shape tracking or host round-trips.
6. **Legacy `neuralnetwork/tensor` Metal coupling under `-tags=metal`:** In `-tags=metal` builds, `onnx`'s `DeviceCPU` path still invokes `neuralnetwork/tensor.GemmInto`, which routes large matrices through the legacy copying `matrix` Metal adapter (`33–37 MB/op` host copies per call, `1.4×–3.6×` slower than `CGO_ENABLED=0` CPU NEON). `REVIEW-2` should evaluate decoupling `tensor` CPU ops from the legacy copying ANN Metal path once `neuralnetwork/gpu` stabilizes.

## 5. Required Test Matrix Results

All 14 entries of the Required Test Matrix passed (`test-matrix.log`):

- `GOWORK=off go fmt ./...` — PASS
- `GOWORK=off go vet ./...` — PASS
- `GOWORK=off go test ./...` — PASS (Metal-only tests skip cleanly; `TestMetalUnavailableReturnsErrUnavailable` PASS)
- `GOWORK=off go test -race ./...` — PASS
- `CGO_ENABLED=0 GOWORK=off go test ./...` — PASS
- `CGO_ENABLED=0 GOWORK=off go test -tags=purego ./...` — PASS
- `CGO_ENABLED=0 GOWORK=off go test -tags=metal ./...` — PASS
- `CGO_ENABLED=0 GOWORK=off go test -tags='metal purego' ./...` — PASS
- `CGO_ENABLED=1 GOWORK=off go test -tags='metal purego' ./...` — PASS
- `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 GOWORK=off go build ./...` — PASS (cross-build only)
- `GOOS=linux GOARCH=amd64 CGO_ENABLED=0 GOWORK=off go build -tags=metal ./...` — PASS (cross-build only)
- `CGO_ENABLED=1 GOWORK=off go vet -tags=metal ./...` — PASS
- `CGO_ENABLED=1 GOWORK=off go test -count=1 -tags=metal ./...` — PASS (all real Metal tests PASS; `TestMetalUnavailableReturnsErrUnavailable` skips when Metal is active)
- `CGO_ENABLED=1 GOWORK=off go test -count=1 -race -tags=metal ./...` — PASS
