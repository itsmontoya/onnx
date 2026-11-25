# ONNX-02 Consumer Report: Phase 3 `neuralnetwork/gpu` Integration in `onnx`

**Task:** `ONNX-02` (`github.com/itsmontoya/onnx`)  
**Upstream dependency:** `github.com/itsmontoya/neuralnetwork@v0.13.0` (`GATE-F`)  
**Upstream commit SHA:** `a392e3971fc802105ad8fe6a8c24e998d68e9fd4`  
**Module checksum:** `h1:XUhe9AFw4ct/X33VyqgaVZ3kiLYVgP5GIP4QdFfqKQY=` (`go.mod`: `h1:SafbkuDehVErsuJAL5FHFsLhKJbeh/OI3ItvbmwDafw=`)  
**Hardware / Environment:** Apple M3 (`10`-core GPU, 24 GiB unified memory, Metal 4), Darwin `25.5.0 arm64`, Go `1.27.4` (`go 1.27.3` in `go.mod`, `GOWORK=off`)  
**Raw artifacts:** `.benchmarks/gpu-onnx-02/2026-11-25-m3/` (gitignored: `test-matrix.log`, `metal-parity.txt`, `bench-cpu-neon.txt`, `bench-metal.txt`)

## 1. Opt-In Metal Execution Architecture (`v0.13.0` Phase 3 Integration)

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
- **Model/Session ownership & view lifecycle:** `executionPlan` reference-counts `Model` and its live `Session`s. Immutable resident weight tensors (`gpu.Tensor`) and the `gpu.Device` handle are shared across concurrent sessions and released only when the model and all sessions have closed. Each `Session` owns a private `metalCache` of reusable FP32 activation buffers, released on `Session.Close()` or reset on run error. Zero-copy `Tensor.View` handles created by `Reshape`/`Flatten`/`Squeeze`/`Unsqueeze` during a `Run` are tracked per `metalEvaluation` and closed cleanly on `Run` completion or error so `metalCache` and `Device.Close()` release with zero leaks.
- **Single-scope residency:** Wiring Phase 3 primitives (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`/`SoftmaxAxis`, `Sub`/`Div`/`Scale`, `*Broadcast`, `LayerNorm`, `BatchedMatMul`, `MatMulLeftTranspose`, `Transpose`, `Tensor.View`) eliminates CPU fallback round-trips across standard ONNX MLPs, residual graphs, and full Transformer encoder blocks.

### Operator Mapping (`onnx` Opset 13/17 → `neuralnetwork/gpu` v0.13.0)

| ONNX Operator | Metal GPU Path | Supported Weight DTypes | CPU Fallback Cases |
| --- | --- | --- | --- |
| `MatMul` | `Scope.MatMul` (2D), `Scope.BatchedMatMul` (3D–6D with batch broadcasting) | `Float32`, `Float16`, `BFloat16` (right operand) | Rank < 2 (vector promotion), rank > 6, empty dims |
| `Gemm` | `Scope.MatMul` (`0,0`), `Scope.MatMulRightTranspose` (`0,1`), `Scope.MatMulLeftTranspose` (`1,0`), `Transpose`+`MatMulRightTranspose` (`1,1`); `Scope.Scale` for `alpha`/`beta`; `Scope.Add`/`Scope.AddBroadcast` for bias | `Float32`, `Float16`, `BFloat16` (B weight); `Float32` (C bias, including rank-0 scalar) | Empty dims, non-Float32 |
| `Add`, `Sub`, `Mul`, `Div` | `Scope.{Add,Sub,Mul,Div}` (same shape) and `Scope.{Add,Sub,Mul,Div}Broadcast` (rank 1–6 multidirectional broadcast, including rank-0 scalar operands) | `Float32` | Rank-0 output, rank > 6, empty dims |
| `Relu`, `Sigmoid`, `Tanh` | `Scope.ReLU`, `Scope.Sigmoid`, `Scope.Tanh` | `Float32` | Rank-0 scalar, empty dims |
| `Softmax` | `Scope.Softmax` (last axis), `Scope.SoftmaxAxis` (general axis, negative ONNX axis normalized) | `Float32` | Rank-0 scalar, empty dims |
| `LayerNormalization` | `Scope.LayerNorm` (optional bias) | `Float32`, `Float16`, `BFloat16` (scale & bias) | Optional `Mean`/`InvStdDev` outputs requested, non-last-axis normalization, non-positive `epsilon`, non-rank-1 scale/bias |
| `Transpose` | `Scope.Transpose` (rank 1–6) | `Float32` | Rank 0, rank > 6, empty dims, `int64` tensors |
| `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze` | Zero-copy `Tensor.View` (0 device copies, 0 cache allocations) | Any resident `Float32` tensor | Rank-0 input/output, empty dims, `int64` tensors |
| `Gather` | `Scope.GatherRows` (negative ONNX indices normalized on host) | `Float32`, `Float16`, `BFloat16` (dequantizing to `Float32`) | `axis != 0`, `data` rank != 2, `indices` rank != 1, `int64` data |
| `Identity` | Zero-copy resident handle propagation | Any resident GPU tensor | Host-only tensors |
| `Conv`, `MaxPool`, `AveragePool`, `GlobalAveragePool`, `BatchNormalization`, `Shape`, `Concat` | — | — | Always CPU fallback (seamlessly interoperates in hybrid graphs such as `m3/models/cnn.onnx`) |

## 2. Numerical Parity, Residency Confirmation & Error Distributions

All Metal parity tests passed against `onnx`'s CPU reference (and checked-in golden outputs for FP32) in `metal-parity.txt`:

- **Single-scope residency confirmed (`HostFallbackOps == 0`, `ScopeSubmits == 1`):**
  - `mlp.onnx` (`batch 1`/`3`): `4` device ops, **`0` fallback ops**, **`1` scope submit** (down from `2` fallbacks / `2` submits in `ONNX-01`), weight bytes `68 → 44 B`, max abs error `0.000e+00` across `FP32`/`FP16`/`BF16`.
  - `branch.onnx` (`batch 1`/`3`): `6` device ops, **`0` fallback ops**, **`1` scope submit** (down from `3` fallbacks / `2` submits in `ONNX-01`), max abs error `0.000e+00` (`batch 1`) / `1.863e-09` (`batch 3`).
  - `pytorch_mlp.onnx` (unmodified PyTorch opset-17 export) and `external/models/mlp.onnx`: **`0` fallback ops**, **`1` scope submit**.
  - `HybridMLP` (`hidden=1024, ffn=4096`, `Gemm → Relu → Gemm`): **`0` fallback ops**, **`1` scope submit** (down from `1` fallback / `2` submits in `ONNX-01`).
  - `m1/*` (27 unit fixtures covering `Gemm` all `transA`/`transB`/`alpha`/`beta`/scalar/column combinations, `Add`/`Sub`/`Mul`/`Div` matching + broadcast, `Relu`/`Sigmoid`/`Tanh`, `Softmax` all axes, batched `MatMul`, `mlp_initializers`, `residual`): all pass with **`0` fallback ops**, **`1` scope submit**.
  - `m2/*` (all 66 shape/indexing fixtures), `m3/*` (all CNN/pooling/BN fixtures including hybrid `cnn.onnx`), and `normalization/*` (all 28 LayerNormalization fixtures): all pass exact reference comparison under `DeviceMetal`.
  - Dedicated tests for resident `LayerNormalization` (with/without bias, FP32/FP16/BF16 weights), 3D/4D batched `MatMul` + `Transpose`, chained zero-copy `Reshape`/`Flatten`/`Squeeze`/`Unsqueeze` views across dynamic batch sizes (verifying zero extra activation buffers and clean `Session.Close()` / `Model.Close()`), and a full 29-op opset-17 Transformer encoder block (`HostFallbackOps == 0`, `ScopeSubmits == 1`, `HostDownloads == 1`).

| Workload | Precision | Weight Bytes (vs FP32) | Submits | Outputs | p50 Abs Err | p95 Abs Err | p99 Abs Err | Max Abs Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ResidentFFN M=1` | `float32` | 50,352,128 (100%) | 1 | 1,024 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `ResidentFFN M=1` | `float16` | **25,186,304 (50.0%)** | 1 | 1,024 | `1.736e-04` | `5.370e-04` | `7.063e-04` | `1.030e-03` |
| `ResidentFFN M=1` | `bfloat16` | **25,186,304 (50.0%)** | 1 | 1,024 | `1.419e-03` | `4.283e-03` | `5.560e-03` | `7.872e-03` |
| `ResidentFFN M=32` | `float32` | 50,352,128 (100%) | 1 | 32,768 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `ResidentFFN M=32` | `float16` | **25,186,304 (50.0%)** | 1 | 32,768 | `1.912e-04` | `5.602e-04` | `7.305e-04` | `1.362e-03` |
| `ResidentFFN M=32` | `bfloat16` | **25,186,304 (50.0%)** | 1 | 32,768 | `1.538e-03` | `4.476e-03` | `5.861e-03` | `9.520e-03` |
| `HybridMLP M=1` | `float32` | 33,574,912 (100%) | 1 | 1,024 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `HybridMLP M=1` | `float16` | **16,797,696 (50.0%)** | 1 | 1,024 | `7.592e-04` | `2.198e-03` | `2.885e-03` | `3.766e-03` |
| `HybridMLP M=1` | `bfloat16` | **16,797,696 (50.0%)** | 1 | 1,024 | `6.444e-03` | `1.694e-02` | `2.170e-02` | `2.904e-02` |
| `HybridMLP M=32` | `float32` | 33,574,912 (100%) | 1 | 32,768 | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `HybridMLP M=32` | `float16` | **16,797,696 (50.0%)** | 1 | 32,768 | `7.715e-04` | `2.239e-03` | `2.956e-03` | `4.664e-03` |
| `HybridMLP M=32` | `bfloat16` | **16,797,696 (50.0%)** | 1 | 32,768 | `6.148e-03` | `1.786e-02` | `2.365e-02` | `3.499e-02` |
| `TransformerBlock (2×16×128)` | `float32` | 793,092 (100%) | 1 | 4,096 | `2.384e-07` | `5.960e-07` | `9.537e-07` | `1.669e-06` |
| `TransformerBlock (2×16×128)` | `float16` | **398,852 (50.3%)** | 1 | 4,096 | `2.166e-04` | `5.761e-04` | `7.182e-04` | `8.899e-04` |
| `TransformerBlock (2×16×128)` | `bfloat16` | **398,852 (50.3%)** | 1 | 4,096 | `1.698e-03` | `4.911e-03` | `6.555e-03` | `8.329e-03` |

## 3. Paired CPU NEON vs. Metal Benchmarks on Apple M3

Medians of 5 runs (`-benchtime=5x -count=5`). Pure ARM64 NEON CPU was measured with `CGO_ENABLED=0` (`bench-cpu-neon.txt`); Metal modes and `-tags=metal` `DeviceCPU` (legacy copying `neuralnetwork/tensor` Metal path) were measured with `CGO_ENABLED=1 -tags=metal` (`bench-metal.txt`).

| Workload | CPU NEON (`CGO=0`) | Legacy `DeviceCPU` (`-tags=metal`) | Metal FP32 | Metal FP16 | Metal BF16 | Best Metal vs CPU NEON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Fixture_mlp_batch_1` (1 submit) | **3.4 µs** (`3.8 KB/op`) | 4.0 µs | 239.3 µs (`2.5 KB/op`) | 241.5 µs | 240.6 µs | 0.01× (submit-bound) |
| `Fixture_mlp_batch_3` (1 submit) | **3.5 µs** (`3.9 KB/op`) | 3.4 µs | 186.1 µs (`2.6 KB/op`) | 186.4 µs | 186.8 µs | 0.02× (submit-bound) |
| `ResidentFFN_M1` (1 submit) | **0.511 ms** (`76.1 KB/op`) | 1.930 ms (`32.1 MB/op`) | 0.924 ms (`34.8 KB/op`) | 0.697 ms (`34.8 KB/op`) | 0.691 ms (`34.8 KB/op`) | 0.74× (2.79× vs legacy) |
| `HybridMLP_M1` (1 submit) | **0.458 ms** (`67.7 KB/op`) | 1.740 ms (`32.1 MB/op`) | 1.043 ms (`18.4 KB/op`) | 0.650 ms (`18.4 KB/op`) | 0.659 ms (`18.4 KB/op`) | 0.70× (2.68× vs legacy) |
| `ResidentFFN_M32` (1 submit) | 5.547 ms (`2.13 MB/op`) | 7.978 ms (`35.4 MB/op`) | **3.636 ms** (`1.00 MB/op`) | **2.172 ms** (`1.00 MB/op`) | **2.514 ms** (`1.00 MB/op`) | **2.55× faster** (FP16) |
| `HybridMLP_M32` (1 submit) | 5.485 ms (`1.88 MB/op`) | 7.875 ms (`35.1 MB/op`) | **3.555 ms** (`514.4 KB/op`) | **1.302 ms** (`514.4 KB/op`) | **1.442 ms** (`514.4 KB/op`) | **4.21× faster** (FP16) |
| `TransformerBlock_B1_S16` (29 ops, 1 submit) | 27.948 ms (`1.47 MB/op`) | 4.835 ms (`14.5 MB/op`) | **1.726 ms** (`138.8 KB/op`) | **1.725 ms** (`138.8 KB/op`) | **1.695 ms** (`138.8 KB/op`) | **16.49× faster** (BF16) |
| `TransformerBlock_B4_S32` (29 ops, 1 submit) | 224.675 ms (`11.89 MB/op`) | 25.280 ms (`68.3 MB/op`) | **7.942 ms** (`1.01 MB/op`) | **7.422 ms** (`1.01 MB/op`) | **6.549 ms** (`1.01 MB/op`) | **34.31× faster** (BF16) |

### Before (`ONNX-01`) vs. After (`ONNX-02`) Eliminating CPU `Relu` & `GatherRows` Bias

| Workload | `ONNX-01` (2 submits, CPU `Relu` + `GatherRows` bias) | `ONNX-02` (1 submit, GPU `ReLU` + `AddBroadcast`) | Improvement |
| --- | ---: | ---: | ---: |
| `Fixture_mlp_batch_1` (Metal FP32) | 720.0 µs (2 submits) | **239.3 µs** (1 submit) | **3.01× faster** |
| `HybridMLP_M1` (Metal FP16) | 0.891 ms (`85.3 KB/op`, 2 submits) | **0.650 ms** (`18.4 KB/op`, 1 submit) | **1.37× faster**, **-78% host allocs** |
| `HybridMLP_M32` (Metal FP16) | 1.826 ms (`2.63 MB/op`, 2 submits) | **1.302 ms** (`514.4 KB/op`, 1 submit) | **1.40× faster**, **-80% host allocs** |
| `HybridMLP_M32` (Metal BF16) | 1.822 ms (`2.63 MB/op`, 2 submits) | **1.442 ms** (`514.4 KB/op`, 1 submit) | **1.26× faster**, **-80% host allocs** |

### Key Observations & Honest Tradeoffs
- **Eliminating the `Relu` boundary delivered both latency and memory wins:** `HybridMLP_M32` dropped from `2` submits / `2.63 MB/op` host allocs to `1` submit / `514.4 KB/op` (**-80% host allocs**), reaching **4.21× faster** than CPU NEON in FP16 (`1.302 ms` vs `5.485 ms`), and removing the `GatherRows` row-bias scratch buffer saved an additional `[M, N]` GPU allocation per `Gemm`.
- **Full Transformer encoder blocks shine on Metal:** A 29-operator opset-17 Pre-LN Transformer encoder block (`hidden=512, heads=8, headDim=64, ffn=2048`) runs in **1 `gpu.Scope`** with `0` CPU fallbacks, achieving **16.5×** speedup over CPU NEON at `B=1, S=16` (`1.70 ms` vs `27.95 ms`) and **34.3×** speedup at `B=4, S=32` (`6.55 ms` vs `224.67 ms`) while cutting host allocations by **11.8×** (`11.89 MB/op → 1.01 MB/op`).
- **Tiny models & small `M=1` MLPs remain faster on CPU NEON:** A single Metal command-buffer submit/wait still costs ~180–240 µs on M3, so tiny fixtures (`mlp.onnx`: `3.4 µs` CPU vs `186–239 µs` Metal) and 2-layer `M=1` MLPs (`0.46 ms` CPU vs `0.65 ms` Metal) remain faster on CPU NEON. Keeping `Options{}` CPU-by-default remains the right design.

## 4. `neuralnetwork/gpu` v0.13.0 API Assessment (Input for `REVIEW-3`)

### What Worked Cleanly
1. **Zero-copy `Tensor.View` with storage-level lease tracking:** `Reshape`, `Flatten`, `Squeeze`, and `Unsqueeze` integrated directly into `metalEvaluation` with zero device copies and zero activation-cache allocations. Closing per-run views in `metalEvaluation.close()` cleanly preserves `metalCache` ownership and passes `-race -tags=metal` across dynamic batch sizes and out-of-order `Model.Close()`.
2. **Native broadcast & scaling primitives:** `AddBroadcast`/`SubBroadcast`/`MulBroadcast`/`DivBroadcast` and `Scale` completely eliminated `metalPlan.rowBiases` and the `GatherRows` bias workaround, and enabled all 4 `Gemm` `(transA, transB)` modes plus arbitrary `alpha`/`beta` and scalar constants (`[]` uploaded as `[1]`) on GPU.
3. **Uniform reduced-precision support:** `LayerNorm`, `BatchedMatMul`, `MatMulLeftTranspose`, `MatMulRightTranspose`, and `GatherRows` all accept `Float32`/`Float16`/`BFloat16` weights uniformly with FP32 accumulation.

### Remaining Items for `REVIEW-3`
1. **Legacy `neuralnetwork/tensor` Metal coupling under `-tags=metal`:** In `-tags=metal` builds, `onnx`'s `DeviceCPU` path still routes large `Gemm`/`MatMul` calls through the legacy copying ANN Metal adapter (`32–68 MB/op` host copies per call). Decoupling `neuralnetwork/tensor` CPU ops from the legacy copying Metal path remains the highest-value cleanup for `-tags=metal` CPU baselines.
2. **Optional `Transpose` + `BatchedMatMul` fusion (`BatchedMatMulRightTranspose`):** ONNX represents attention `Q @ K^T` as `Transpose(K, [0,2,3,1]) → MatMul(Q, K_T)`. Currently `onnx` maps these 1-to-1 via `Scope.Transpose` + `Scope.BatchedMatMul` (keeping both in one `Scope`). A future pattern-fusion pass in `onnx` could fuse single-consumer last-two-axis transposes into `Scope.BatchedMatMulRightTranspose` to save one dispatch and activation buffer.
3. **Additional operators if target models require them:** `Conv`, pooling (`MaxPool`/`AveragePool`/`GlobalAveragePool`), `BatchNormalization`, `GELU`, and multi-output `LayerNormalization` (`Mean`/`InvStdDev`) currently fall back cleanly to CPU; prioritize only when driven by concrete target models.

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
