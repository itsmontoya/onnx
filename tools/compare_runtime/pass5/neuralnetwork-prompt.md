# Standalone handoff for Codex Sol 6.1 in github.com/itsmontoya/neuralnetwork

Historical handoff written against v0.10.1. Published v0.10.2 now includes the
residual-column kernel; see the [integration results](v0.10.2/README.md) before
using this prompt for further work.

You are working inside `github.com/itsmontoya/neuralnetwork`. Investigate and
implement one focused convolution optimization supported by fresh measurements.
Read this repository's `AGENTS.md` and `STYLEGUIDE.md` first. Preserve its public
contracts, existing numerical expectations, portable builds and explicit style.
Do not spawn other agents unless the instructions in your session authorize it.

**Profile first. The direction below is an evidence-supported hypothesis, not
an instruction to implement a predetermined algorithm.** Confirm the baseline
and choose the fastest correct approach on the actual machine. Keep alternatives
separate so their performance effects can be attributed.

## Objective and boundary

The ONNX Go runtime's fifth optimization investigation found a high-impact gap
in this library's ARM64 convolution **residual-column coverage**. Improve the
work that still uses `conv2DFourChannels`, including padded horizontal columns
and fully interior columns left over after four-column tiling. Preserve
accumulation semantics and every existing destination-validation guarantee.

Do not put a copy of this kernel into the ONNX repository. Do not combine this
change with pooling, arithmetic, activation, graph fusion, memory planning,
threading, dependency upgrades or unrelated cleanup. Those have independent
profiles and should be separate changes. Tail handling and padded-column
handling are related coverage problems; evaluate each separately, then combine
only if a single residual-kernel design naturally addresses both. A tail-only
change with a measured material gain is acceptable; report its remaining edge
cost explicitly.

## Exact measured versions and environment

Measurements taken 2026-11-10 on Apple M3, macOS arm64, Go 1.27.2:

* ONNX: `github.com/itsmontoya/onnx`, revision
  `85bc301` (full revision can be resolved in its checkout).
* Backend: published `neuralnetwork v0.10.1`, commit
  `ea17b585c42264ca85a637705608c284b80464bc`.
* Module sum: `h1:3dkabHz0YnFp++xvjEMaFGyehMhTeUWAQ+bscebdRrs=`.
* `GOMAXPROCS=1`, `CGO_ENABLED=0`, `GOARM64=v8.0`, `GOWORK=off`.
* ORT 1.19.2 CPU, sequential, one intra-op/inter-op thread, graph optimization
  **disabled**. Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2, protobuf 4.25.3.

Seven unchanged ONNX comparison invocations, each with 19 cases × 100 measured
calls and 10 warmups per case, give Go **204.771 µs** versus ORT **42.500 µs**
by median of run medians: **4.818×**. The median paired ratio is 4.745×;
paired ratios range 4.562–4.876×. A final verification gives 204.313 versus
44.042 µs, or **4.639×**. These are noise/configuration observations, not gains:
ONNX production code and its dependency remained unchanged during this pass.

Nine original `BenchmarkMNIST` repetitions, 1 second each, give **201.927 µs/op**,
range 196.486–204.426 µs, SD 2.496 µs, CV 1.24%, **179,440 B/op / 174 allocs/op**.
The benchmark's deterministic input is `float32(index%256)/255` shaped
`[1,1,28,28]`. It is distinct from the comparison driver's 19-case distribution.
Never change either workload to improve its reported number.

All 19 numerical/top-1 cases pass at `atol=1e-4, rtol=1e-5`, maximum absolute
Go/ORT error `1.9073486328125e-5`. Go output bits equal the previous pass.

## Relevant files and call paths

In this repository:

* `tensor/conv2d.go`: `Conv2DInto`, `conv2DValuesInto`,
  `conv2DChannelBlocksInto`, `conv2DFourChannels`, `conv2DFourValues`,
  `conv2DValue`.
* `tensor/conv2d_tile.go`: ARM64 `conv2DFourByFour`, clipped-row/span setup,
  bias application and destination stores.
* `tensor/conv2d_tile_arm64.go` and `tensor/conv2d_tile_arm64.s`:
  `conv2DInteriorTile`, the existing four-filter × four-column NEON kernel.
* `tensor/conv2d_tile_portable.go`: fallback behavior/build constraints.
* `tensor/conv2d_benchmark_test.go`: `BenchmarkConv2DInto`,
  `BenchmarkConv2DIntoNoBias` (the latter matches MNIST's nil Conv bias).
* `tensor/conv2d_test.go`, `tensor/conv2d_internal_test.go`,
  `tensor/conv2d_channels_internal_test.go`, `tensor/conv2d_tile_arm64_test.go`:
  geometry, reference, alias/error, dynamic weights, concurrency and bit tests.
* `tensor/conv2d_config.go`, tensor validation/destination/span helpers.

ONNX call path:
`Session.Run → evaluation.run → evaluation.execute → cnnCall.run →
tensor.Conv2DInto → conv2DValuesInto → conv2DChannelBlocksInto`.
This is the direct Into path, not a layer-training API or im2col matrix adapter.

The fixed network has 12 nodes:
`Conv → Add → Relu → MaxPool → Conv → Add → Relu → MaxPool →
Reshape`, plus an independent constant-weight `Reshape`, then `MatMul → Add`.
ORT with optimization disabled still executes all twelve nodes. Its speed
advantage in this baseline is not due to removing the Add/ReLU nodes.

## Workload and evidence

Both convolutions use kernel 5×5, groups 1, stride `[1,1]`, dilation `[1,1]`,
pads `[2,2,2,2]`, **nil bias**. Bias is added by a separate ONNX node.

| Convolution | Input | Weights | Output | Real-intermediate prepared latency |
| --- | --- | --- | --- | ---: |
| First, ONNX node 0 | `[1,1,28,28]` | `[8,1,5,5]` | `[1,8,28,28]` | 18.194 µs |
| Second, ONNX node 4 | `[1,8,14,14]` | `[16,8,5,5]` | `[1,16,14,14]` | 100.288 µs |

Five 300-ms repetitions produced these medians, excluding destination allocation
and ONNX call planning. Together the convolutions consume approximately **59%**
of full inference. ORT's instrumented means for these nodes are 8.39 and
26.08 µs; its profiler has microsecond resolution and overhead, so this is an
approximate comparison, not identical instrumentation.

Existing dispatch uses a four-column tile only when all four columns are
horizontal interior positions. For output width 28, SIMD starts at
`2,6,10,14,18,22`; columns `0,1,26,27` use `conv2DFourChannels`.
For width 14, SIMD starts at **2 and 6**; columns
**0,1,10,11,12,13** use the scalar helper. Columns **10 and 11 are fully
interior**. Only their residual tile width keeps them on the scalar path.

For the second convolution, scalar processing covers **42.86% of columns** and
**37.5% of valid MACs**. The first convolution has 14.29% scalar columns and
10.45% scalar MACs. Rows are already clipped by the interior wrapper.

A diagnostic benchmark partitions the **unchanged existing helpers**, with
synthetic finite values and exact geometry. Before timing, it combines scalar
and tiled partitions and compares all output bits to ordinary `Conv2DInto`.
Seven 500-ms repetitions yield:

| Existing helper work | First | Second |
| --- | ---: | ---: |
| Complete kernel | 17.477 µs | 99.482 µs |
| All scalar columns | **6.269 µs** | **78.400 µs** |
| Padded columns | 6.323 µs | 46.881 µs |
| Fully interior residual columns | None | **31.870 µs** |
| Existing SIMD tiles | 10.975 µs | 20.489 µs |

Padding and interior-tail rows are subsets of all scalar columns; do not add
overlapping measurements. Helper measurements have **0 B/op / 0 allocs/op**.
Full `Conv2DInto` is 32 B/op / 1 alloc/op. Partition traversal differs slightly
from full traversal, so **84.7 µs / 42% of inference is an estimate**, supported
by both geometry-matched partition timings and real-intermediate node profiles.

Scalar throughput in the second convolution is about **5.0 GFLOP/s**, versus
**32.0 GFLOP/s** for its tiled partition, counting two operations per valid MAC.
The scalar assembly has four scalar `FMADDS` plus indexing/control work; compiler
bounds analysis confirms checks at the input and four filter loads at
`tensor/conv2d.go:288–292` in v0.10.1. The interior assembly reaches `VFMLA`.
This is incomplete vector coverage, not globally failed SIMD dispatch.

Do not trust the whole-process CPU profile literally: this environment assigns
80.8% to `runtime.kevent` despite the process using almost a full core. A no-GC
profile also overattributes `madvise` relative to measured system CPU time.
The cause is unresolved. Conditional samples within `Conv2DInto` attribute
roughly 36% of the first and 77% of the second convolution to the scalar helper,
consistent with timed partitions. Use independent timing to cross-check profiles.

## Why memory or general graph work is not this task

An ONNX prepared-graph diagnostic reuses every destination and planned call,
executes all twelve kernels, and checks output agreement. It reduces allocation
from 179,440 B / 174 allocations to 41,920 B / 33 allocations but only changes full graph
latency from 201.927 to **187.449 µs**, about 7.2%. Nine paired `GOGC=off`
ablations save a median 1.668 µs. These are imperfect experiments, but neither
supports memory management as the next largest win.

Second-convolution cache rotation across 1, 64 and 1,024 independent tensor
sets measures 100.147, 100.880 and 101.275 µs; the largest numerical working set
is about 30.9 MiB. Cache residency alone does not explain the scalar cost.
No reliable hardware counters were available, so precise cache/bandwidth or
instruction-throughput limits remain unestablished.

Other independently measured costs are broadcast biases ≈25.8 µs, max pooling
≈20.9 µs, and ONNX ReLU correction ≈10.4 µs. These deserve separate work.
Full inference is 327.346 µs under `purego`, confirming the current ARM64 tile
already helps substantially. Preserve and test that portable path.

## Approaches to evaluate

1. Extend acceleration to one/two/three residual columns and clipped horizontal
   windows, using a proved input/filter span and preserving each output's
   channel/row/column accumulation order. Measure tail and padding separately.
2. For the fully interior tail, consider an overlapping final four-column tile
   whose stores repeat already-computed outputs, or a dedicated partial-width
   tile. Repeated outputs must be identical and all loads/stores in bounds.
   Compare redundant arithmetic and wrapper costs against scalar work.
3. Evaluate an ARM64 four-filter residual kernel or an explicitly checked-span
   scalar kernel to remove repeated inner bounds checks and shape/address work.
   Vectorizing across filters may need gathering strided weights or packing;
   measure total preparation cost and reject hidden per-call packing overhead.
4. Hoist per-output geometry and slice proofs outside the inner MAC loop where
   correctness permits. Compare against existing SIMD coverage rather than
   assuming assembly alone is necessary.

Use a general fallback for groups, filter tails, unsupported stride/dilation and
empty/padded geometries. Do not apply a model-name or exact-weight special case.
Packed or panel-based convolution is a larger alternative worth considering
only if profiling shows the small residual approach cannot meet the goal.
It introduces scratch, mutable-weight invalidation and nonfinite-padding
questions; do not silently impose a cache or workspace contract on `Conv2DInto`.

## Correctness requirements

* Preserve validation and every error leaving destination unchanged.
* Destination remains disjoint; do not mutate inputs/weights/bias or weaken
  alias detection. Honor dynamic weights and concurrent independent calls.
* Keep accumulation ordering and bias addition after the complete reduction.
  Preserve existing float32/FMA behavior and output-bit expectations.
* **Skip padded products**, including products involving NaN or infinite weights.
  Replacing excluded samples with numeric zero is not equivalent for nonfinite
  weights. Check top/bottom and left/right edges and entirely padded windows.
* Preserve NaN propagation/operand behavior, signed zero, infinities and existing
  tolerance policies. Do not edit goldens to accommodate an optimization.
* Check smallest spans and payload endings: no over-read, over-write or unsafe
  assumptions about spare capacity, alignment, row padding or adjacent planes.
* Include filter counts below/above four, nonmultiples of four, group boundaries,
  multiple batches, empty tensors, negative pads where supported, heavy padding,
  stride/dilation fallbacks, optional bias and nonzero finite payloads.
* Preserve ARM64 `!purego` build constraints; default portable, `purego`, race
  and supported amd64 builds must continue to work.
* Do not introduce goroutines or thread-count changes. Keep Into scratch/allocation
  guarantees; ideally retain 32 B/op / 1 alloc/op for full Conv2DInto.

## Measurements and acceptance

Start with this repository's normal full test suite and the existing
`BenchmarkConv2DIntoNoBias/{MNISTFirst,MNISTSecond}`. Inspect the benchmark file
before running; confirm names and fixture setup rather than guessing.
Precompile binaries and run benchmarks serially, alternating baseline/candidate
order across at least seven 500-ms or longer pairs. Record raw samples,
medians, dispersion, allocations and bytes. Profile the full Into call and the
residual/interior components before and after.

Also run the existing `BenchmarkConv2DInto` grid to detect regressions in
non-MNIST geometries and optional bias. Add readable tests for any newly reached
tail/edge path; explicitly compare optimized and portable/scalar references
with special values. Do not create tests that merely restate dispatch conditions.

If the ONNX checkout is available at
`/Users/itsmontoya/go/src/github.com/itsmontoya/onnx`, its pass-5 evidence is in
`tools/compare_runtime/pass5/`, with optional `BenchmarkMNISTNodes` and
`BenchmarkMNISTPipeline` in the root package. The helper-partition source is
`tools/compare_runtime/pass5/conv_paths_test.go.txt`: copy it into a temporary
backend checkout as `tensor/pass5_diagnostic_test.go`. Do not use a module-cache
overlay; Go disallows replacing files beneath `GOMODCACHE`.

For ONNX integration, use an isolated temporary copy or explicit temporary
workspace referencing the candidate backend. Do not persist a replace directive
or change the user's ONNX dependency. Keep Go build flags and the full original
benchmark unchanged:

```sh
GOMAXPROCS=1 CGO_ENABLED=0 GOARM64=v8.0 go test -c -o /tmp/onnx-candidate.test .
GOMAXPROCS=1 /tmp/onnx-candidate.test -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=1s -test.count=7 -test.benchmem
```

Use `GOWORK=off` for published baseline measurements; use the explicit temporary
workspace only for candidate integration. Keep measurements separate from
compilation/tests/profiling, and reverse pair order to account for thermal drift.

If fixtures are absent, ONNX's `tools/compare_runtime.py` prepares the pinned
MNIST archive (26 kB), adapts IR7/opset12 to IR8/opset13 and checks it. It uses
the ONNX Model Zoo revision `4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177`, archive
`validated/vision/classification/mnist/model/mnist-12.tar.gz`, archive SHA-256
`a53a59dcaca8804a0f6dfda9a3cf2e082979589391dbde73640b60684f1d24e9` and adapted
model SHA-256 `310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
Use its pinned requirements and default 19 cases, tolerances, seed and one-thread
ORT-disabled configuration. Do not silently compare to ORT with fusion enabled.
If the ONNX repository is unavailable, clearly report that limitation and create
the two exact-shape direct-kernel benchmarks above; do not claim an inference
speedup solely from a backend microbenchmark.

The residual partition fraction `f≈0.42` means a 2× improvement should save
about **21% of end-to-end latency**: `1/((1-f)+f/2)≈1.27×` speedup. A 4× residual
improvement would save about 31%. These are Amdahl estimates, not promises.
Accelerating just the two interior tail columns from 31.9 to about 5 µs would
save roughly 13% by itself.

Target **at least 15% lower unchanged ONNX inference latency**, with **10% a
useful minimum** if the focused implementation is simpler and demonstrably safe.
For a tail-only implementation, a convincing ≥10% inference gain is meaningful.
Reproduce the baseline on your machine before using absolute targets; second
convolution around 60–75 µs would be a useful direction from the measured
100 µs. Require unchanged full-inference allocation cost or explain any measured
tradeoff before accepting it. Investigate repeatable regressions above 3–5% in
the existing kernel grid with longer measurements; do not dismiss them as noise.

Finish with full default, `purego`, race and applicable platform checks, focused
edge/tail correctness tests, unchanged 19-case ONNX/ORT numerical agreement when
available, and a before/after profile. Report measured facts separately from
inference and speculation, including the remaining scalar fraction and whether
padding and tail changes should be reviewed together or independently. Provide
before → after → ORT → slowdown, allocations and bytes, commands, raw samples,
and remaining limitations. Do not alter outputs, workloads, case counts,
tolerances or precision to meet the target.
