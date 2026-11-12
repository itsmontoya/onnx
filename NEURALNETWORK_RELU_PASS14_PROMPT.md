# Standalone investigation: preserving ReLU after ONNX Performance Pass 13

Work in **github.com/itsmontoya/neuralnetwork**. Investigate the remaining
preserving-ReLU cost before choosing an optimization. Follow this repository's
AGENTS.md and style/contribution guidance. Do not begin with a SIMD or assembly
implementation. A measured rejection is a valid result.

This prompt is independently usable; reading the ONNX Pass 13 report is not
required. Do not change the ONNX repository or other numerical operators as part
of this investigation. Read-only inspection and optional end-to-end validation
against an available ONNX checkout are useful.

## Measured context and scope

The ONNX baseline is commit `9f1905e200e1998ba522933bb26af5b3498a7136` plus the
accepted Pass 13 constant-only Reshape preparation. Its dependency is
**neuralnetwork v0.10.7**, commit
`1f033b7d13788661a9002462151a997f30b620b4`, with no module replacement. Record
your current HEAD and any differences from this baseline; preserve later
accepted changes if the repository has advanced.

Host: Apple M3, macOS **26.5.2 (25F84)**, darwin/arm64, Go **1.27.2**,
**GOARM64=v8.0**, **GOMAXPROCS=1**, **CGO_ENABLED=0**, GOWORK=off, default
optimization, no build tags/gcflags. Host CGO_ENABLED defaults to 1. Benchmark
purego separately. Record actual OS, architecture, Go, CPU, environment,
dependency versions and dispatch before reproducing any result.

Fresh final ONNX measurements, seven alternating whole-runtime pairs:

| Scope | Before | After | Reference |
| --- | ---: | ---: | ---: |
| Complete fixed-policy Go Run | 62.375 µs | **61.750 µs** | ORT **44.167 µs**, **1.3981×** |
| Ordinary ramp Go Run | 63.026 µs | **62.114 µs** | Separate workload |
| Run allocation | 103,691 B / 93 | **93,322 B / 88** | — |
| Stripped preallocated ramp graph | about 52.0 µs with constant copy | **51.594 µs** without it | Not comparable in scope to full ORT inference |

Pass 13 removes the repeated `[16,4,4,10] → [256,10]` weight Reshape, retaining
one immutable 10,240-B payload per shared plan. The production saving is only
**0.625 µs / 1.0020%**, accepted under the conditional guideline. Pass 11 constant-clone elimination and Pass 12 operator
metadata preparation are already included. None may be counted again.

Final Go report medians range **59.542–64.042 µs, CV 2.29%**; ORT medians range
**41.625–45.041 µs, CV 2.66%**. Final Go within-report p95 median is 112.5923 µs.
The initial Pass 13 baseline was 62.2705 / 42.000 = 1.4826×; ORT reference
movement is real and must not be disguised as Go optimization. Whole-Run paired
savings are noisy, including one negative pair; paired median saving is 0.9165 µs,
range −1.5835–2.666, CV 155.78%. The primary result remains the difference of
engine medians, 0.625 µs. Three earlier complete seven-pair sets gave primary
savings 1.1035, 1.479 and 0.916 µs; later changes refined only load-time quota
guards and left the eligible MNIST inference algorithm unchanged. Retain all
valid samples and do not substitute the larger paired saving for the primary one.

The complete comparison policy is fixed: pinned MNIST model, upstream/zero/one
and sixteen seed-0 uniform random inputs, **19 cases**, **10 warmups + 100 timed
calls per case**, atol=1e-4, rtol=1e-5. Go then ORT run sequentially with owned
outputs. Build, loading, input encoding and output serialization are excluded.
ORT **1.19.2**, CPUExecutionProvider, ORT_SEQUENTIAL, one intra-op/inter-op thread,
ORT_DISABLE_ALL, CPU arena and memory pattern enabled. Python 3.9.6, NumPy
1.26.4, ONNX 1.16.2, protobuf 4.25.3. Preserve input fingerprints, tolerances,
conversion, upstream/top-1 checks, timing boundaries and output ownership.

The optional ONNX fixture is
`.cache/compare-runtime/mnist-ir8-opset13.onnx`, SHA-256
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
An available checkout may be at `../onnx`; its `tools/compare_runtime.py` and
`mnist_diagnostic_benchmark_test.go` contain the established harness. The ordinary
ramp input is `[1,1,28,28]`, value `float32(index%256)/255`. ReLU operands are the
post-bias-Add activations of this actual graph. Do not substitute synthetic input
distributions and describe their timings as exact MNIST reproduction. If the
fixture is unavailable, report that limitation and use explicit representative
datasets while keeping the backend investigation independently useful. Tests
must not download optional fixtures during normal execution.

## The fresh ReLU evidence

Five repetitions per prepared-node scope, preallocated disjoint destinations:

| Operation | Elements | Complete ONNX adapter median µs | Range / CV | Preserving backend median µs | Range / CV |
| --- | ---: | ---: | --- | ---: | --- |
| ReLU32, `[1,8,28,28]` | 6,272 | **4.974** | 4.957–4.998 / 0.36% | **4.988** | 4.941–5.010 / 0.57% |
| ReLU114, `[1,16,14,14]` | 3,136 | **2.385** | 2.381–2.404 / 0.40% | **2.385** | 2.378–2.414 / 0.59% |
| Sum of medians | **9,408** | **7.359** | Separate component medians | **7.373** | Separate component medians |

All these scopes allocate **0 B / 0 allocations**. The historical ONNX benchmark
label `WithCorrection` now invokes the current one-pass preserving adapter; it
does not mean the old finite-input snapshot/correction implementation is still
used. Before Pass 13, prepared adapter sum was 7.277 µs. This small movement is
not caused by a ReLU implementation change.

The separate **legacy ReLUInto** diagnostic, which converts NaNs to +0 and has
no context argument, measures **4.296 + 2.096 = 6.392 µs**. Its cheaper semantics
make it unsuitable as a production replacement. The approximately 0.981-µs
preserving/legacy difference is not a cancellation-only cost measurement.
Nevertheless, much of the total time exists even in that simpler operation.

Fresh ORT profiled full-graph ReLU node means sum to approximately **2.000 µs**
(three repetitions: 2.000, 1.997, 2.012). ORT events have integer-microsecond
duration resolution and include a different scope than isolated unprofiled Go
adapters. The **5.359-µs raw indication is not a demonstrated recoverable saving**.
Do not promise that ReLU optimization can deliver it.

At one float32 read and one write, these calls move at least:

- First: 25,088 B read + 25,088 B written = **50,176 B**.
- Second: 12,544 B read + 12,544 B written = **25,088 B**.
- Total: **75,264 B**, or about **10.21 GB/s** at the 7.373-µs backend sum.

This is a useful traffic estimate, not proof of a DRAM bandwidth bottleneck:
working sets may be cached, stores can incur additional traffic, and validation
also reads metadata. Determine the actual constraint. Report cycles/element only
with a supported cycle/frequency measurement, not an invented M3 clock speed.

## Establish the current algorithm and contracts

Inspect at least `tensor/relu_preserving_nan.go`, `tensor/activation.go`,
`tensor/destination.go`, Tensor storage/identity validation, tests and dispatch.
Explain exact validation, alias checks, setup, loop, cancellation and destination
behavior. At v0.10.7 the preserving routine:

1. Validates float32 input, then exact-shape/disjoint destination.
2. Rejects nil context after tensor validation; checks ctx.Err before traversal.
3. Traverses chunks of **at most 1,024 elements**, polling before every chunk.
4. Uses float32 bits: NaN if `bits & 0x7fffffff > 0x7f800000`; records hasNaN;
   otherwise a set sign bit produces +0; writes one float32 per element.
5. Checks ctx.Err after traversal, including an empty traversal.

Source and the fresh Apple M3 objdump show **scalar FMOVS loads/stores, integer
bit tests/branches and a destination bounds check** in this loop. There is no
preserving SIMD dispatch in the observed path. Confirm on your checkout, inspect
compiler output and bounds-check elimination, and determine why that code was
generated. An ARM scalar register move using an FP register is not evidence of
vector processing. Do not assume assembly is necessary or that adding it is the
best first change.

Preserve exactly:

- Positive finite values/subnormals and +Inf, bit for bit.
- Negative finite values/subnormals and −Inf become **+0**.
- +0 remains +0; **−0 becomes +0** under the existing contract.
- Every original NaN bit, including payload, sign and signaling bits.
- Correct hasNaN on success; context errors return hasNaN=false.
- Input-before-destination-before-context error precedence.
- Nil/zero receivers, dtype, rank/shape, alias checks including scalar/empty
  shallow aliases; inputs and validated-error destinations remain unchanged.
- No retained caller references, no payload scratch, read-only shared input and
  independent concurrent destinations.

After successful validation, polling is pre-traversal, before every ≤1,024-element
chunk including the first, and post-traversal. These MNIST sizes mean **9 + 6 =
15 backend ctx.Err calls**. Cancellation may leave a rectified prefix and an
unchanged suffix, or a fully written destination if caught at the last check;
pre-traversal cancellation leaves it unchanged. Audit exact tests and preserve
the promised behavior. Polling is **already chunked**; do not propose the existing
1,024-element scheme as a new optimization. Do not silently relax poll frequency,
ordering, error precedence, partial writes or empty-call checks.

The ONNX adapter adds contracts beyond the backend: on cancellation it restores
the legacy fully rectified private destination; on NaNs it clones the preserving
result and restores the supplied destination to legacy NaN-to-zero values. Finite
MNIST avoids that clone. Preserve hasNaN and cancellation behavior so this adapter
continues to work correctly. Do not modify ReLUInto's separate NaN-to-zero API
contract or weaken preserving behavior for speed. A preserving API already exists;
justify any additional API rather than silently replacing its contract.

## Investigation before optimization

Reproduce both complete-contract calls separately and together using actual
operands where available. Measure medians, ranges, CV, B/op, allocs/op and p95
only where individual samples exist. Run performance processes sequentially;
alternate control/candidate order and retain all valid samples. Keep allocation
and CPU profiles separate from latency measurements. Darwin CPU samples in ONNX
were dominated by runtime.kevent and did not resolve small costs; do not infer
precise attribution from that artifact.

Decompose validation, destination validation/aliasing, setup, interface/context
polling, scalar traversal, any existing SIMD path, function-call overhead and
memory traffic. Use bounded diagnostic ceilings:

- Current complete preserving backend contract.
- Validation-only and validation-free traversal.
- Context/polling-only and cancellation-free traversal.
- Raw copy bandwidth using equivalent source/destination residency.
- Raw threshold loop and bit-preserving scalar loop.
- Portable-Go loop alternatives and compiler-generated code.
- A preserving SIMD diagnostic only if evidence makes it worthwhile.

Diagnostics may omit checks to identify a ceiling; production acceptance must
restore every required check/contract. Their savings overlap and cannot simply
be added. Use output bits and non-escaping benchmark sinks to prevent dead-code
elimination. Separate zeroing/allocation from preallocated kernel work.

Test real finite distributions, all positive, all negative, zeros, alternating
signs, branch-unpredictable inputs, sparse/dense NaNs and subnormals. Cover sizes
0, 1, chunk/vector boundaries and tails, both MNIST sizes and larger/smaller
working sets. Vary cache residency and alignment where useful; do not call a
hot-cache ceiling a production bandwidth result.

Investigate portable Go, purego, ARM64 NEON and AMD64 SIMD paths. Begin with what
the compiler and existing dispatch actually do. Explain missing dispatch or
vectorization before changing it. Check whether branch removal, bounds-check
elimination, loop structure or compiler-friendly Go suffices. Only introduce
architecture-specific code when a measured complete-contract benefit justifies
its maintenance. Preserve the portable fallback; build/test supported targets.

## Acceptance and end-to-end relevance

Do not optimize solely to make a raw loop look better. Require meaningful
complete-contract backend improvement at these actual sizes, with bitwise tests,
unchanged cancellation/destination/alias behavior and no new allocations. A
proposed ≥1-µs aggregate backend improvement would be worth testing in ONNX; smaller
changes need proportionately lower complexity. This is a decision guideline,
not a benchmark target or permission to remove semantics.

If end-to-end validation is available, use a temporary local module replacement
and the same ONNX model/policy; report it as a local experiment, with exact backend
commit/source hashes. Do not publish, bump ONNX dependencies or change ONNX
production behavior as part of this investigation. Distinguish backend recovery
from demonstrated whole-Run recovery; record initialization/memory costs too.

Current target arithmetic at **44.167 µs ORT**:

- 1.25× = **55.20875 µs**; Go 61.750 needs **6.54125 µs** further saving.
- 1.10× = **48.5837 µs**; further saving required is **13.1663 µs**.
- Even all of the coarse 5.359-µs ReLU indication would nominally leave 1.18225 µs
  to 1.25×. Scope mismatch prevents treating that indication as a real ceiling.

Other fresh prepared Go/coarse ORT sums: Conv **34.938 / 36.387 µs**, Pool
**3.584 / 3.017**, Add **3.582 / 2.846**, MatMul **1.717 / 1.000**. Conv has no
demonstrated positive excess and remains closed. Do not optimize Conv, Pool,
MatMul or fusion in this task. ReLU is the strongest next investigation signal,
but a single-bottleneck route to 1.25× is unproved. Do not promise either target.

## Required deliverable

Produce a self-contained report with the current cost model, bottleneck evidence,
architecture/dispatch and compiler findings, realistic recoverable saving,
proposal or explicit rejection, semantic proof, tests, benchmark statistics and
expected ONNX impact with its confidence/scope limits. If a production change is
justified, implement the smallest coherent change in neuralnetwork and complete
its correctness/race/purego/cgo-disabled/cross-build checks. Otherwise retain the
current implementation and explain the measured limit.

No semantic weakening. No counting prior constant/metadata savings again.
No predetermined SIMD. Investigate, measure, then decide.
