# Pass 16 — Published NeuralNetwork v0.10.10 Integration

2026-11-13, Apple M3. ONNX now uses published **NeuralNetwork v0.10.10**.
Complete inference allocates **92,874 → 92,826 B/op / 73 → 72 allocs/op**.
The equal-shape Add accounts for the entire saving: **336 → 288 B / 7 → 6
objects**, removing one invocation-lived result copy. MatMul remains unchanged.

Seven fresh alternating fixed-policy version pairs measure **54.9795 → 54.625
µs** Go, a 0.3545-µs / 0.6448% difference of medians. Paired saving median is
only **0.021 µs**, and the separate ordinary ramp is **54.945 → 55.264 µs**.
These timing results do not establish a durable whole-Run speedup. The allocation
transfer is exact and reproducible.

## Released source and scope

ONNX measurement baseline: `0803d3ea12ae02597592d014e4792c95ec5ca364`.
Dependency integration: `38933a4` (`Upgrade neuralnetwork to v0.10.10`).
Measurements ran before integration commits, so reports record that HEAD with a
dirty working tree containing the dependency and documentation changes.

| Published backend | Commit | Module checksum |
| --- | --- | --- |
| v0.10.9 control | `740a244bf56214bb3443a69287a2d800674620e4` | `h1:kmfxkbiZZ8fbw0cxCxvar2nfYJ2Bx6kci/QUNH0w6oQ=` |
| v0.10.10 candidate | `2ffa5bd9fd2034382c49d14efc732ea4d026aa09` | `h1:CcoQQN8w9aGodz61FQGKSWtbnKzAHf/gtXbukwVUkj0=` |

The backend change from [PR #128](https://github.com/itsmontoya/neuralnetwork/pull/128)
uses the existing `Matrix.ValuesInto` with invocation-local captured tensor
slices. It preserves private input/output matrices and arithmetic dispatch,
while removing `Values()` allocation and its extra copy. Its
[backend report](https://github.com/itsmontoya/neuralnetwork/blob/v0.10.10/Benchmarks_matrix_adapters.md)
documents the failure audit, tests, alternatives and backend measurements.

ONNX's production integration changes only **go.mod/go.sum**, v0.10.9 → v0.10.10.
No public API, indirect dependency, numerical implementation or workspace charge
changes. The two versions are published modules without replace directives;
the control uses separate module files, the candidate uses normal go.mod.
No local backend checkout supplies production code and no backend files were edited.

The [original investigation and standalone handoff](PERFORMANCE_PASS16.md)
remain historical evidence; their samples are not pooled into these results.

## Environment and unchanged comparison policy

Mac15,13, Apple M3, 8 cores (4 performance / 4 efficiency), 24 GB;
macOS 26.5.2 (25F84), Darwin 25.5.0, darwin/arm64, Go 1.27.2.
Performance builds use **GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off, GOARM64=v8.0**,
default optimization, no tags/gcflags. Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2,
protobuf 4.25.3 and ORT 1.19.2. Go dependencies remain protobuf v1.36.12,
simd v1.3.0 and x/sys v0.45.0.

The policy is unchanged: nineteen sorted upstream/zeros/ones/sixteen seed-0
uniform-random cases, ten warmups and 100 timed calls per case, owned outputs,
Go then ORT, atol=1e-4 / rtol=1e-5. Builds, loading, input conversion and
serialization stay outside inference timing. ORT uses CPUExecutionProvider,
sequential execution, one intra/inter-op thread, disabled graph optimization
and its default arena/memory pattern. Model SHA-256 remains
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
Driver, runner, model and every input fingerprint match the historical policy.

## Complete-runtime transfer

All seven alternating pairs and every valid sample are retained. The primary
estimator is the difference of medians of per-report engine medians; paired
differences are reported separately. CV uses population standard deviation.

| Released version | Go median µs | Go range / CV | ORT median µs | ORT range / CV | Go/ORT |
| --- | ---: | --- | ---: | --- | ---: |
| v0.10.9 | 54.9795 | 53.292–55.125 / 1.28% | 42.250 | 41.792–43.8125 / 1.47% | 1.3013× |
| v0.10.10 | 54.6250 | 54.209–55.2295 / 0.58% | 42.0835 | 41.625–44.084 / 2.04% | 1.2980× |

Go movement is 0.3545 µs / 0.6448%; paired saving median **0.021 µs**, range
**-1.021–0.6045 µs**. ORT moves 0.1665 µs faster, separately from Go movement.
The ordinary-ramp medians are **54.945 → 55.264 µs**, ranges 54.263–55.810 /
53.637–55.575. Neither its timing nor profiled allocations are used as the
fixed-policy Go/ORT numerator. There is no established 1.25× milestone.

All ramp repetitions reproduce **92,874 → 92,826 B / 73 → 72 allocations**:
48 B / one object removed per Run. Full-Run rate-one profiles remove the Add
`Matrix.Values` allocation site, retaining both input matrix adapters, output
matrix and MatMul planning/staging. Profiles include the harness calibration
call as well as the requested 1,000 calls; tiny-object sampling is not substituted
for ordinary allocation counters.

All fourteen default comparison reports pass numerical, conversion, upstream
and top-1 checks: **266 case comparisons / 26,600 timed calls per engine**.
The separate purego report adds nineteen cases / 1,900 calls per engine and is
not pooled into default timing. Go output bits agree across versions, repetitions
and purego; maximum Go/ORT absolute error remains **1.9073486328125e-5**.

## Components and limits of attribution

Seven 200-ms repetitions on actual ramp intermediates, pre-resolved numerical
adapters with disjoint existing destinations. Session dispatch, destination
allocation, retirement and publication are excluded.

| Component | v0.10.9 µs · B · allocs | v0.10.10 µs · B · allocs |
| --- | ---: | ---: |
| 01 channel Add | 1.8820 · 0 · 0 | 1.8730 · 0 · 0 |
| 05 channel Add | 1.1580 · 0 · 0 | 1.1680 · 0 · 0 |
| 11 equal-shape Add | 0.1917 · 336 · 7 | 0.1812 · 288 · 6 |
| 10 MatMul | 1.6220 · 144 · 3 | 1.6600 · 144 · 3 |

The selected Add component indication is **0.0105 µs**, consistent with the
small backend result and well below the earlier diagnostic ceiling. All three
Add medians sum to **3.2317 → 3.2222 µs**. Complete fixed-policy movement exceeds
that indication by about 0.345 µs; no single mechanism was isolated for the
excess. The paired estimator and ordinary ramp do not support crediting that
extra movement to result copying. Channel Add outliers and unchanged MatMul
movement remain in the evidence; no unrelated kernel speedup is claimed.

Source-derived Add copying falls **160 → 120 logical bytes** for ten floats:
two 40-B input copies and one output-to-tensor copy remain. Copy read/write
traffic is 320 → 240 B, excluding metadata, zeroing and arithmetic; this is not
a DRAM bandwidth measurement. No new retained state, payload, initialization
step or public borrow/view API is introduced. The backend reports a compiler-
dependent 48-B larger ARM64 adapter stack frame, distinct from the removed heap
object. No peak-memory or RSS reduction is claimed.

LoadBytes medians are **373.640 → 377.222 µs**, roughly 345.3 KB / 2,563
allocations, with retained outliers of 345,807 B / 2,566 allocations in each arm.
Session creation is **25.97 → 26.06 ns**, both 24 B / one allocation. These
measurements do not establish an initialization optimization.

## Validation and integration contract

Fresh ONNX and published-backend suites pass in default, CGO-disabled, purego,
race and purego+race configurations. Vet, formatting, tidy-diff, module
verification, supported cross-builds and all four existing bounded fuzz smoke
targets pass. CGO-disabled default/purego cross-builds cover darwin/arm64,
darwin/amd64, linux/arm64, linux/amd64, windows/amd64 and linux/386; cross-builds
are compilation checks, not nonnative execution.

ONNX's native Metal suite also passes. The first unrestricted backend Metal
suite had one failure in `Test_SequentialResidentLongRunningMixedStress`: its
resource snapshot reported more live buffer bytes after GC. Five isolated
repetitions pass in **each** published version, and one complete Metal suite per
version passes with package execution serialized (`-p=1`). Native Matrix/device
Metal tests pass with no skips. Initial sandbox runs skipped GPU tests because
no default device was exposed; both those logs and the host reruns are retained.
The model/device code is identical across these releases, but the single stress
failure was not reproduced or explained, so it is **not claimed resolved**.

Existing suites retain input/initializer bits, owned output survival across
Runs/closure, dynamic/symbolic/default/override cases, destination alias rules,
cancellation/deadlines, failure recovery, Busy/Close behavior, concurrent sessions
and load/live/workspace limits. The backend's released focused tests additionally
cover SIMD boundaries, IEEE values, read-only inputs, destination backing storage,
changed current weights, lifetime and private-output error atomicity.

The existing equal-shape Add four-buffer scratch charge stays conservative;
host/Metal MatMul charges and complete destination staging remain necessary and
unchanged. No dependent ONNX implementation or additional optimization candidate
is bundled. Complete raw evidence and replay instructions are in
[the release evidence directory](tools/compare_runtime/pass16/release/README.md).
