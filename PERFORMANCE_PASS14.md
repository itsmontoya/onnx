# Performance Optimization Pass 14 — Released Preserving ReLU

2026-11-12, Apple M3. **Validated integration of released neuralnetwork v0.10.8.**
Seven alternating fixed-policy pairs measure **61.5415 → 55.1455 µs Go**:
**6.396 µs / 10.3930%** faster. Allocations remain **93,322 B/op · 88 allocs/op**.
The candidate's fresh ORT median is **42.417 µs**, giving **1.3001× ORT**.
No local module replacement or ONNX numerical workaround is used.

The earlier local experiment's 1.2521× ratio used ORT at 44.208 µs. The released
Go timing is slightly lower, but today's candidate ORT reference is also faster.
The 1.25× milestone has not been established on this fresh reference.

## 1. Baseline and release provenance

ONNX baseline: `47271fa3dd7d641486c3c7b547d5bc89dc24ebad`.
Dependency integration commit: `9327c919b3c47ccc1705df1606fed082c14cc56e`.
Measurements used the same module contents before that commit; report metadata
therefore records the baseline HEAD with the dependency working tree dirty.
The only runtime change here is the dependency **v0.10.7 → v0.10.8** and its
checksums. The control uses an unchanged v0.10.7 module file outside inference
timing; the candidate uses the normal go.mod, with no replace directive.

| Dependency | Commit | Module checksum |
| --- | --- | --- |
| neuralnetwork v0.10.7 | `1f033b7d13788661a9002462151a997f30b620b4` | `h1:LPdDwSZz5MkiYGL1UAiMVZmpITvu5xVX4PHKoaf4BEw=` |
| neuralnetwork v0.10.8 | `be95e90b7f99e5ab59213f360a8b79980def6efa` | `h1:J/ont78/Q9d0OzzemUem6yNw0cI1gIt7Pi3sCF5Qjas=` |

v0.10.8 is the published release of
[neuralnetwork PR #126](https://github.com/itsmontoya/neuralnetwork/pull/126).
All four production ReLU files match the exact source fingerprints validated
in the earlier Pass 14 investigation. `go mod verify` passes. No indirect
dependency version changes.

Fresh host: Apple M3, Mac15,13, 8 cores (4 performance/4 efficiency), 24 GB;
macOS **26.5.2 (25F84)**, Darwin kernel **25.5.0**, **darwin/arm64**, Go **1.27.2**.
Performance processes use **GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off,
GOARM64=v8.0**, default optimization, no gcflags and no build tags. Python
**3.9.6**, NumPy **1.26.4**, ONNX **1.16.2**, protobuf **4.25.3**, ORT **1.19.2**.
ORT uses CPUExecutionProvider, sequential execution, one intra-op/inter-op thread,
disabled graph optimization, and the default CPU arena/memory pattern.

The pinned model SHA-256 remains
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
Every report agrees on model, input, driver and runner fingerprints. Complete
provenance and statistics are in
[summary.json](tools/compare_runtime/pass14/data/summary.json).

## 2. Cost decomposition

The backend investigation preceded production implementation. Its original
complete pair measured **7.069 µs**; the fresh released-dependency preserving
backend component sum is **7.121 → 1.314 µs**, allocation-free in both versions.
These are different measurement scopes, explicitly retained separately.

The original execution path validates input type/storage/shape, validates exact
destination shape and disjoint identity, rejects a nil context, checks context
before traversal, before each interval of at most 1,024 elements, and after
traversal. Each scalar element is loaded, classified by integer NaN/sign bits,
rectified and stored. Successful validation has no payload allocation.

Earlier controlled diagnostics on the actual two ramp operands isolate roughly
**0.06845 µs validation**, **0.017317 µs polling**, and **0.4826 µs hot copy**,
against **7.065 µs complete preserving traversal**. Removing validation gives
only a small diagnostic difference; removing cancellation gives no material win.
The operation moves at least **37,632 B read + 37,632 B written** across 9,408
elements. These cached traffic estimates are not DRAM bandwidth measurements.

The dominant demonstrated mechanism is scalar instruction/branch/loop throughput.
The small check-removal ceilings overlap the kernel work and are not added to it.
The [earlier backend diagnostics](tools/compare_runtime/pass14/data/backend-investigation.json)
and [source identity](tools/compare_runtime/pass14/data/backend-investigation-identity.json)
are retained as pre-release evidence, rather than relabeled as fresh integration
measurements.

## 3. Generated-code findings

The original ARM64 loop is scalar: load, float/integer register transfer,
magnitude mask and NaN comparison, NaN/sign branches, conditional zero,
destination addressing/bounds check, store, and loop control. No preserving SIMD
dispatch or compiler vectorization exists in v0.10.7. Legacy ReLUInto is scalar
too, with an initial zero store and another store for positive elements.

Destination slicing removes per-element bounds checks but loses in the measured
candidate. A portable mask reduces branches but remains scalar. Only after these
diagnostics did the investigation test NEON.

The released integer NEON loop handles four lanes with ten instructions per
vector and a final NaN-mask reduction. It performs no floating-point arithmetic
or comparisons and has no data-dependent lane branch. Go validates/bounds slices,
retains the scalar tail and polls at exactly the old boundaries. The pre-release
text-size measurement adds 336 bytes, with no initialization, scratch allocation
or persistent numerical payload. The purego public function's normalized
168-instruction listing matches the original. Cycles/element were not measured
and are not inferred from a nominal M3 clock.

## 4. Controlled diagnostic results

These earlier actual-operand results justified the backend change; they are sums
of component medians, five repetitions per operand. All rows allocate **0 B/op /
0 allocs/op**. Traversal traffic is at least **37,632 B read / 37,632 B written**;
GB/s uses that lower bound. Legacy actually writes another 12,844 B on positive
elements. Validation/polling rows have no payload traffic.

| Diagnostic | Both calls µs | ns/element | Effective GB/s |
| --- | ---: | ---: | ---: |
| Raw copy | 0.4826 | 0.0513 | 155.95 |
| Raw positive threshold | 5.341 | 0.5677 | 14.09 |
| Legacy ReLUInto | 6.064 | 0.6446 | 12.41 |
| Original complete preserving, initial set | 7.065 | 0.7510 | 10.65 |
| Original complete preserving, second set | 6.998 | 0.7438 | 10.75 |
| Exact-source validation-free | 6.891 | 0.7325 | 10.92 |
| Exact-source cancellation-free | 7.060 | 0.7504 | 10.66 |
| Validation-free chunk 256 / 1,024 / 4,096 | 7.056 / 7.039 / 7.033 | 0.7500 / 0.7482 / 0.7476 | 10.67 / 10.69 / 10.70 |
| Flat Go bit-preserving loop | 7.584 | 0.8061 | 9.92 |
| Simple Go float-preserving loop | 7.885 | 0.8381 | 9.55 |
| Portable mask, full contract | 5.620 | 0.5974 | 13.39 |
| Sliced destination, full contract | 7.843 | 0.8337 | 9.60 |
| NEON prototype, full contract | 1.3149 | 0.1398 | 57.24 |

The diagnostic data retain per-size ranges/CVs, synthetic positive/negative,
signed-zero, unpredictable-sign, sparse/dense NaN and subnormal patterns, vector
and poll boundaries, both MNIST sizes and larger tensors. A rotating 205.5-MB
working set retains a substantial benefit but is a different cache experiment.
Tiny-call differences are not overinterpreted. No percentile is invented from
benchmark means, and no diagnostic ceiling is counted as an additional Run win.

## 5. Exact semantic contract

Positive finite values, positive subnormals and +Inf retain their original bits.
Negative finite values, negative subnormals, -Inf and -0 become +0. +0 stays +0.
Every NaN retains sign, payload and signaling/quiet bits; hasNaN reports whether
any input NaN was present.

Magnitude bits above `0x7f800000` classify NaNs without touching floating-point
state. NEON clears only negative non-NaN lanes; OR-reduction reports NaNs.
It therefore preserves signaling bits and positive subnormals as integer data.
The original fallback is kept for non-ARM64/purego and vector tails.

Validation order, shape/storage errors, scalar/empty aliases, destination ownership
and concurrency behavior remain intact. Cancellation still checks before traversal,
before each at-most-1,024-element interval including the first, and afterward,
even when empty. The two MNIST calls make **9 + 6 = 15** checks. Errors return
directly with hasNaN=false, retaining the same prefix-write/final-check behavior.
ONNX's NaN clone and legacy-destination restoration on NaNs/cancellation are
unchanged. No ONNX production Go source is modified by this integration.

## 6. Optimization decision

Use the reviewed, versioned **v0.10.8** backend. Its small ARM64 integer vector
prefix offers a much larger complete-contract gain than portable masking, bounds
changes or check removal. There is no new API, altered cancellation frequency,
model-specific dispatch, cache, unsafe trick or other operator change.
AMD64 retains the portable fallback pending its own investigation. This ONNX
change updates only the dependency and documents/validates its transfer.

## 7. NeuralNetwork result and validation

The earlier direct pair measured **7.069 → 1.237 µs**, **0.7514 → 0.1315
ns/element**, with **0 B/op / 0 allocs/op**. Seven repetitions span
**7.052–7.767 / 1.231–1.352 µs**, CV **3.63% / 3.49%**. This predicts about
**5.832 µs** transfer. Released source matches those production fingerprints.

Fresh prepared preserving backend component sums on the actual ramp operands
are **7.121 → 1.314 µs**, saving **5.807 µs**, also 0 B/op / 0 allocations.
The corresponding complete ONNX adapters measure **7.127 → 1.3228 µs**.
These fresh sums support the prediction without substituting another estimator.

ONNX default, CGO-disabled, purego, race and purego+race suites pass, as do the
released backend's full default and purego package suites. Existing tests cover
NaN payload/sign/signaling bits, signed zero, infinities, subnormals, invalid
storage, all four alignments, guarded tails, vector/poll boundaries, aliases,
deadlines, exact cancellation checks, concurrency and ownership. No new Go tests
are needed for the unchanged ONNX adapter.

Vet, formatting, module tidy/verify, and all four CI fuzz smoke targets pass.
CGO-disabled default and purego builds pass for **darwin/arm64, darwin/amd64,
linux/arm64, linux/amd64 and windows/amd64**. Cross-builds are compilation checks;
they are not execution on those hosts. The earlier backend investigation also
validated its race/purego-race suites and linux/386 builds. Assembly memory access
is covered by bitwise/guard proofs and tests, alongside the Go race tests.

## 8. Released ONNX complete-runtime result

The comparison policy remains nineteen sorted cases: upstream, zero, one and
sixteen seed-0 uniform-random inputs; ten warmups and 100 timed calls per case;
owned outputs; Go then ORT; atol=1e-4 and rtol=1e-5. Input construction, conversion,
upstream/top-1 checks and timing boundaries are unchanged. Builds, loading and
serialization stay outside inference timing. Both versions use released modules.

Seven alternating pairs, with every valid sample retained:

| Set | Go median µs | Go range / CV | ORT median µs | ORT range / CV | Ratio |
| --- | ---: | --- | ---: | --- | ---: |
| v0.10.7 control | 61.5415 | 61.333–62.2705 / 0.55% | 44.208 | 42.083–44.583 / 2.78% | 1.3921× |
| v0.10.8 candidate | 55.1455 | 54.333–55.5625 / 0.78% | 42.417 | 41.959–44.500 / 1.98% | 1.3001× |

Primary saving, difference of Go medians: **6.396 µs / 10.3930%**. Paired savings
have median **6.750 µs**, range **5.771–7.584**, CV **9.34%**; that does not replace
the primary estimator. Median within-report p95 is **114.2101 → 103.7583 µs**.
One set does not establish a durable tail-latency guarantee.

The separate ordinary ramp measures **61.806 → 55.696 µs**, ranges
**61.416–65.227 / 55.173–56.817**, CV **2.17% / 0.97%**. Both versions allocate
**93,322 B/op / 88 allocs/op**. Ramp timing is never divided by fixed-policy ORT.

All fourteen reports pass all nineteen numerical/top-1/conversion/upstream case
checks: **266 case comparisons** and 26,600 timed calls per engine. Returned Go
output bits agree across both versions and all repetitions. Maximum Go/ORT
absolute error remains **1.9073486328125e-5** with unchanged tolerances.

The [committed evidence](tools/compare_runtime/pass14/README.md) contains lossless
individual timings/output bits, benchmark logs, node-profile samples, validation
commands and source fingerprints. Large binaries/profiles remain outside git.

## 9. Prediction audit

Earlier backend prediction: **5.832 µs**. Fresh released backend indication:
**5.807 µs**. Fresh ONNX adapter recovery: **5.8042 µs**. Actual complete fixed-policy
recovery is **6.396 µs**, approximately **0.589 µs more** than the fresh backend
indication. Ordinary-ramp complete recovery is **6.110 µs**.

The fixed-policy inputs differ from the ramp operands, and the original scalar
kernel is sensitive to sign patterns. Cache, allocator and layout conditions
also differ between prepared and whole-Run scopes. Paired whole-Run savings
span **5.771–7.584 µs**. The extra fraction of a microsecond is not isolated or
attributed to another optimization; no overlapping diagnostic saving is added.
The released complete-runtime recovery reproduces the earlier local experiment's
6.3955-µs indication, while the ORT reference movement changes the ratio.

## 10. Fresh excess-latency ledger

Go values are sums of five-repetition prepared ramp medians. ORT values are the
median of three full-graph node-mean sums, each 1,000 profiled calls after ten
warmups. ORT's integer-microsecond events and unprofiled Go calls have different
scopes. Raw differences are indications, not additive recoverable latency.

| Operation | Go control µs | Go candidate µs | Coarse ORT µs | Candidate difference µs | Finding |
| --- | ---: | ---: | ---: | ---: | --- |
| Conv | 34.537 | 34.798 | 36.906 | -2.108 | Closed; no positive excess |
| ReLU | 7.127 | 1.3228 | 2.026 | -0.7032 | This signal is closed |
| Pool | 3.3264 | 3.404 | 3.063 | +0.341 | Low priority |
| Add | 3.4102 | 3.4949 | 3.024 | +0.4709 | Allocation/planning lead |
| MatMul | 1.669 | 1.709 | 1.053 | +0.656 | Small numerical lead |
| Reshape, general execution, destinations, publication | Not isolated | Not isolated | Different scope | Unknown | No invented residual assignment |

Coarse profiles vary: Conv sums **35.952–47.262 µs**, ReLU **2.019–2.338**,
Add **2.894–3.983**, Pool **3.053–4.036**, MatMul **1.008–1.241**. All three profiles
remain in the evidence; no favorable profile is selected to redefine the ledger.

The complete gap changes **17.3335 → 12.7285 µs**. Go recovers **6.396 µs**;
the ORT reference gets **1.791 µs faster**, offsetting part of the gap reduction.
Prior constant-Reshape and metadata wins are already included in both arms.

The fresh CPU profile still contains substantial Darwin runtime.kevent sampling
(37.82%), preventing precise small-cost attribution. Application samples include
Conv, broadcast Add, Pool, validation and allocation/GC. The allocation profile
shows broadcastDimensions/Strides and contiguousStrides. Prepared Add calls
still allocate **784 B / 22 allocations** across the three nodes, despite
preallocated destinations. Profile setup allocations are not substituted for
ordinary B/op measurements.

## 11. Updated 1.25× roadmap

At the candidate's measured **42.417-µs** ORT reference,
**1.25× = 53.02125 µs**, leaving **2.12425 µs** below current Go **55.1455 µs**.
The control's own reference required **6.2815 µs**. The target difference includes
reference movement and must not be reported entirely as Go recovery.

Holding ORT at 44.208 would instead put the target at 55.260 µs, slightly above
current Go. That is reference normalization, not another measured result or a
stable milestone. The earlier local 0.0945-µs remaining estimate is superseded by
the fresh released-version comparison.

There is no demonstrated next single mechanism sufficient to recover 2.124 µs.
Investigate the remaining positive signals and repeated allocation separately;
do not sum coarse Add/MatMul/Pool differences as deployable savings. There is no
reason here to reopen Conv or add more ReLU complexity.

## 12. Recommendation for Pass 15

Investigate **complete-contract Add broadcast planning and allocation**. Its
fresh latency indication is smaller than MatMul's, but its three prepared calls
still allocate **216 + 216 + 352 = 784 B** and **7 + 7 + 8 = 22 objects** per
three-call traversal. The independent allocator-pressure evidence makes it the
stronger combined CPU/allocation investigation. Determine why shapes and strides
are rebuilt, preserve dynamic/override/destination/cancellation behavior, and
require a whole-Run transfer before accepting a change.

MatMul remains an alternative small numerical opportunity. No particular
saving is promised for either. ReLU's required work is now small, while repeated
allocation remains an independent objective even if a later stable reference
places the runtime below 1.25×.
