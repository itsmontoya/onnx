# Pass 15 — Published neuralnetwork v0.10.9 Integration

2026-11-12, Apple M3. ONNX now uses published **neuralnetwork v0.10.9**.
The three Add calls allocate **784 → 336 B / 22 → 7 objects**; complete inference
allocates **93,322 → 92,874 B/op / 88 → 73 allocs/op**. Seven fresh alternating
fixed-policy pairs measure **55.417 → 54.750 µs Go**, a **0.667-µs / 1.2036%**
saving. Candidate ORT is **42.375 µs**, giving **1.2920× ORT**.

This validates the published dependency without a local replacement. The earlier
[Pass 15 investigation](PERFORMANCE_PASS15.md), fourteen-pair local comparison
and descriptor experiment remain separately archived; their samples are not
pooled into this released-module result.

## Release provenance and scope

ONNX baseline: `a73fd4bfd3f3b0494194d1e6a45bf88173f74ec9`.
Dependency integration: `3b37416360c1ef9ef98c8bf38f011897d334bb58`.
The measured working tree contained these exact module changes before committing;
reports therefore record the baseline HEAD with a dirty working tree.

| Dependency | Commit | Module checksum |
| --- | --- | --- |
| v0.10.8 control | `be95e90b7f99e5ab59213f360a8b79980def6efa` | `h1:J/ont78/Q9d0OzzemUem6yNw0cI1gIt7Pi3sCF5Qjas=` |
| v0.10.9 candidate | `740a244bf56214bb3443a69287a2d800674620e4` | `h1:kmfxkbiZZ8fbw0cxCxvar2nfYJ2Bx6kci/QUNH0w6oQ=` |

The v0.10.9 tree matches the tested backend commit `647e0c8`; all four changed
production files match the original recorded SHA-256 fingerprints. Module
verification passes. The only runtime integration changes are go.mod/go.sum:
**v0.10.8 → v0.10.9**, with no indirect dependency or ONNX Go-source changes.

The control uses a separate v0.10.8 module snapshot. The candidate uses normal
go.mod. Both use published modules and neither has a replace directive. The
original local-experiment replay script now pins its historical control snapshot
so this dependency upgrade cannot silently change its baseline.

## Fresh environment and policy

Apple M3, Mac15,13, 8 cores (4 performance/4 efficiency), 24 GB; macOS
**26.5.2 (25F84)**, Darwin **25.5.0**, darwin/arm64, **Go 1.27.2**. Performance
processes use **GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off, GOARM64=v8.0**,
default optimization and no tags/gcflags. Host information was collected again.
Python **3.9.6**, NumPy **1.26.4**, ONNX **1.16.2**, protobuf **4.25.3**, ORT
**1.19.2**. Other Go module versions remain protobuf v1.36.12, simd v1.3.0 and x/sys v0.45.0.

The unchanged policy uses upstream, zeros, ones and sixteen seed-0 uniform-random
inputs: nineteen sorted cases, ten warmups and 100 timed calls per case, owned
outputs, Go then ORT, atol=1e-4 and rtol=1e-5. Input construction, conversion,
upstream/top-1 checks and timing boundaries are unchanged. Builds, loading and
serialization remain outside inference timing. ORT uses CPUExecutionProvider,
sequential execution, one intra/inter-op thread, disabled graph optimization and
default arena/memory pattern.

The model SHA-256 remains
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
Driver, runner and all input fingerprints match the original investigation.

## Complete-runtime transfer

Every valid sample is retained. The primary estimator remains the difference
between medians of per-report engine medians; paired differences are secondary.

| Released set | Go median µs | Go range / CV | ORT median µs | ORT range / CV | Ratio |
| --- | ---: | --- | ---: | --- | ---: |
| v0.10.8 | 55.417 | 54.479–55.459 / 0.74% | 42.500 | 41.333–44.542 / 3.08% | 1.3039× |
| v0.10.9 | 54.750 | 53.4375–55.292 / 1.29% | 42.375 | 41.667–44.709 / 2.92% | 1.2920× |

Go improves **0.667 µs / 1.2036%**. Paired saving median is **0.3755 µs**,
range **-0.271–1.9795 µs**. ORT becomes **0.125 µs faster** between arms;
that is separate reference movement and offsets part of the gap reduction.

The separate ordinary ramp measures **56.363 → 55.486 µs**, ranges
55.387–60.552 / 54.903–59.228, CV 3.67% / 2.66%. That is a small positive
indication with substantial spread, not a durable latency guarantee. It is never
divided by fixed-policy ORT timing.

All ramp repetitions reproduce **93,322 → 92,874 B/op / 88 → 73 allocs/op**:
**448 B / 15 allocations removed per Run**. The allocation transfer is the
stronger repeatable result. Every one of fourteen reports passes all nineteen
numerical, conversion, upstream and top-1 cases: **266 case comparisons / 26,600
timed calls per engine**. Go output bits agree across versions/repetitions.
Maximum absolute Go/ORT error remains **1.9073486328125e-5**, with unchanged tolerances.

## Component result and prediction audit

Fresh prepared adapter sums, five repetitions on actual ramp intermediates:

| Scope | Control µs · B · objects | Released candidate µs · B · objects |
| --- | ---: | ---: |
| Three Add calls | 3.5533 · 784 · 22 | 3.3511 · 336 · 7 |
| Complete ordinary-ramp inference | 56.363 · 93,322 · 88 | 55.486 · 92,874 · 73 |

The fresh component indication is **0.2022 µs**, close to the original local
0.2211-µs indication. Complete fixed-policy saving is **0.667 µs**, approximately
**0.4648 µs more** than the component indication. That additional movement is
not isolated or attributed to planning. Prior local source sets ranged from
latency-neutral to a larger saving, and the ordinary-ramp result also varies.
Cache, allocator, binary layout and host conditions differ between scopes; no
single cause was demonstrated for the extra fraction. Do not add diagnostic
ceilings to the released saving.

Allocation savings transfer exactly. Node 11 still has seven matrix-adapter/value
allocations, while the two channel Adds now have zero. The full **92,874 B / 73
object** ledger in the original report remains applicable: the other **92,538 B /
66 objects** are unchanged. New large per-Run sources remain owned tensor payloads
and metadata, execution argument slices and the existing matrix adapters.

## Initialization and correctness

No new prepared state or numerical payload is retained. Common-rank planning uses
bounded stack scratch with arbitrary-rank heap fallback; initialization does not
receive moved Run work. LoadBytes measures **382.106 → 378.295 µs**, approximately
345.3 KB / 2,563 allocations in both arms. Session creation measures
**26.15 → 26.12 ns / 24 B / 1 object**. Small loading differences are not claimed
as a separate optimization.

Fresh ONNX and published-backend default, CGO-disabled, purego, race and purego+race
suites pass. Vet, formatting, tidy/verify and all four bounded fuzz smoke targets
pass. CGO-disabled default/purego cross-builds pass for darwin/arm64, darwin/amd64,
linux/arm64, linux/amd64, windows/amd64 and linux/386. Cross-builds establish
compilation, not execution on those hosts.

Existing tests retain dynamic/symbolic shape changes, initializer overrides,
broadcast/empty/scalar semantics, aliases/preallocated destinations, cancellation,
deadlines, memory quotas, concurrent sessions, lifecycle and output ownership.
The release contains the tested ownership/rank-boundary/overflow regressions.
No new ONNX test or numerical workaround is needed for unchanged adapter code.

## Fresh excess-latency ledger

Go values sum five-repetition prepared medians. ORT values are medians of three
full-graph node-mean sums, each 1,000 calls after ten warmups. These scopes differ;
integer-microsecond ORT events give indications, not exact recoverable excess.

| Operation | Go candidate µs | Coarse ORT µs | Difference µs |
| --- | ---: | ---: | ---: |
| Conv | 35.004 | 36.067 | -1.063 |
| ReLU | 1.3336 | 2.008 | -0.6744 |
| Pool | 3.4588 | 3.016 | +0.4428 |
| Add | 3.3511 | 2.453 | +0.8981 |
| MatMul | 1.686 | 1.004 | +0.682 |
| Remaining runtime machinery | Not isolated here | Different scope | Unknown |

ORT samples span Add **2.434–3.094**, MatMul **1.002–1.036**, Pool **3.014–3.131**,
Conv **35.754–38.071** and ReLU **1.857–2.019 µs**. The larger fresh Add difference
does not invalidate the measured planning/copy ceilings or promise 0.8981 µs.
Do not force these rows to sum to the complete-runtime gap. Conv and ReLU remain closed.

## Updated targets and next investigation

At fresh candidate ORT **42.375 µs**, **1.25× = 52.96875 µs**, leaving **1.78125
µs** below current Go **54.750 µs**. **1.10× = 46.6125 µs**, leaving **8.1375 µs**.
Neither remaining difference is a demonstrated next optimization.

The earlier local comparison used ORT **44.583 µs** and gave **1.2208×**. ORT is
now **2.208 µs faster**, while measured Go is **0.32325 µs slower** across those
separate sets. The ratio movement to **1.2920×** must not be attributed entirely
to Go. Holding ORT at 44.583 would put released Go at about **1.2280×**; that is
normalization, not another measurement. The fresh released comparison does not
establish the 1.25× milestone.

The evidence still supports investigating complete-contract matrix-adapter
copying/ownership, beginning with the seven remaining equal-shape Add allocations
and assessing the related MatMul staging boundary. The static descriptor remains
a measured API tradeoff with no additional allocation benefit. Keep the original
evidence-first recommendation; do not reopen Conv/ReLU or optimize toward a target.

All source provenance, individual timings/output bits, profiles, checks and replay
instructions are preserved in [the released-integration evidence](tools/compare_runtime/pass15/release/README.md).
