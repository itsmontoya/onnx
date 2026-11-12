# Performance Optimization Pass 15 — Broadcast Metadata Ownership

2026-11-12, Apple M3. The selected backend change removes **448 B / 15
allocations per inference**: **93,322 B / 88 objects → 92,874 B / 73 objects**.
All 22 original Add allocations are accounted for. Seven remaining allocations
are matrix adapters and copied values, not immutable broadcast metadata.

The final fourteen-pair fixed-policy comparison measures **55.22925 → 54.42675
µs**, a **0.8025-µs / 1.4530%** saving. This exceeds the component indication;
the ordinary ramp has no latency improvement, and the initial seven-pair
prototype comparison was latency-neutral. The allocation result is reproducible;
the fixed-policy latency result is not a durable speedup guarantee.

Production code is in neuralnetwork commit
`647e0c8ef3c93d90707df5077046346cd658f68e`, on local branch
`optimization-pass-15`. ONNX production source and its released v0.10.8 dependency
remain unchanged. Integration was tested through separate temporary module files,
first against an isolated checkout and then against the identical sibling source.
This is a validated, unreleased backend candidate; no release or remote publication
was performed. The scripts, patch, diagnostic overlays and lossless evidence are
in [the Pass 15 evidence directory](tools/compare_runtime/pass15/README.md).

## 1. Baseline

Fresh environment: macOS **26.5.2 (25F84)**, Darwin **25.5.0**, Apple **M3**,
Mac15,13, 8 cores (4 performance/4 efficiency), **24 GB**, darwin/arm64,
**Go 1.27.2**. Python **3.9.6**, NumPy **1.26.4**, ONNX **1.16.2**, protobuf
**4.25.3**, ORT **1.19.2**. Host information was collected again for this pass.

ONNX baseline: `752a8ac093718c1e78218e029997f705dbc802db`.
neuralnetwork baseline: v0.10.8, `be95e90b7f99e5ab59213f360a8b79980def6efa`.
Other module versions: protobuf v1.36.12, simd v1.3.0, x/sys v0.45.0.
Performance environment: **GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off,
GOARM64=v8.0**, default optimization, no build tags or gcflags. Escape analysis,
allocation profiling and race tests are separate from latency measurements.

The unchanged fixed policy uses nineteen sorted cases: upstream, zeros, ones,
sixteen seed-0 uniform-random inputs; **10 warmups / 100 timed calls per case**;
owned outputs; Go then ORT; **atol=1e-4 / rtol=1e-5**. Input generation,
conversion, upstream/top-1 checks and serialization boundaries are unchanged.
ORT uses CPUExecutionProvider, sequential execution, one intra/inter-op thread,
disabled graph optimization and its default arena/memory pattern.
Model SHA-256: `310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.

Fresh final control: **55.22925 µs**, **93,322 B/op / 88 allocs/op**.
Its separate ordinary-ramp median is **55.9155 µs**. Neither ramp timing nor
profiled node timing is divided by fixed-policy ORT timing.

## 2. Add execution-path analysis

All three calls use fixed, validated float32 signatures:

| Node | Left | Right | Output | Traversal |
| --- | --- | --- | --- | --- |
| 01 | [1,8,28,28] | [8,1,1] | [1,8,28,28] | 8 contiguous spans of 784 |
| 05 | [1,16,14,14] | [16,1,1] | [1,16,14,14] | 16 contiguous spans of 196 |
| 11 | [1,10] | [1,10] | [1,10] | Equal-shape matrix adapter |

`evaluation.execute` checks context and resolves value indices into a two-pointer
argument slice. That slice is a separate **16 B / 1 object per Add** in complete
inference, excluded from the prepared Add microbenchmarks. `preparedCall.run`
copies the immutable numerical descriptor to its invocation stack, assigns those
arguments, resolves/charges a private destination and retains workspace-release
semantics. `numericalCall.run` checks context, allocates a destination if needed,
dispatches to `tensor.AddInto`, then checks context again. Backend Add has no
context argument or additional cancellation polling; this remains unchanged.

Baseline `binaryFloat32Into` validates both tensor types/storage/shapes, calls
`BroadcastShape`, validates destination type/storage/exact shape and both storage
identities, then returns early for empty output. Broadcasting validates input
dimensions/products, aligns trailing axes without allocating normalized operands,
calculates output dimensions, and validates the output product. The public shape
result is owned even though Add only needs it temporarily.

Nodes 01/05 discover a scalar suffix, try the reverse operand order if needed,
then plan both strides before any write. Each `broadcastStrides` calls
`ValidateBroadcastTo`, which constructs another output shape, constructs input
contiguous strides, then constructs output-rank strides with zeros on broadcast
axes. Local coordinates carry only over the prefix; source offsets are local
integers. The unchanged loop loads one scalar per span and adds it to contiguous
values in the original operand order.

Node 11 instead calls `matrixBinaryInto`: copy each input into a Matrix, allocate
a Matrix output, dispatch Matrix.AddInto, obtain a copied Values slice and copy
it into the tensor destination. Ten elements select the portable matrix loop on
this ARM64 build. These copies are independent of broadcast planning.

Pass 12 already stores output shape, byte charge, workspace and operator metadata
in `numericalCall`. Its report explicitly left backend strides behind the general
AddInto API. ONNX has sufficient fixed information here; the limitation was the
backend ownership/API boundary, not missing model information. Dynamic/symbolic
inputs decline preparation. Binding and produced-value validation enforce fixed
dimensions and original declarations before prepared consumers run. Initializer
values may change without changing those validated signatures.

Prepared calls do not request input Shape snapshots every Run. Output validation
still takes copied shape metadata; the compiler can keep nonescaping small
copies local. Context interfaces are already supplied, function expressions are
static, and the workspace defer does not create a heap closure. Neither returned
shape storage nor matrix storage is a closure/interface-boxing allocation on the
successful path. Error formatting boxes arguments only on error paths.

## 3. Allocation ledger

Exact baseline per-call accounting, including allocator size classes:

| Site/object | Node 01 B / count | Node 05 B / count | Node 11 B / count |
| --- | ---: | ---: | ---: |
| Binary output shape, []int64 | 32 / 1 | 32 / 1 | 16 / 1 |
| ValidateBroadcastTo result shapes, []int64 | 64 / 2 | 64 / 2 | 0 / 0 |
| Input contiguous strides, []int64 | 56 / 2 | 56 / 2 | 0 / 0 |
| Aligned broadcast strides, []int64 | 64 / 2 | 64 / 2 | 0 / 0 |
| Matrix objects, 48 bytes each | 0 / 0 | 0 / 0 | 144 / 3 |
| Copied input float32 buffers, 40 requested → 48 each | 0 / 0 | 0 / 0 | 96 / 2 |
| Intermediate output / Values copy, same rounding | 0 / 0 | 0 / 0 | 96 / 2 |
| **Total** | **216 / 7** | **216 / 7** | **352 / 8** |

Total: **784 B / 22 allocations**. With 10,000 complete calls of each shape,
the filtered profile accounts for 70,000 broadcast-dimension allocations,
40,000 contiguous-stride allocations, 40,000 broadcast-stride allocations,
40,000 FromSlice allocations, 20,000 Matrix.New allocations and 10,000 Values
allocations: exactly **220,000** objects. Benchmark setup is outside that filter.

The fifteen shape/stride arrays are invocation-lived, shape-dependent metadata;
their successful contents do not depend on values. Returned slices cross helper
boundaries and escape. Ownership of a public BroadcastShape result is required;
ownership of Add's internal temporary shape is not. Coordinates are mutable,
invocation-local metadata and already do not escape for these calls.

The seven matrix allocations are invocation-lived adapters/payloads. Their
contents depend on values, so retaining them as immutable model metadata would
be incorrect. Removing their copies may be possible through a different arithmetic
boundary, but is outside the selected metadata change.

## 4. Static versus dynamic classification

| Work/state | Fixed MNIST calls | General contract |
| --- | --- | --- |
| Operand/output shape, rank alignment, broadcast dimensions | A: model invariant | C when dimensions/signatures are deferred |
| Contiguous/broadcast strides, scalar span and prefix rank | A | C for changing signatures; never value-dependent |
| Resolved output shape and workspace charge | A, already prepared | C on the existing dynamic path |
| Limits/backend selection | Model configuration; potentially B in other architectures | Must match configured execution policy |
| Destination, type/storage validity, exact shape and aliases | C | Validate each invocation |
| Context/deadline, input/default selection, symbolic bindings | C | Retain Run behavior |
| Coordinates/offsets | C: mutable traversal state | Never share in a descriptor |
| Actual arithmetic and matrix-copy contents | D: value dependent | Never prepare as immutable metadata |

There is no session-only broadcast cache to extend. The selected change works
with current shapes every time and therefore needs no static-signature guard,
learning cache, override invalidation or tensor retention.

## 5. Cost decomposition

Initial controlled five-repetition medians, µs per call:

| Diagnostic | Add 1 | Add 2 | Add 3 | B/op, objects across three |
| --- | ---: | ---: | ---: | ---: |
| Current complete backend Add | 1.924 | 1.253 | 0.1876 | 784, 22 |
| Shape validation/result construction | 0.0580 | 0.0622 | 0.0462 | 80, 3 |
| Input validation | 0.0278 | 0.0285 | 0.0229 | 0, 0 |
| Destination validation | 0.0201 | 0.0209 | 0.0163 | 0, 0 |
| Shape/stride planning | 0.2566 | 0.2606 | 0.0470 | 448, 15 |
| Resolved arithmetic only | 1.636 | 0.9212 | 0.00661 | 0, 0 |
| Validated static diagnostic | 1.691 | 0.9693 | 0.04577 | 0, 0 |

The last two Add-3 cells deliberately bypass the matrix adapter: they are ceilings,
not the selected implementation. The plan-owned prototype below preserves that
adapter. Planning includes allocation and validation; no isolated GC/allocator
latency is inferred. These scopes overlap and are not additive savings.

The shape/stride planning sum is about **0.564 µs**. Required validation remains;
eliminating all planning is not a realistic promise. Initial arithmetic ranges
are 1.633–1.817 / 0.9175–0.9217 µs for the channel calls, while planning ranges
are 0.2542–0.2932 / 0.2588–0.2644 µs. Full ranges/CVs are retained in summary.json.

Channel traversal touches at least **50,208 + 25,152 B**: contiguous input/output
plus one scalar load per span. The equal-shape adapter touches about **440 B**,
of which 120 B is useful addition traffic and 320 B is copying. These are cached
traffic counts, not DRAM bandwidth. At 6,272 / 3,136 / 10 elements, the initial
complete costs are about **0.307 / 0.400 / 18.76 ns per element**; resolved
channel arithmetic is **0.261 / 0.294 ns per element**.

## 6. Escape and generated-code findings

Escape analysis confirms returned broadcast/contiguous arrays escape, while
coordinates do not for these ranks. Operation function values do not escape.
Destination error formatting previously made the expected-shape argument escape
even on success. Copying that argument only inside the error branch permits
caller scratch to remain on the stack without changing the error string.

ARM64 listings show scalar span arithmetic, suffix detection, checked stride
products and bounds/panic branches at slice/index boundaries. There is no new
SIMD or arithmetic dispatch. The matrix function pointer remains an indirect
call. Heap makeslice calls remain behind insufficient-capacity branches and the
compiler's large-coordinate fallback. The binary frame grows **96 → 176 B**;
the specialized broadcast frame grows **176 → 304 B**. Explicit shape/stride
arrays add at most 256 B across the nested common-rank scopes, plus ordinary
call-frame overhead. No unsafe trick or interface ABI change is used.

The ownership change removes intermediate contiguous arrays and their redundant
product traversal; strides are filled backward with checked suffix multiplication.
Empty suffix overflow behavior is preserved, including zeros before overflowing
suffixes. Shape compatibility and destination validation retain their order.

## 7. Controlled diagnostics and dynamic fallback

Fresh direct backend measurements, forward order, five repetitions:

| Contract/shape | Control µs · B · objects | Candidate µs · B · objects |
| --- | ---: | ---: |
| First channel Add | 1.925 · 216 · 7 | 1.912 · 0 · 0 |
| Second channel Add | 1.223 · 216 · 7 | 1.165 · 0 · 0 |
| Equal shape | 0.1850 · 352 · 8 | 0.1789 · 336 · 7 |
| Scalar broadcast | 0.3070 · 144 · 6 | 0.2530 · 0 · 0 |
| Multidirectional rank five | 0.6797 · 336 · 7 | 0.5961 · 0 · 0 |
| Generic per-element fallback | 0.3561 · 168 · 7 | 0.2775 · 0 · 0 |
| Both operands vary across suffix | 0.8434 · 104 · 7 | 0.7876 · 0 · 0 |
| Empty tensor | 0.0782 · 24 · 1 | 0.0693 · 0 · 0 |
| Rank eight | 0.4922 · 448 · 7 | 0.4032 · 0 · 0 |
| Rank nine, heap fallback | 0.5289 · 560 · 7 | 0.4750 · 400 · 5 |

Reverse-order samples and all ranges/CVs remain in the evidence. These operands
are synthetic finite values; they are not substituted for the real-intermediate
ONNX results. Small differences are not overinterpreted.

The changing-shape ONNX Add fixture measures **0.9331 → 0.9233 µs** with
preparation enabled but ineligible, **1,280 → 1,272 B / 26 → 25 objects**.
Stable dynamic shapes measure **0.9213 → 0.9069 µs**, **1,264 → 1,256 B / 26 →
25 objects**. There is no demonstrated material dynamic-path regression.

The separate immutable plan-owned prototype extends the existing numerical
descriptor, with exact signature checks and conservative optional load charging.
It stores only copied shapes, strides, span and rank; coordinates stay local.
Dynamic/unsupported preparation falls back. It preserves the matrix adapter:

| Prototype, complete ONNX adapters | Ownership µs | Static descriptor µs | B / objects |
| --- | ---: | ---: | ---: |
| Add 1 | 1.935 | 1.752 | 0 / 0 |
| Add 2 | 1.186 | 0.9981 | 0 / 0 |
| Add 3 | 0.1967 | 0.1601 | 336 / 7 |
| **Sum** | **3.3177** | **2.9102** | **336 / 7** |

Its seven-pair complete fixed-policy experiment measures **54.792 → 54.334 µs**
(0.458 µs), ownership range 54.229–55.062 / CV 0.50%, descriptor range
53.292–54.9165 / CV 0.94%. Both allocate **92,874 B / 73 objects**. This is
evidence of a small additional latency option, not an allocation benefit.

## 8. Optimization decision and correctness

Select internal buffer-accepting shape/stride helpers and common-rank stack
scratch, retaining arbitrary-rank heap fallback. Public BroadcastShape still
returns independent, non-nil scalar metadata. The general backend owns this
work; ONNX does not duplicate numerical code or create another preparation cache.
Shared Sub/Mul/Div and stride-helper callers benefit from the same metadata
implementation; their arithmetic remains unchanged.

Alternatives considered: a public immutable Add descriptor, caller-supplied
strides, another geometry fast path, and the ownership change. The descriptor
has a measured ~0.4-µs extra indication, but adds API/signature/lifetime/memory
contracts with no further allocation reduction. Exposing individual strides
widens the invariant surface. Another fast path changes arithmetic dispatch.
The ownership change has no new public API, persistent state or preparation
dependency. It is the smaller allocation-focused production choice; the
descriptor remains a reviewable diagnostic option rather than being called free
or ineffective.

Default, CGO-disabled, purego, race and purego+race suites pass for ONNX and the
backend; the actual applied source also passes both complete CGO-disabled suites.
Vet, module tidy/verify, formatting, diff checks and all four existing bounded
fuzz smoke targets pass. CGO-disabled default/purego cross-builds pass for
darwin/arm64, darwin/amd64, linux/arm64, linux/amd64, windows/amd64 and linux/386.
Cross-builds are compilation checks, not execution on those hosts.

Existing tests explicitly exercise dynamic/symbolic changes, defaults/overrides,
scalar/rank/equal-rank broadcasts, incompatible/empty/zero-dimensional results,
preallocated destinations and shallow/identity aliases, cancellation/deadlines,
workspace/live/load limits, concurrent sessions, model closure and output
ownership. New backend tests cover caller capacity/ownership, reused stride
clearing, rank 7/8/9/32/33 boundaries in both orders, scalar slice identity and
empty suffix overflow. No arithmetic, tolerance, cancellation boundary, quota
charge or output ownership rule changes.

## 9. Initialization and memory economics

Selected change: **zero additional plan construction, retained bytes, retained
objects or numerical payload**. No work moves to initialization; it remains
bounded invocation-local planning. There is no preparation amortization threshold.
Stack zeroing/growth is not free; the complete measurements include steady-state
stack work after the prescribed warmups.

Measured loading: **379.621 → 378.159 µs** medians, both **345,318 B / 2,563
allocations**. Session creation: **26.17 ns / 24 B / 1 object** in both arms.
Those small time differences do not establish a loading optimization.

For comparison, the rejected descriptor constructor costs **0.7308 µs**, range
0.7268–0.7325, and retains **784 B / 16 objects** across the three calls:
three 144-byte descriptors plus 352 bytes of shape/stride arrays. Numerical
payload is zero. Its raw constructor amortization at a 0.4075-µs component gain
is about **two Runs**, excluding extra integration/load work and retained-memory
cost. Its diagnostic quota bound is deliberately conservative rather than a
discount for eliminating Run allocation. None of this state is shipped.

## 10. Component result

Five-repetition prepared ONNX adapter medians on actual ramp intermediates:

| Call | Control µs · B · objects | Candidate µs · B · objects | Control/candidate range µs |
| --- | ---: | ---: | --- |
| 01 Add | 2.074 · 216 · 7 | 1.975 · 0 · 0 | 2.068–2.079 / 1.965–1.981 |
| 05 Add | 1.324 · 216 · 7 | 1.212 · 0 · 0 | 1.313–1.336 / 1.208–1.218 |
| 11 Add | 0.2064 · 352 · 8 | 0.1963 · 336 · 7 | 0.2053–0.2098 / 0.1955–0.1965 |
| **Sum** | **3.6044 · 784 · 22** | **3.3833 · 336 · 7** | Sum of component medians |

Indicated saving: **0.2211 µs / 6.13%**. Individual CVs are below 0.9%; the
different direct-backend and initial diagnostic scopes remain separately labeled.

## 11. Complete ONNX runtime result

The final source received fourteen alternating pairs: seven initially, followed
by one predefined seven-pair confirmation because the whole-Run movement exceeded
the component indication and differed from the initial prototype. Every valid
sample remains. The primary estimator remains the difference of medians of
per-report engine medians; paired differences do not replace it.

| Final set | Go median µs | Go range / CV | ORT median µs | ORT range / CV | Ratio |
| --- | ---: | --- | ---: | --- | ---: |
| Released control | 55.22925 | 53.792–57.3545 / 1.68% | 44.396 | 44.000–45.208 / 0.82% | 1.2440× |
| Local backend candidate | 54.42675 | 51.2915–56.000 / 2.21% | 44.583 | 41.625–46.208 / 2.30% | 1.2208× |

Go saving: **0.8025 µs / 1.4530%**. Paired saving median **0.72875 µs**, range
**-0.875–3.917 µs**. CV of differences near zero is not a useful reliability
summary. ORT becomes **0.187 µs slower** between arms; that is separate reference
movement, not a Go improvement.

The separate ramp is **55.9155 → 55.9555 µs**, ranges 55.547–58.935 /
54.323–57.858, CV 2.24% / 1.40%. It shows no latency win. Allocation results
are invariant across all final ramp samples: **93,322 → 92,874 B/op** and
**88 → 73 allocs/op** (**0.480% fewer bytes / 17.05% fewer allocations**).

Initial prototype results are retained separately: **55.0205 → 55.208 µs** Go,
CV 1.29% / 0.87%, the same allocation improvement. Before freezing final source,
scalar public helpers were corrected to preserve non-nil empty slices, the stack
bound was named, and the error-copy rationale was documented. Initial and final
source sets are not silently pooled. Both argue against promising a fixed latency
saving from the allocation count alone.

Across the initial, final and separate descriptor experiments, all **56 reports /
1,064 case comparisons / 106,400 timed calls per engine** pass unchanged
numerical/conversion/upstream/top-1 checks. Go output bits agree across all
variants and repetitions. Maximum Go/ORT absolute error remains
**1.9073486328125e-5**. No performance sample is selected for favorable accuracy
or timing.

## 12. Prediction audit

Predicted component transfer: approximately **0.22 µs / 448 B / 15 objects**.
Actual final fixed-policy transfer: **0.8025 µs / 448 B / 15 objects**.
The extra **0.5814 µs** is not isolated or attributed to broadcast planning.
Initial source showed no whole-Run latency gain, and the final ordinary ramp
still shows none. Workload differences, host variance, compiler layout and
allocator/cache context can contribute; no one mechanism was demonstrated for
that excess movement. Production acceptance rests on exact allocation transfer
with preserved behavior, not on treating the extra fraction as recoverable work.

The independent descriptor probe predicts about **0.4075 µs** beyond ownership
and measures **0.458 µs** in its complete fixed-policy experiment. These are
alternative implementations, not additive Pass 15 production savings.

## 13. Updated allocation profile

Filtered Session.Run profiles and a separate constructor census exclude model
load/profile setup. The new full accounting is:

| Per-Run source | B/op | Objects |
| --- | ---: | ---: |
| Ten owned backend tensors: payloads, shapes, structs, identities | 91,610 | 40 |
| Twelve execution argument slices | 160 | 12 |
| Evaluation/symbol map, value slots, budget | 448 | 4 |
| Equal-shape Add matrix adapters/copies | 336 | 7 |
| Two Conv output-shape arrays | 64 | 2 |
| Two Pool output-shape arrays | 64 | 2 |
| MatMul descriptor and output shape | 96 | 2 |
| MatMul temporary output payload | 48 | 1 |
| Dynamic Reshape result shape | 16 | 1 |
| Published output wrapper/list | 32 | 2 |
| **Total** | **92,874** | **73** |

The ten tensor constructions are two [1,8,28,28], one [1,8,14,14], two
[1,16,14,14], one [1,16,4,4], one [1,256] and three [1,10], including publication.
Their allocator-rounded payloads total **90,384 B**, shapes **256 B**, rounded
tensor structs **960 B**, and tiny identities **10 B**. Heap profiling coalesces
tiny objects, so profile object samples are not substituted for the ordinary
allocation counter. The source/census ledger accounts for that difference.

Add's new component ledger is **336 B / 7 objects**, all on node 11. All other
complete-runtime allocations remain **92,538 B / 66 objects**, unchanged. Large
owned numerical destinations dominate bytes; argument slices and tensor metadata
dominate many small-object counts. Their existence is not evidence that pooling,
cross-Run retention or an arena would preserve ownership.

## 14. Updated excess-latency ledger

Fresh Go values are sums of five-repetition prepared ramp medians. ORT values
are medians of three full-graph profiled node-mean sums, each 1,000 calls after
ten warmups. Integer-microsecond ORT events and unprofiled Go calls differ in
scope. No ledger is forced to sum to the complete-runtime gap.

| Operation | Go control µs | Go candidate µs | Coarse ORT µs | Candidate minus ORT |
| --- | ---: | ---: | ---: | ---: |
| Conv | 34.976 | 35.076 | 36.647 | -1.571 |
| ReLU | 1.3374 | 1.3376 | 2.012 | -0.6744 |
| Pool | 3.5700 | 3.5166 | 3.028 | +0.4886 |
| Add | 3.6044 | 3.3833 | 2.988 | +0.3953 |
| MatMul | 1.725 | 1.686 | 1.001 | +0.685 |
| Evaluation creation/binding only | — | 0.9405, 448 B / 4 objects | Different contract | Unknown |
| Remaining dispatch, retirement, destinations, publication | Not isolated | Not isolated | Different contract | Unknown |

ORT node sums span Conv 36.056–42.951, ReLU 2.003–2.199, Pool 3.009–3.573,
Add 2.839–3.634 and MatMul 1.000–1.135 µs. All three profiles remain.
The CPU profile has **85.45% runtime.kevent** samples, limiting precise small-cost
attribution. Application samples still include the existing convolution, Add,
pooling, ReLU and allocation paths. Conv and ReLU remain closed.

## 15. Updated 1.25× roadmap

At the final candidate's fresh **44.583-µs ORT** reference, **1.25× = 55.72875
µs**. The local candidate's **54.42675 µs** is **1.302 µs below** that target,
giving **1.2208×** in this comparison. The control is already **1.2440×** at its
own reference; this is not a newly achieved released-runtime milestone.

Relative to Pass 14, ORT moved **42.417 → 44.583 µs** (2.166 µs slower), moving
the target by **2.7075 µs**. The direct Go control/candidate improvement is only
**0.8025 µs**. Holding the old reference fixed would put this local candidate at
about **1.2831×**, not below 1.25×. That is normalization, not a measured run.
Initial source was 1.2524× on its own fresh reference. Ratios remain sensitive to
reference movement; release integration and later confirmation remain separate.

## 16. Updated 1.10× roadmap

At **44.583 µs**, **1.10× = 49.0413 µs**; the remaining difference is **5.38545
µs**. The control's own reference leaves **6.39365 µs**. Of that 1.0082-µs
difference, **0.8025 µs** is the measured Go movement and **0.2057 µs** is the
target's movement with ORT. No demonstrated mechanism here recovers the remaining
5.38545 µs, and coarse operator differences are not summed into a promise.

## 17. Recommendation for Pass 16

Investigate **complete-contract matrix adapter copying/ownership at the tensor
arithmetic boundary**, beginning with the seven independently demonstrated
equal-shape Add allocations and inspecting how the same boundary relates to
MatMul staging. Node 11 still copies values through four buffers for ten additions;
MatMul retains a small positive latency indication plus three planning/staging
allocations. This is a stronger concrete mechanism than assuming raw kernel time
is removable. Measure the small absolute ceiling before changing dispatch or APIs.

The static Add descriptor is a measured alternative if maintainers prefer its
additional latency option; it has no further allocation advantage and remains
unshipped. Large tensor destinations and execution argument slices deserve
ownership analysis, not automatic pooling. Keep Conv/ReLU closed, require fresh
whole-Run transfer, and select any production work from that evidence.
