# Performance Optimization Pass 13

2026-11-12, Apple M3. **ACCEPT**, under the conditional acceptance guideline,
the bounded, copied constant-only Reshape.
**62.375 → 61.750 µs Go → 44.167 µs ORT → 1.3981×** in the final seven-pair
comparison: **0.625 µs / 1.0020%** faster. **103,691 → 93,322 B/op;
93 → 88 allocs/op.** Added model/plan-owned numerical payload: **10,240 B**,
shared by sessions. No numerical kernel or dependency change.

## Current Baseline

Starting ONNX HEAD: `9f1905e200e1998ba522933bb26af5b3498a7136`, the accepted
Pass 12 merge. neuralnetwork remains **v0.10.7**, commit
`1f033b7d13788661a9002462151a997f30b620b4`, module sum
`h1:LPdDwSZz5MkiYGL1UAiMVZmpITvu5xVX4PHKoaf4BEw=`. No module replacement.
Production implementation: `d9ad3df51d622a01360b4cf05927c804b62d5787`.
Contract tests: `e817c8ebe8882f16d26559f0aa7fd5fc87010f89`.
Measurements used identical sources before these commits were created.

Go **1.27.2**, macOS **26.5.2 (25F84)**, **darwin/arm64**, Apple M3,
**GOARM64=v8.0**. Performance processes use **GOMAXPROCS=1, CGO_ENABLED=0,
GOWORK=off**, default optimization and no tags/gcflags; comparison builds use
`-buildvcs=false`. Host CGO_ENABLED defaults to 1; host GOMAXPROCS is unset.
ORT **1.19.2**, CPUExecutionProvider, ORT_SEQUENTIAL, one intra-op and one
inter-op thread, ORT_DISABLE_ALL, default CPU arena and memory pattern enabled.
Python **3.9.6**, NumPy **1.26.4**, ONNX **1.16.2**, protobuf **4.25.3**.

The established policy is unchanged: upstream/zero/one and sixteen seed-0
uniform random inputs, **19 cases**, ten warmups and 100 timed calls per case;
atol=1e-4, rtol=1e-5. Owned outputs, sorted case order, Go then ORT execution,
input construction, conversion, upstream and top-1 checks retain their original
boundaries. Builds, loading and serialization remain outside inference timing.
The ordinary ramp is separately index%256 / 255. Its timing is never divided
by the fixed-policy ORT timing.

| Set / engine | Median µs | Range of report medians µs | CV | Median within-report p95 µs |
| --- | ---: | --- | ---: | ---: |
| Initial / Go | 62.2705 | 61.1045–63.4375 | 1.23% | 116.3771 |
| Initial / ORT | 42.0000 | 41.625–43.625 | 1.57% | 45.1670 |
| Final control / Go | 62.3750 | 62.208–62.8955 | 0.42% | 114.9914 |
| Final control / ORT | 44.0420 | 41.209–44.541 | 3.52% | 45.0420 |
| Final candidate / Go | 61.7500 | 59.542–64.042 | 2.29% | 112.5923 |
| Final candidate / ORT | 44.1670 | 41.625–45.041 | 2.66% | 45.2910 |

The initial ratio is **1.4826×**, rather than the historical 1.4006×: Go is in
the expected region but ORT initially runs faster. The final control is
**1.4163×**. Reference movement is recorded, not attributed to Go optimization.
CV is sample SD / mean across report medians; p95 is the median of the seven
within-report p95s. It is not a percentile of benchmark averages.

| Ordinary ramp set | Median µs | Range µs | CV | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: | ---: |
| Initial | 61.659 | 60.789–63.086 | 1.18% | 103,691 | 93 |
| Final control | 63.026 | 62.526–63.275 | 0.41% | 103,691 | 93 |
| Final candidate | 62.114 | 61.482–62.258 | 0.42% | 93,322 | 88 |

All valid samples remain. An initial production prototype measured
**62.6875 → 61.584 µs**, saving 1.1035 µs; an intermediate complete seven-pair
set measured **62.479 → 61.000 µs**, saving 1.479 µs. A subsequent set measured
**62.500 → 61.584 µs**, saving 0.916 µs. The final set follows load-only guard
refinements: allocation-free budget computation and allowance for invalid input
bindings before declaration validation. The eligible inference algorithm did not
change between these sets. One final paired difference is negative: paired
savings span **−1.5835–2.666 µs**, median **0.9165 µs**, CV **155.78%**. The
difference of engine medians, **0.625 µs**, is the primary result; we do not switch
estimators to claim the strong-accept band. Neither tail improvement nor
sub-microsecond measurement precision is established.

Replay scripts, original-HEAD overlays, diagnostic sources, JSON comparisons,
profiles and statistics are local under ignored `tools/compare_runtime/pass13/`;
temporary builds are under `/private/tmp/onnx-pass13`. `measure.py` provides
prepare/baseline/ceiling/final/profiles phases, `checks.py` the correctness matrix,
and `supplement.py`/`memory.py` load, disassembly and budget diagnostics.
`summarize.py` checks model/input/policy/driver fingerprints and exact Go output
bits against Pass 12. An accidentally overlapping, interrupted diagnostic batch
is archived under `data/overlapped-discarded/` and was rerun sequentially. No
timing-value exclusions were made. Later measurement scripts use a shared lock.
`data/before-binding-budget-proof/` retains the preceding complete set. Final
Run comparisons, initialization, census, CPU/allocation profiles and the complete
correctness matrix were repeated after that proof changed. Kernel-only/stripped
probes precede the final load-only refinement; their eligible plan, numerical
methods, operands and destinations are unchanged.

## Pass 12 Audit

The fresh matched setup probe reproduces **4.528 → 1.196 µs**:
**9,232/59 → 160/12 B/allocations**. Dynamic range is 4.500–4.583 µs, CV 0.79%;
prepared range is 1.182–1.203 µs, CV 0.73%. The initial production allocation
result is still exactly **103,691 B/op / 93 allocs/op**. This supports the
existing Pass 12 attribution; its historical 2.8545-µs production saving is
already in the baseline and is not added to this pass.

Descriptors remain immutable. Runtime arguments are freshly gathered and placed
in local call copies; no caller tensor, context or destination is retained.
There is no signature cache or learning path. Unknown/symbolic metadata and
overridable shape contents retain the original constructor/planner path.
Current-value/declaration checks remain before consumers; backend kernels are
unchanged. Fresh dynamic Add probes measure **0.9239 → 0.9233 µs**, and
alternating signatures **0.9303 → 0.9315 µs**, with unchanged allocations.
Pass 12 differential cancellation, fallback, symbols, defaults and lifetime
tests also pass. Its metadata saving is not being recovered a second time.

## Constant Reshape Trace

| Property | Observed value |
| --- | --- |
| Node | index 9, `Times212_reshape1`, Reshape, opset 13, allowzero=false |
| Data | value 1, initializer `Parameter193`, producer −1, not a graph input |
| Data dtype/shape | float32, `[16,4,4,10]`, 2,560 elements |
| Target | value 7, initializer `Parameter193_reshape1_shape`, producer −1, not a graph input |
| Target dtype/contents | int64, shape `[2]`, contents `[256,10]`, 16 B |
| Result | value 18, `Parameter193_reshape1`, float32 `[256,10]`, **10,240 B** |
| Consumer | index 10, `Times212`, MatMul input 1; the activation is `[1,256]` |
| Fanout | exactly one input occurrence/consumer, proven by enumerating every step and the lifetime consumer table |
| Original lifetime | produced at 9, retired after 10; no graph-output exposure |

Originally the result owns a fresh disjoint backend destination every Run. The
existing free activation buffer has a different shape, so this Reshape cannot
reuse it. Pass 12 already prepared the target and resolved shape. Run still
reserves 16 B workspace, acquires/reserves the 10,240-B destination, allocates
zero-filled tensor storage and copies all 2,560 values. After MatMul, retirement
makes this weight the free buffer; final Add needs `[1,10]`, so it is dropped.

Backend ReshapeInto validates source and destination, resolves equal element
counts, checks dtype/shape/disjoint storage and performs a row-major copy. ONNX
also checks current values, output declarations, context boundaries and budgets.
Failures can include cancellation, workspace/live/tensor limits, invalid current
data/target or incompatible shape. Proven immutable valid operands make the
eliminated repeated backend validation invariant; Run checks remain in place.

## Ceiling Reproduction

Two independent probes reproduce the direction before production design:

| Probe | Control µs | Retained µs | Difference of medians | B/op before → after | allocs/op |
| --- | ---: | ---: | ---: | --- | --- |
| Inherited combined, seven repetitions | 63.159 | 61.842 | 1.317 µs | 103,531 → 93,162 | 81 → 76 |
| Guarded current-runtime diagnostic, nine alternating pairs | 62.719 | 61.461 | 1.258 µs | 103,691 → 93,306 | 93 → 87 |

Inherited ranges/CVs are **62.720–64.041 / 0.68%** and
**61.705–62.191 / 0.31%**. Guarded ranges/CVs are
**60.954–66.224 / 2.36%** and **61.128–65.581 / 2.26%**. Guarded paired savings
have median **1.144 µs**, range **−0.244–3.536**, CV **85.46%**; eight of nine
pairs are positive. This establishes an approximately 1-µs direction, not an
exact 1.036-µs constant or a tail-latency guarantee.

The guarded probe retains binding, argument gathering, current-value checks,
all selected context checks, workspace/destination reservations, publication,
retirement and the same two within-Run reuses. It asserts that the traced weight
never acquires a reused destination. It is a fixed-model diagnostic: it does
not itself add resident accounting or general protection against retirement
into writable storage. Its helper also lets the compiler remove one argument
slice allocation; that extra 16 B / one allocation is **not** a production win.
The production result removes only five allocations and charges/pins the copy.
Persistent payload required by either retained-copy experiment is 10,240 B.
Computing the already-planned copy measures about 1.061–1.076 µs; complete
optional materialization, measured later, is 1.551 µs.

## Cost Decomposition

Seven-repetition isolated scopes overlap; they must not be added into a Run:

| Scope | Median µs | Range µs | CV | B/op / allocs |
| --- | ---: | --- | ---: | --- |
| Allocate/zero tensor | 0.6016 | 0.4386–0.6437 | 12.76% | 10,353 / 4 |
| ReshapeInto with existing destination, including validation | 0.1888 | 0.1841–0.1943 | 1.82% | 16 / 1 |
| Prepared shape call, fresh destination | 1.0760 | 1.002–1.087 | 3.44% | 10,369 / 5 |
| Current output/declaration validation | 0.06032 | 0.06011–0.06063 | 0.29% | 0 / 0 |
| Workspace/destination reservations | 0.008203 | 0.008190–0.008239 | 0.20% | 0 / 0 |
| Six selected context checks | 0.008999 | 0.008918–0.009696 | 3.01% | 0 / 0 |
| Original dynamic metadata constructor/planner | 0.4023 | 0.4011–0.4448 | 3.94% | 1,216 / 7 |

The last row is already absent from the Pass 12 production baseline. None of
it is counted again. Zeroing is included in tensor.New; a separate zeroing-only
cost was not isolated. The copy-only row also includes backend validation and
its resolved-shape allocation. Destination acquisition remains; actual payload
allocation/zeroing/copy disappear. Retirement changes by pinning this result;
its separate latency is not isolated. The noisy full-stage census is not a
substitute for the guarded or production measurements.

## Immutability Proof

The existing `preparedCallConstants` table seeds only non-input immutable
initializers, records Constant nodes, and propagates the same provenance through
Identity chains. This pass reuses that table. It neither evaluates arbitrary
constant subgraphs nor marks a computed Reshape as new provenance for further
folding. Data and target must both have this proof; target metadata must already
be eligible for the Pass 12 descriptor.

Decoded constants own their dtype, shape and payload; public tensor constructors
and accessors copy. Borrow is internal and read-only. Source/Constant/Identity
storage is pinned, so neither operand can become a destination. Both selected
operands are non-input initializers, with no override, runtime symbol or
session-local dependency. Declarations and symbol constraints still run normally.
The prepared result is a separate owned tensor. MatMul stages/reads its inputs
and writes only its disjoint destination; it does not modify either operand.

## Override Analysis

The new matrix crosses eight data and eight target modes: initializer, Constant,
Identity, Identity chain, graph-input default, Identity of default, chain of
default, and required input. Only the first four modes on **both** sides qualify.
Runs exercise default → A → B → default. An immutable data tensor with an
overridable target stays dynamic; an overridable data tensor with an immutable
target can retain metadata but never a prepared result. A malformed default
target `[-1,-1]` still loads, fails when selected at Run, and can be replaced by
a valid override. Invalid current signatures retain original validation/errors.

Malformed immutable targets with negative dimensions, repeated inference,
incompatible counts, invalid zero positions or overflow retain their existing
compile/load failures. Known invalid dtype/rank and malformed encoded tensor
contents are already rejected by the compiler/decoder. Optional computation
does not introduce a new load error: failed materialization leaves the Run path;
load cancellation is propagated. Inferred, copied-zero, literal-zero, empty,
scalar and int64 results are covered without tolerance changes.

## Copy vs View

| Choice | Added payload | Ownership/complexity | Decision |
| --- | ---: | --- | --- |
| A. Prepared owned copy | 10,240 B | Existing backend constructor, immutable descriptor, plan reference, pinned lifetime | Accept with quota guards |
| B. Zero-copy immutable view | Potentially 0 B | Backend fields are private; no public shape-view API; shared identity and payload lifetime/accounting would need an explicit new contract | Reject for this pass |
| C. Per-Run copy | 0 B retained | Existing fallback; about 1 µs repeated cost | Preserve whenever preparation is ineligible |

Tensor stores dtype, owned shape, numeric slices and a storage-identity pointer
in one object. Shape/value accessors copy; there is no supported same-payload,
different-shape constructor. A backend view could theoretically share identity,
but would require backend API, ownership/alias tests, lifetime protection and
quota decisions. Public alias exposure and writable destination reuse would
need auditing. That exceeds a nearly free existing option. No unsafe access,
general tensor-view system or backend change was introduced.

## Memory Accounting

| Quantity | Before | After |
| --- | ---: | ---: |
| Resident numerical payload, shared plan | 24,008 B | 34,248 B |
| Added persistent numerical payload | 0 | **10,240 B** |
| Charged cumulative load work | 491,933 B | 502,205 B |
| Peak Run logical live payload, excluding workspace | 101,328 B | 111,568 B |
| Peak reserved workspace | 150,528 B | 150,528 B |
| Peak combined live + workspace | 251,856 B | 262,096 B |
| Ordinary Run allocation | 103,691 B / 93 | 93,322 B / 88 |

The copy increases this small model's resident numerical payload by **42.65%**.
It is model/plan-owned, not session-owned. The selected 16-B Run workspace
reservation and logical destination reservation remain. The result is pinned
rather than retired; its logical Run reservation can therefore outlive its last
consumer. These costs are included in the conservative eligibility bound.

Preparation has no additional numerical scratch beyond the new 10,240-B result.
It allocates a 16-B temporary backend resolved shape; the result's owned shape is
16 B. Both shapes plus payload are charged to MaxLoadBytes: **10,272 B**.
The extra numerical peak during preparation is 10,240 B while the original source
remains resident. The cumulative load figure is the repository's logical upper
bound, not measured process RSS or exact simultaneous Go heap. Full-load Go
allocation is reported separately below; an actual RSS/heap peak was not measured.

An independently instrumented budget confirms the exact old MNIST combined
boundary: loading with **MaxLiveBytes=251,856** declines preparation and succeeds;
**251,855** declines it and fails at the same ReLU32 reservation in both versions.
The retained 10 KB is never silently discounted when physically present.

## Preparation Budget

Pass 12 descriptors are prepared first, preserving their quota priority. Only
then may a proven constant-only Reshape retain a result. Eligibility requires:

1. Existing immutable provenance for data and target and a valid prepared shape.
2. MaxTensorBytes and MaxWorkspaceBytes compatibility.
3. Remaining MaxLoadBytes for result payload plus owned/temporary shape arrays.
4. Cumulative live headroom for resident constants, **all** named value payloads,
   **MaxTensorBytes for every graph input**, all graph-output publication copies,
   the configured MaxWorkspaceBytes, and all additionally prepared payloads.

The live bound uses checked multiplication and scalar backend dtype sizing with
**zero allocations**. Unknown dimensions decline nonempty optional payloads.
LayerNormalization declines preparation because its optional unnamed statistics
are not covered by the named-slot bound. This intentionally conservative policy
does not build a second lifetime planner or a runtime invalidation mechanism.
It may decline otherwise affordable preparations on constrained models.
Input slots use MaxTensorBytes because bind reserves supplied storage before
checking its declaration. Bounding only valid input shapes could change an invalid
call's ErrInvalidArgument into ErrLimit. A tight-budget regression proves the
original error priority and subsequent valid-call recovery. Models with many
inputs may therefore decline preparation even when their valid inputs are small.

Insufficient optional budget leaves the previous Run computation intact and
does not increase the original required minimum load budget. Tests cover exact
load/live boundaries, one-byte shortages, cumulative copies, defaults, unknown
values and normalization's unpublished buffers. Preparation is bounded by existing
per-tensor, cumulative load and live limits, with no MNIST-specific size threshold.

## Initialization Cost

| Scope | Before µs | After µs | B/op before → after | allocs before → after |
| --- | ---: | ---: | --- | --- |
| Isolated new live bound | — | 0.5065 | 0 | 0 |
| New optional materialization, including bound and selection | — | **1.551** | 10,369 | 5 |
| Execution-plan construction | 11.003 | 12.736 | 32,536 → 42,992 | 177 → 182 |
| Compile graph | 185.103 | 185.066 | 173,016 → 173,016 | 991 → 991 |
| Complete warm LoadBytes/Close | 385.194 | 381.382 | 334,867 → 345,317 | 2,558 → 2,563 |
| Session creation/close | 0.02596 | 0.02619 | 24 → 24 | 1 → 1 |

Materialization: seven samples, **1.546–1.557 µs, CV 0.26%**. Live bound:
0.5047–0.5110 µs, CV 0.38%. Plan construction: five samples each,
**10.913–11.099 / 0.70%** before and **12.706–12.822 / 0.40%** after.
Full loading: seven samples each, **375.936–404.531 / 2.72%** before and
**377.702–384.250 / 0.69%** after. Every sample remains.

These are separate timers, not additive measurements. Isolated preparation costs
**1.551 µs**; the complete plan median delta is **1.733 µs**. At 0.625 µs/Run,
these amortize after **2.48** and **2.77 Runs**, both rounded up to **three**.
The full warm-load subtraction is −3.812 µs; an earlier valid set instead gave
**+5.103 µs**. Neither sign establishes a causal load speedup or precise total
increment. We retain that uncertainty and charge the positive complete-plan
increment for primary amortization. Charging the earlier full-load delta as a
sensitivity gives **8.16 Runs, rounded up to nine**. First process-load comparison
timings are still noisier (candidate CV 18.4%) and cannot resolve this increment.

| Runs | Net µs saved, charging 1.551-µs materialization | Charging 1.733-µs plan delta | Charging earlier 5.103-µs load delta |
| ---: | ---: | ---: | ---: |
| 1 | −0.926 | −1.108 | −4.478 |
| 2 | −0.301 | −0.483 | −3.853 |
| 10 | 4.699 | 4.517 | 1.147 |
| 100 | 60.949 | 60.767 | 57.397 |
| 1,000 | 623.449 | 623.267 | 619.897 |

Persistent payload per microsecond of steady-state Run saving is approximately
**16,384 B/µs**. This workload-specific metric does not justify arbitrary folding
of larger tensors or single-use models.

## Lifetime and Concurrency

The execution plan owns the copy. Lifetimes are constructed after descriptors
and optional payloads, so the Reshape output and all Identity aliases are pinned
before retirement is scheduled. It never enters writable free storage. Existing
destination acquisition still executes; any private reusable destination it
acquires is not used to modify the retained tensor. Current output validation
and owned publication still execute. MatMul only reads the weight.

Model.Close drops its reference; surviving sessions retain the same plan and
same prepared pointer. Session.Close drops its reference, and the last reference
permits GC. Cleanup tests prove both no-session and surviving-session lifetime,
and reclamation after final closure. Concurrent separate sessions repeatedly
read one prepared tensor and publish independent bit-exact outputs, including
NaN payload/sign/signaling bits, infinities, subnormals and signed zero. Repeated
load/close cycles, same-session exclusion, races and ordinary lifecycle tests pass.

Run retains the two eliminated copy context boundaries, both prepared-plan
boundaries, argument checks, and the surrounding execution-step boundaries.
Differential tests match context-check counts against the original dynamic path
and cancel at every Run check. New load preparation is also canceled at every
observed load check. No required execution-step boundary is removed.

## Acceptance Decision

**ACCEPT.** This uses the conditional 0.5–0.8-µs guideline. The final primary saving
is **0.625 µs**, with paired median 0.9165 µs. The other complete sets' primary
savings are 1.1035, 1.479 and 0.916 µs; all agree in direction with both ceilings.
This supports a modest aggregate gain, with considerable per-pair noise, rather
than a guaranteed strong-accept result. The implementation is generic constant-only
Reshape, one owned copy, existing immutable descriptors and existing pinning, with
conservative optional quota guards. It adds no public API or runtime invalidation.
The explicitly charged 10,240-B payload is shared by sessions, and tight budgets
retain the old path. There is no view system, new IR, general subgraph executor,
numerical change or dependency update. Those properties justify conditional
acceptance of this small saving; a broader folding/view design would not qualify.

The saving is modest and individual paired differences remain noisy. Acceptance
is based on reproducible aggregate direction, exact allocation removal, bounded
ownership and quota behavior; it does not assert every Run is faster or that the
entire profiled Reshape stage is recoverable. Single-use models do not amortize
the extra preparation under any reported initialization scope.

## Implementation

`execution_plan.go` reuses constant provenance, prepares descriptors before
optional copies, checks cumulative load/live headroom, and computes only eligible
Reshape results. Failures leave the original computation available; cancellation
still fails loading. `prepared_call.go` stores an optional immutable owned result
and returns it after the original reservations/checks. `value_lifetimes.go` pins
that result and its Identity group. No backend or arithmetic file changes.

## Correctness

The new tests cover the 64 data/target provenance combinations and transitions;
malformed defaults replaced by overrides; invalid immutable targets/overflow;
inference, zero modes, empties, scalar and int64 payloads; exact/cumulative load
budgets and conservative live fallback; workspace limits; aliases and downstream
input immutability; Run/load cancellation; repeated Runs; shared/concurrent
sessions; Model.Close with surviving sessions; final Session.Close and GC.

The complete existing suites add malformed decoding/dtypes, symbolic dimensions,
destination and ownership contracts, limits, numerical cases, and lifecycle
coverage. Validation completed successfully:

- gofmt and git diff --check; go vet ./...; go mod tidy -diff.
- Full uncached default, CGO_ENABLED=0, purego and race suites.
- Default and purego builds for darwin/arm64, darwin/amd64, linux/arm64,
  linux/amd64 and windows/amd64, all with cgo disabled.
- The four CI fuzz targets, three seconds each with two workers.
- All fixed-policy comparisons, unchanged tolerances/upstream/top-1 checks;
  exact Go output bits against the Pass 12 reports for every case.
- Instrumented old/new MNIST peak accounting and original exact Run boundary.

MatMul prepared-adapter latency is **1.709 → 1.717 µs**, ranges
**1.703–1.744 → 1.709–1.725**, CV **0.98% → 0.38%**. Allocations remain
**144 B / 3**. This 8-ns movement is not material evidence of a MatMul speedup.
Its source, reduction order, shape, values and backend dispatch are unchanged;
all complete graph output bits agree. A direct original-path comparison also
matches all 2,560 weight elements and all ten MatMul output elements bit for bit.
No MatMul saving is credited to this pass.

## Results

**62.375 µs before → 61.750 µs after → 44.167 µs ORT → 1.3981×.**
The control's own ORT median is 44.042 µs. Ordinary ramp is
**63.026 → 62.114 µs**, saving **0.912 µs / 1.45%** in that separate workload.
Final fixed-policy saving is **0.625 µs / 1.0020%**. A durable tail-latency gain
is not established; benchmark averages supply no per-call p95.

## Allocation and Persistent Memory

Exactly **10,369 B / five allocations** disappear per ordinary Run:

| Allocation site | Requested bytes removed | Allocations |
| --- | ---: | ---: |
| tensor.newTensor: float32 payload | 10,240 | 1 |
| tensor.newTensor: Tensor struct | 88 requested, 96 allocator size class | 1 |
| tensor.newTensor: owned shape | 16 | 1 |
| tensor.newTensor: identity | 1, amortized tiny allocation | 1 |
| ReshapeShape: copied resolved shape | 16 | 1 |

Requested total is 10,361 B; measured allocation including the struct size class
is 10,369 B. Argument gathering remains. There is no Pass 12 planner allocation
in this list. Rate-one allocation profiles remove one newTensor subtree and one
ReshapeShape invocation per Run, retaining activation Reshape, MatMul staging,
broadcast metadata, current argument arrays and output cloning.

Persistent requested Go metadata adds **201 B**: twelve descriptor pointers
(96 B), one Tensor header (88 B), owned shape (16 B) and identity (1 B).
The Pass 12 7,904 B / 39 metadata objects become approximately **8,105 B /
42 metadata objects**, plus the separate **10,240-B / one-object payload**.
The extra four persistent objects and one temporary resolved-shape allocation
explain the five preparation allocations. Allocator rounding/tiny-block packing
differ from requested metadata bytes. No payload is duplicated per session.

## Numerical Graph Delta

The preallocated, stripped numerical graph remains about 52 µs if the copy is
deliberately retained. In the same baseline binary, seven samples measure
**52.067 µs** (51.852–52.340, CV 0.30%) versus **51.747 µs** with only the
constant Reshape skipped (51.643–52.083, CV 0.31%): **0.320 µs** of numerical
work. This graph already has destinations, so it cannot measure the production
allocation saving. Allocation is **1,088/31 → 1,072/30 B/allocations**.

The candidate binary reproduces **51.909 → 51.594 µs**, a **0.315-µs** delta;
the no-constant scope ranges 51.504–52.691 µs, CV 0.82%. Ordinary production
is 62.114 µs, leaving a broad **10.520-µs scope difference** for binding,
validation, destinations, budgets, lifetime and owned publication. It is not a
deployable saving. The older PreparedGraph benchmark still explicitly executes
both Reshapes and measures 51.982 µs; it is not silently relabeled as the new
per-Run graph. Stripped ramp measurements are not equivalent to complete ORT
inference on nineteen fixed-policy inputs.

Prepared-node composition after this pass is approximately Conv **34.938 µs**,
preserving ReLU **7.359**, Pool **3.584**, Add **3.582**, MatMul **1.717**, plus
the remaining activation Reshape. Independently timed component medians do not
form an exact additive graph identity.

## Excess-Latency Ledger

Fresh ORT values below are medians across three repetitions of sums of profiled
node means. Events have integer-microsecond duration resolution, include full
graph node scopes, and differ from unprofiled prepared Go calls. Unknown ORT
runtime cost remains unknown. These rows overlap and are not forced to sum.

| Region | Before Go µs | After Go µs | Coarse ORT µs | Recoverable excess / confidence |
| --- | ---: | ---: | ---: | --- |
| Conv, prepared | 34.913 | 34.938 | 36.387 | No positive excess demonstrated; closed |
| Preserving ReLU, prepared | 7.277 | 7.359 | 2.000 | 5.359 raw indication; recoverable production amount unproved |
| Pool, prepared | 3.515 | 3.584 | 3.017 | ~0–0.4 plausible; low confidence; no optimization here |
| Add, prepared | 3.496 | 3.582 | 2.846 | Planning/fusion investigation; recoverable net unknown |
| MatMul, prepared | 1.709 | 1.717 | 1.000 | 0.717 raw indication; contract-preserving transfer unproved |
| Both Reshapes, full stage census | 2.916 | 0.616 | 0.592 | Accepted whole-Run saving 0.625; mixed allocation/runtime scope |
| Binding/evaluation, separate probe | 0.946 | 0.938 | Unknown | Required current binding; no new removal demonstrated |
| Walk/check/setup | 1.196 | Same algorithm | Unknown | Pass 12 already removed selected metadata |
| Destination/lifetime | Included in full stages | Included in full stages | Unknown | Required semantics; remaining ceiling unknown |
| Publication, census | 0.131 | 0.128 | Unknown | Owned output copy retained |
| Unresolved | Not isolated | Not isolated | Unknown | No invented residual assignment |

Complete gap changes **18.333 → 17.583 µs** using each set's own ORT median.
Go contributes 0.625 µs; reference movement contributes +0.125 µs. Most of the
gap remains unresolved. Coarse ORT reshape means vary **0.282–1.129 µs** across
profiles, illustrating why that row cannot supply exact excess accounting.

Full-stage census medians, from five independent instrumented means:

| Complete stage | Before µs | After µs | B/Run before → after | allocs/Run |
| --- | ---: | ---: | --- | --- |
| Evaluation/bind | 1.082 | 0.968 | 448 → 448 | 4 → 4 |
| Convolution28 | 10.048 | 9.323 | 27,456 → 27,456 | 6 → 6 |
| Plus30 | 3.127 | 4.146 | 27,640 → 27,640 | 12 → 12 |
| ReLU32 | 5.279 | 5.290 | 8 → 8 | 1 → 1 |
| Pooling66 | 3.054 | 3.437 | 6,712 → 6,712 | 6 → 6 |
| Convolution110 | 28.086 | 28.861 | 13,760 → 13,760 | 6 → 6 |
| Plus112 | 2.867 | 2.473 | 13,944 → 13,944 | 12 → 12 |
| ReLU114 | 2.691 | 2.700 | 8 → 8 | 1 → 1 |
| Pooling160 | 1.346 | 1.366 | 1,208 → 1,208 | 6 → 6 |
| Activation Reshape | 0.468 | 0.412 | 1,184 → 1,184 | 6 → 6 |
| Weight Reshape | 2.448 | 0.204 | 10,400 → 16 | 6 → 1 |
| MatMul | 1.981 | 1.961 | 336 → 336 | 8 → 8 |
| Plus214 | 0.482 | 0.530 | 544 → 544 | 13 → 13 |
| Publication | 0.131 | 0.128 | 208 → 208 | 6 → 6 |

Median complete census totals are **63.245 → 61.601 µs**; the sum of individual
stage medians need not equal the median total. Clock instrumentation, cache and
allocator-state effects make its 2.244-µs weight-stage delta unsuitable as a
claimed production saving. Per-stage ReadMemStats/tiny-allocation rounding also
makes its byte delta differ from ordinary B/op. The two actual destination
reuses remain unchanged. CPU profiles again contain the Darwin sampling artifact
(baseline 90.34% runtime.kevent); run samples show Conv, preserving ReLU and Add
but cannot precisely attribute sub-microsecond setup. Whole-Run pairs, allocation
profiles and matched component probes carry the quantitative claims.

## Updated 1.25× Roadmap

At the final **44.167-µs** ORT reference, **1.25× = 55.20875 µs**. Current Go
**61.750 µs** needs **6.54125 µs** more saving. On the control's own reference
the requirement was 7.3225 µs; the change includes 0.625 µs Go recovery and
0.15625 µs of target movement from the different ORT reference.
For historical normalization only, ORT 44.833 gives target 56.04125 and remaining
5.70875 µs; that is not today's measured ratio or an additional Pass 13 win.

ReLU's fresh 5.359-µs raw difference remains strategically large but is not a
demonstrated recovery. Even granting all of it would nominally leave **1.18225 µs**
on today's reference. Add planning/fusion, MatMul and execution/destination
mechanisms remain conditional candidates; their overlapping ceilings cannot be
added. Conv remains closed and Pool remains low priority.

## Updated 1.10× Roadmap

**1.10× = 48.5837 µs**, requiring **13.1663 µs** further saving. Historical
44.833 normalization instead gives 49.3163 and 12.4337 µs remaining. Neither
reference makes another small constant/metadata cleanup sufficient. The stripped
ramp's approximately 51.6 µs is useful evidence about necessary Go numerical work,
not an equivalent-scope ratio or a proof about every fixed-policy input.
Faster necessary activation/product work, complete-contract fusion, destination
execution changes, or another demonstrated mechanism would be required. No
feasibility promise or reopening of Conv follows from the present evidence.

## Is 1.25× a Single-Bottleneck Problem?

**Possibly, but ORT scope prevents conclusion.** ReLU is the largest remaining
specific signal. Its raw difference is smaller than today's required recovery,
and no complete-contract new ReLU saving has been measured. The remaining general
Run scope is also substantial. A single-mechanism claim is unsupported; the
roadmap must retain several mechanisms until new backend evidence changes it.

## Candidate Ranking

This ordinal ranking weighs absolute recoverable opportunity, confidence and
complexity; it does not invent numeric confidence scores or add overlapping rows.

| Rank | Candidate | Evidence / limitation |
| --- | --- | --- |
| 1 | Preserving ReLU investigation | 7.359 µs complete adapters, 7.373 µs preserving backend; 2.000 coarse ORT; scalar loop confirmed; net recovery unproved |
| 2 | Backend Add planning | 3.582 µs versus 2.846 coarse ORT; repeated broadcast work remains; fresh complete-contract ceiling unknown |
| 3 | MatMul | 1.717 µs versus 1.000 coarse ORT; most time remains necessary work; small backend investigation, no change here |
| 4 | Add+ReLU fusion | Potentially shares traversals/destinations; high contract coupling; overlaps ranks 1–2 |
| 5 | Destination/execution | Broad ~10.5-µs scope difference; irreducible work and unclear recoverability; substantial complexity |
| 6 | Remaining constant work | Selected computation is removed; no next proven constant payload computation identified; broad folding not justified |
| 7 | Pool | Small indication, weak scope agreement; low priority |
| 8 | Conv | No demonstrated positive excess; arithmetic remains closed |

## Pass 14 Recommendation

Investigate the **cause** of preserving ReLU cost in neuralnetwork. Fresh data
still gives about 7.4 µs and a large coarse ORT difference after constant work is
removed. Source and Apple M3 disassembly show scalar element processing in
ReLUPreservingNaNInto, with no preserving SIMD dispatch. This identifies a useful
question, not an instruction to write assembly. Quantify validation, polling,
loop behavior, traffic and realistic complete-contract recovery first.

The separate legacy NaN-to-zero ReLUInto probe still takes **6.392 µs** across
the same two calls, versus **7.373 µs** in the preserving backend. Their different
semantics prevent treating the 0.981-µs difference as a cancellation-only cost;
they also show that preserving NaN classification alone cannot explain all the
observed Go time. The standalone prompt includes this evidence.

## Standalone NeuralNetwork ReLU Investigation Prompt

The independently usable prompt is in
[NEURALNETWORK_RELU_PASS14_PROMPT.md](NEURALNETWORK_RELU_PASS14_PROMPT.md). It
contains the measured workload, policy, semantics, current scalar implementation,
scope cautions, targets and acceptance requirements. No neuralnetwork changes
were made from this repository.
