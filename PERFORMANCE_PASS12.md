# Performance Optimization Pass 12

2026-11-11, Apple M3. **65.646 → 62.7915 µs Go → 44.833 µs ORT → 1.4006×**.
Guarded operator-call metadata preparation saves **2.8545 µs / 4.35%** in the
final seven-pair comparison. **112,763 → 103,691 B/op; 140 → 93 allocs/op**.
Retained Go metadata: **7,904 requested bytes / 39 allocations**; added tensor
payload: **0 B**. Isolated preparation: **3.847 µs**, approximately two Runs to
break even. Numerical kernels, dependency versions and comparison policy are unchanged.

## Current Baseline

Starting HEAD is `df5068860a96eec8e762d8e9604643808348ada8`, the merge of Pass 11
PR #32. neuralnetwork is **v0.10.7**, commit
`1f033b7d13788661a9002462151a997f30b620b4`, module sum
`h1:LPdDwSZz5MkiYGL1UAiMVZmpITvu5xVX4PHKoaf4BEw=`. Production has no module
replacement. The final source is the working-tree change accompanying this report.

Go **1.27.2**, darwin/arm64, macOS **26.5.2 (25F84)**, Apple M3,
**GOARM64=v8.0**. Performance processes use **GOMAXPROCS=1, CGO_ENABLED=0,
GOWORK=off**, default optimization, no build tags/gcflags; the comparison build
uses `-buildvcs=false`. Host CGO_ENABLED defaults to 1; GOMAXPROCS is otherwise
unset. ORT **1.19.2**, CPUExecutionProvider, ORT_SEQUENTIAL, one intra-op and one
inter-op thread, ORT_DISABLE_ALL, default CPU arena and memory pattern. Reference
Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2 and protobuf 4.25.3 are unchanged.

The established policy still uses upstream/zero/one and sixteen seed-0 uniform
random inputs: 19 cases, ten warmups and 100 timed calls per case; atol=1e-4,
rtol=1e-5. Input construction, timing boundaries, owned outputs, execution order,
conversion, upstream and top-1 checks are unchanged. Builds/loading/serialization
are outside inference timing. The separate ordinary ramp is index%256 / 255;
its timing is never divided by fixed-policy ORT timing.

The initial seven repetitions reproduce **65.375 / 44.417 = 1.47185×**.
Statistics are sample SD / mean across report medians; p95 is the median of the
seven within-report p95 values, not a percentile of benchmark averages.

| Set / engine | Median µs | Range of report medians µs | CV | Median within-report p95 µs |
| --- | ---: | --- | ---: | ---: |
| Initial / Go | 65.3750 | 64.333–65.708 | 0.66% | 119.883 |
| Initial / ORT | 44.4170 | 44.125–46.105 | 1.49% | 45.000 |
| Final control / Go | 65.6460 | 61.375–69.792 | 3.84% | 116.960 |
| Final control / ORT | 44.6250 | 44.292–45.333 | 0.94% | 46.000 |
| Final prepared / Go | 62.7915 | 61.750–63.521 | 1.16% | 117.217 |
| Final prepared / ORT | 44.8330 | 43.875–45.521 | 1.13% | 46.419 |

| Ordinary ramp set | Median µs | Range µs | CV | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: | ---: |
| Initial | 67.522 | 66.924–78.921 | 6.19% | 112,763 | 140 |
| Final control | 67.345 | 66.465–70.301 | 1.80% | 112,763 | 140 |
| Final prepared | 63.601 | 63.116–64.837 | 0.86% | 103,691 | 93 |

All samples remain. An earlier complete seven-pair production set measured
66.021 → 62.792 µs, saving 3.229 µs; its candidate ramp includes a 102.884-µs
sample and has 22.13% CV. That set is retained separately. The final set follows
the load-time change to compute constant provenance in one graph walk rather
than repeatedly following Identity chains. The inference path did not change.
Neither set establishes improved tail latency; the final Go p95 is slightly higher.

Replay scripts, frozen source overlays, raw comparisons, censuses and profiles
are local under ignored `tools/compare_runtime/pass12/`, with temporary builds in
`/private/tmp/onnx-pass12`. `data/summary.json` verifies policy fingerprints and
bitwise outputs against Pass 11. `data/provenance.json` records final production
source hashes. The preexisting user edit to `.gitignore` is preserved. Production
code, tests, benchmarks and this report are outside tools/.

## Pass 11 Audit

The audit freezes pre-retention `evaluation.go` from `4d4b36d` against the same
v0.10.7 backend, then compares it with current Pass 11 source. It does not use
v0.10.6 or change numerical adapters.

| Scope | Clone control µs | Retained constants µs | Allocation before → after |
| --- | ---: | ---: | --- |
| Binding plus evaluation creation, five repetitions | 3.673 | 0.948 | 26,288/36 → 448/4 B/allocs |
| Fixed-policy complete Run, five alternating pairs | 68.3545 | 65.5830 | 138,608/172 → 112,763/140 B/allocs |
| Prepared numerical graph | 52.468 | 52.215 | 1,088/31 → 1,088/31 B/allocs |

The fresh whole-Run saving is **2.7715 µs**, versus historical 3.271 µs.
Control range is 66.459–68.855 µs, CV 1.39%; retained range is 65.084–67.980,
CV 1.73%. Binding attribution and allocation changes reproduce directly;
process timing varies. No numerical source changed, and the small prepared-graph
timing difference is not an arithmetic improvement.

Inspection and regression tests confirm Borrow returns the already-decoded
tensor; initializer and Constant aliases remain pinned outside writable free
storage. Source payload is not copied or newly retained. Runtime defaults are
selected after overrides, publication still clones, and concurrent surviving
sessions remain valid after Model.Close. Those Pass 11 guarantees remain intact.

## Static-Metadata Ceiling Reproduction

Replaying the inherited preparation diagnostic on the Pass 11 baseline gives:

| Mode | Median µs, five repetitions | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Retained constants, ordinary call construction | 67.632 | 112,763 | 140 |
| Plus retained constant Reshape | 65.555 | 101,162 | 127 |
| Plus static call templates | 62.451 | 93,162 | 76 |

The incremental direction reproduces: **3.104 µs**, rather than historical
3.661 µs, and exactly **8,000 B / 51 allocations**. These are diagnostic scopes.
In particular, the inherited static helper combines shape workspace/destination
reservation and omits the two shape-plan context checks. Its retained-Reshape
branch also approximates a separate constant-computation optimization. It is
not used as a proof of production cancellation or quota equivalence.

A second, stricter prototype keeps **both Reshape computations**, current tensor
arguments, the original shape reservation sequence and both plan context checks.
It retains binding, current-value validation, backend calls, destination reuse,
retirement and owned publication. **67.733 → 63.852 µs**, a **3.881-µs** complete
inference diagnostic direction, **112,763/140 → 103,531/81 B/allocs**. It uses
validated fixed-model intermediates to prepare templates, so eligibility still
needs the production proof below. No kernel is substituted in either experiment.

## Setup Decomposition

The old walk/check/setup diagnostic measures **4.540 µs**, including **1.173 µs**
for walking, current-value checks and argument gathering. The final matched
setup-only harness measures **4.616 → 1.207 µs**, removing **3.409 µs** of setup
without executing numerical kernels, allocating destinations or publishing.
Whole-Run recovery is measured separately, not inferred from that subtraction.

The table contains overlapping scopes. Microbenchmarks are five-repetition
medians over the actual MNIST plan; they include their small traversal overhead.

| Component | µs | B/op | allocs/op | Static? | Guardable? | Must remain dynamic? |
| --- | ---: | ---: | ---: | --- | --- | --- |
| CNN call construction, four nodes | 1.142 | 3,232 | 16 | Geometry conditional | Concrete validated inputs | Tensor arguments and kernel validation |
| Numerical construction excluding MatMul, five nodes | 0.925 | 2,784 | 13 | Shape/workspace conditional | Concrete validated inputs | Data, destination, backend planning |
| MatMul call construction | 0.219 | 624 | 4 | Geometry conditional | Concrete validated inputs | Backend plan and arithmetic |
| Shape call construction plus planning, two nodes | 0.776 | 2,432 | 14 | Input shape plus target contents | Fixed input and immutable target | Copies, budgets, context checks |
| Twenty argument Shape snapshots, forced escape probe | 0.546 | 512 | 20 | Shape conditional | Yes | Validation snapshots still used |
| Rank plus dtype accessor probe | 0.639 | 0 | 0 | Conditional | Existing checks prove type/rank | Current tensors must be validated |
| CNN output geometry only | 0.417 | 128 | 4 | Shape/attributes conditional | Yes | Backend repeats its destination checks |
| Broadcast output-shape planning | 0.240 | 80 | 3 | Shape conditional | Yes | Backend stride planning remains |
| Backend stride plans for two bias Adds | 0.422 | 368 | 12 | Shape conditional | Requires backend preparation API | Current backend still plans each call |
| MatMulShape planning | 0.133 | 96 | 2 | Shape conditional | Yes | Backend product planning remains |
| MatMul arithmetic-only probe | 1.480 | 0 | 0 | Current data dependent | Geometry only | Arithmetic and full-contract staging/validation |
| ResolveReshapeMode, both targets | 0.383 | 800 | 8 | Contents and shape conditional | Immutable target plus fixed source | Runtime targets use original planner |
| Int64 target accessor staging | 0.105 | 32 | 2 | Contents conditional | Only immutable sources | Overridable target contents remain dynamic |
| Output declarations, rank/type/dimension checks | 0.988 | 0 | 0 | Contracts static, values dynamic | These checks supply the guard | Yes, including symbol bindings |
| Twelve reservation/release probes | 0.096 | 0 | 0 | Byte request may be prepared | Request only | Live/free/workspace budget state |
| Context-check traversal | 0.083 | 0 | 0 | No | No | All existing checks remain |
| Per-step input slices and slot gathering | 0.268 | 160 | 12 | Slot indices already compiled | Indices only | Current tensor pointers and slices remain |
| Lifetime-table lookup probe | 0.076 | 0 | 0 | Lookup table already compiled | Already prepared | Retirement/free storage remain dynamic |
| Prepared-descriptor lookup probe | 0.076 | 0 | 0 | Descriptor presence static | Direct indexed lookup | Probe includes loop cost, not a pure guard delta |
| Resolved attributes and slot indices | Already compiled | 0 new | 0 new | Model invariant | Already proved | No new saving attributed |
| Benchmark symbolic binding | No symbols in MNIST | — | — | Bindings Run invariant | Never cached | Dynamic-model symbol checks remain |

CNN construction divides into **0.644 µs Conv / 0.498 µs Pool**. These include
dtype access, snapshots, geometry, limits and call allocation; they do not include
Conv or Pool arithmetic. Geometry, snapshots and type probes overlap these rows.
Exact backend broadcast-stride and arithmetic probes are reported below.

## Invariance Classification

| Class | Fields / behavior | Treatment |
| --- | --- | --- |
| Model invariant | Operator/schema, resolved attributes, slot indices, declared constraints, non-input constants and their Identity provenance | Reuse existing compiled Step; prove immutable Reshape target in one load-time walk |
| Session invariant | Configured limits and backend build selection | Plan-owned; backend-specific workspace formulas remain unchanged |
| Shape-signature invariant | CNN pads/configs/output geometry, numerical output shape/workspace, MatMul output geometry, Reshape resolved shape | Prepare only if every present input has concrete dtype and fixed dimensions in Info |
| Run invariant | Validated bindings, selected defaults/overrides, current symbolic bindings | Recompute for each Run; do not put in descriptors |
| Fully dynamic | Current tensor pointers/data, overridable shape contents, context, destinations, free storage, live/workspace state, requested outputs | Original Run machinery |

Tensor identity is not a reuse key. A different immutable Tensor with the same
validated shape and type can use the same descriptor. CNN/numerical preparation
does not depend on payload. Reshape depends on target contents and therefore
requires non-input initializer/Constant provenance, including Identity aliases.
An input default is excluded even when its current default happens to be valid.

## Allocation Decomposition

The production reduction is exactly the original call-setup allocation scope:

| Family | Nodes | B/op removed | Allocations removed |
| --- | ---: | ---: | ---: |
| Conv | 2 | 1,696 | 10 |
| Pool | 2 | 1,536 | 6 |
| Add and ReLU metadata | 5 | 2,784 | 13 |
| MatMul metadata | 1 | 624 | 4 |
| Reshape metadata | 2 | 2,432 | 14 |
| Total | 12 | **9,072** | **47** |

This also explains the historical **51** exactly. Once constant Reshape is
retained, its seven setup allocations / 1,216 B are already gone. Of the
remaining production metadata scope, **40 allocations / 7,856 B** disappear.
The diagnostic closure additionally avoids **eleven argument-slice allocations /
144 B**. Thus **40+11=51** and **7,856+144=8,000 B**. With both Reshapes executing,
the analogous diagnostic removes 47 metadata plus 12 argument allocations,
9,072+160 B. Production deliberately keeps those 12 current-argument slices.

Constructor objects, CNN shape arrays, numerical shape snapshots/output plans,
Reshape accessor/resolution arrays and the extra ONNX MatMul plan account for
the removed objects. Backend plans are still present. This is allocation
attribution, not proof that allocation alone caused all latency recovery.

With five 2,000-call repetitions, GC-on medians are **67.890 → 64.193 µs**;
GC-off medians are **67.248 → 63.780 µs**. The saving survives disabled GC.
GC-on control includes an 82.504-µs sample; these separate diagnostics do not
replace the final comparison or quantify a clean GC tax. No arena is introduced.

## Shape Snapshot Analysis

The pinned backend Shape method calls Validate, allocates a dimension slice and
copies it. Rank, DataType and ByteSize also validate, but return no slices.
Backend numerical/CNN kernels use private shapes; there is no public immutable
shape-view accessor to use here. No new public accessor is introduced.

The metadata constructors request **18 backend Shape snapshots per MNIST Run**:
six CNN arguments, ten numerical arguments and two Reshape data shapes. CNN
also copies those six argument shapes into its temporary shape collection.
Reshape target Rank/Len checks and target Int64Values are separate operations.
Argument-wide snapshot probing includes twenty tensors because it also requests
the two shape-tensor shapes; it measures 512 B / 20 escaping snapshots.

Validation and destination selection still access current shapes. The compiler
can keep some copying accessors on the stack: declaration-check probing allocates
zero bytes, so a source-level Shape call must not automatically be counted as a
heap allocation. Destination guards do not expose or mutate private shapes.
A diagnostic-only copy of the pinned module counts **48 → 30 total Shape
calls per Run**: the 18 metadata calls disappear; 21 source/output validation
calls and nine free-destination shape comparisons remain. Counting is outside
performance runs and does not modify the module cache or production dependency.

## CNN Preparation

The descriptor retains the validated output shape/bytes and resolved Conv/Pool
configs, including auto padding, strides, dilation, groups, kernel and ceil/pad
behavior. Concrete current input/weight/bias types and shapes are established by
the existing bind/checkValue sequence before the consumer executes.

Conv2DInto and Pool Into still perform backend validation and acquire the current
disjoint destination. Their arithmetic is untouched. Prepared Conv kernels
measure **34.988 → 34.997 µs** in aggregate; Pool **3.547 → 3.539 µs**. Those
essentially unchanged scopes support setup attribution. The new coarse ORT Conv
scope is 36.460 µs; it is not an equivalent scope or evidence to reopen Conv.

## Broadcast/Add Preparation

Three Add output shapes and corresponding workspace decisions are reused.
Broadcast validity is proved for those fixed signatures; current input data
never enters the descriptor. Backend AddInto still selects its scalar/same-shape
path, computes strides/coordinates and validates destination/aliasing.

ONNX BroadcastShape probes cost **0.240 µs / 80 B / 3 allocations**. Prepared
backend Add calls cost **3.668 µs** together and still allocate **784 B / 22
objects** across the three nodes. An isolated backend probe of both CNN bias
Adds' two-input stride plans costs **0.422 µs / 368 B / 12 allocations**. This
work remains inside AddInto. The temporary diagnostic module exposes only the
existing private planner to the probe; it changes no loop. This is a remaining
scope, not a saving claimed by this implementation or a demonstrated guarded
backend API improvement.
There is no Add+ReLU fusion or arithmetic change.

## MatMul Preparation

Only ONNX output geometry, output byte count and the existing build-specific
workspace calculation move into preparation. ONNX MatMulShape costs **0.133 µs /
96 B / 2 allocations** in isolation; total ONNX call setup costs **0.219 µs /
624 B / 4 allocations**. These scopes overlap.

MatMulInto still builds its own private plan, performs tensor/destination/alias
validation, allocates transactional product staging and runs the identical
reduction order. The prepared adapter is **1.718 → 1.721 µs**, still **144 B / 3
allocations**. In a temporary diagnostic copy of the pinned module, **MatMulInto
measures 1.644 µs / 144 B / 3 allocations**, versus **1.480 µs / 0 B / 0
allocations** for its identical matrixProductInto arithmetic with fixed, disjoint,
prevalidated arguments and destination. The ~0.164-µs difference includes
planning, validation and transactional staging; it is not arithmetic recovery.
MatMulShape's separate 0.133-µs scope overlaps that difference and must not be
added to it. The stripped arithmetic helper is not production or a complete-contract
alternative. Both paths' outputs are checked bitwise outside timing. An earlier
untrimmed probe gives 1.656 / 1.522 µs and 0.433 µs for the stride scope; both
sets are archived separately.

Gemm stays on the general path. Its bias and optional metal copying workspace
are not generalized into this pass merely to expand operator coverage.

## Shape Operator Preparation

Both MNIST Reshape descriptors have fixed data shapes and immutable target
tensors. They retain resolved shape, dtype, byte count, target axes and workspace
request. **Both data copies still execute every Run.** The activation tensor is
always current; the constant weight transformation is not folded or retained.

Runtime target tensors, graph-input defaults and deferred data shapes keep the
original constructor/planner. Metadata preparation of the constant-weight
Reshape removes its planning, not its 10,240-B destination and computation.
The separate retained-computation diagnostic remains only a Pass 13 candidate.

## Guard Design

The guard is an existing invariant, not a new hash or tensor-identity cache:
**every source is checked by bind, and every produced value by checkValue,
before consumers can see it**. Those checks enforce the same concrete Info from
which the descriptor was built. Inputs, outputs, declarations and symbols keep
their original validation. Tensor shapes are immutable through the public API.
Consequently a successful Run cannot reach an eligible consumer with a different
input signature. Unsupported or deferred metadata has a nil descriptor.

| Guard approach | Measurement / decision |
| --- | --- |
| Direct runtime dtype/dimension comparison via Shape | Strict prototype 63.852 → 65.063 µs: **1.211 µs** extra, no additional allocations in that optimized diagnostic; copies/access checks still cost time |
| Compact shape hash | Not implemented; requires shape traversal and new state while existing validation already supplies the proof |
| Tensor generation/version | Not implemented; tensor identity changes across Runs and immutable public tensors have no mutable shape generation to track |
| Fixed-model proof plus general fallback | Selected; existing validation plus indexed descriptor selection; lookup traversal probe **0.076 µs** across twelve steps, including its loop |

The strict fixed-template direction is **3.881 µs**. An added snapshot guard
would reduce that diagnostic saving to **2.670 µs**. Production uses the existing
validation proof and recovers **2.8545 µs**. These separate scopes do not justify
calling their difference an exact guard tax. Production keeps 160 B / 12 current
argument allocations that the template diagnostic avoids.

## Dynamic Fallback

Unknown/symbolic dimensions, deferred shapes, mutable target contents, unsupported
families and insufficient optional load budget use the original dynamic calls.
There is no first-Run learning, per-session mutation, signature map or cache growth.
Changing inputs never replaces a prepared descriptor.

The supported graph-input contract permits dynamic leading dimensions;
runtime-changing spatial dimensions and ranks are exercised through runtime
Reshape nodes, as in the existing M3 suite. This pass does not broaden or narrow
that contract. Fixed-shape declaration violations remain errors rather than
being accepted by a fallback. Different *valid* dynamic signatures use the
general path, including alternating broadcast and product/CNN geometry.

## Prototype Results

The stricter A/B/C/D/E series keeps the original shape cancellation/reservation
protocol and both Reshape computations. Five repetitions; all use current
arguments and bitwise-check the fixed-model result.

| Prototype | µs | Saving µs | B/op saved | Allocs saved | Nodes |
| --- | ---: | ---: | ---: | ---: | --- |
| Control | 67.733 | — | — | — | All dynamic calls |
| A: CNN | 65.855 | 1.878 | 3,280 | 20 | 0, 3, 4, 7 |
| B: numerical excluding MatMul | 66.612 | 1.121 | 2,848 | 18 | 1, 2, 5, 6, 11 |
| C: shape | 66.588 | 1.145 | 2,464 | 16 | 8, 9 |
| D: MatMul | 67.360 | 0.373 | 640 | 5 | 10 |
| E: combined | 63.852 | 3.881 | 9,232 | 59 | All twelve |

The diagnostic's argument-slice savings exceed the production reduction, as
explained above. These are templates over a retained real-intermediate fixture,
not independently deployable dynamic caches. Production representation retention
by the same families is recorded separately; diagnostic closure/fixture storage
must not be mistaken for model-owned retention.

## Composition Analysis

| Family in combination order | Isolated saving µs | Combined incremental saving µs |
| --- | ---: | ---: |
| CNN | 1.878 | 1.878 |
| Numerical | 1.121 | 0.653 |
| Shape | 1.145 | 0.917 |
| MatMul | 0.373 | 0.433 |
| Combined | Isolated sum 4.517 | **3.881 total** |

Combined times are 65.855, 65.202, 64.285 and 63.852 µs respectively.
The isolated sum overstates the combined measurement by **0.636 µs**.
Shared setup, allocation/GC placement, branch/closure overhead and timing noise
prevent arithmetic addition. MatMul's slightly larger incremental result also
illustrates that these are measurements, not a perfectly additive partition.

## Architecture Decision

Selected: a small generic holder for **operator-specific immutable descriptors**,
stored beside the existing execution steps. The holder reuses the current CNN,
numerical and shape adapters as local invocation copies with new operands.
Planning helpers are shared between ordinary construction and preparation.

CNN-only preparation leaves substantial numerical/Reshape work. A complete
fixed-shape executor would duplicate validation, lifetime and budget machinery.
A shape-signature cache would introduce invalidation and concurrency costs
without helping this statically provable case. The selected boundary captures
the demonstrated families while leaving Gemm, normalization, other shape
operators and all deferred calls alone. No general compiler or backend API is added.

## Initialization / Persistent Memory

Preparation occurs during execution-plan construction, after compilation and
lifetime planning. Immutable target provenance is computed once in graph order;
the temporary pointer table is discarded. This avoids quadratic Identity-chain
walks. Cancellation is checked throughout preparation.

| Scope, five repetitions | Before µs | After µs | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| LoadBytes | 375.690 | 377.403 | 324,819 → 334,867 | 2,485 → 2,558 |
| Graph compilation | 183.678 | 184.630 | 173,016 → 173,016 | 991 → 991 |
| Execution-plan construction | 7.083 | 11.337 | 22,488 → 32,536 | 104 → 177 |
| Call preparation alone | — | **3.847** | — → 10,016 | — → 73 |
| Session creation/close | 0.0261 | 0.0262 | 24 → 24 | 1 → 1 |

Full-load timing includes much larger unchanged work and noise. Its 1.713-µs
median delta is not a substitute for the isolated 3.847-µs preparation cost.
The isolated ratio is **3.847 / 2.8545 = 1.35 Runs**, hence approximately **two
Runs** to amortize preparation. Graph compilation is unchanged in source and
allocation. Session creation adds no preparation.

Requested retained Go storage is **7,904 B**, including the plan's new fields,
pointer table, twelve descriptor holders, twelve call objects, twelve output
shape arrays and two target arrays: **39 heap objects**, excluding allocator
rounding/bookkeeping. Existing Step backing arrays and tensor payload are shared,
not duplicated. Each production descriptor has nil inputs and no context or destination.

Family retention is: CNN **2,912 B**, numerical excluding MatMul **2,864 B**,
Reshape **1,536 B**, MatMul **560 B**, plus **32 B** in plan fields. The table
pointer slots are attributed to the families. These are the production holder's
requested bytes, not the diagnostic fixture/closure heap size.

The conservative **12,800-B logical preparation reservation** is charged to
MaxLoadBytes before publishing descriptors. Preparation is optional when that
remaining budget cannot fit it; models still load at their former required
minimum and use dynamic calls. MaxLiveBytes/MaxWorkspaceBytes continue to bound
tensor payload and workspace, not Go object overhead. No extra tensor payload
is resident; existing **24,008 B** resident payload and **101,328 B** peak charged
live payload are unchanged. Within-Run reuse remains two destination reuses.

An isolated GC cleanup probe proves the plan remains alive after Model.Close
while a session survives, then becomes collectible after the last Session.Close.
Multiple sessions share the same immutable descriptors; Close clears the existing
plan references. No destination or input can be retained through a descriptor.

## Correctness

All **19 recorded validation checks** pass: default, cgo-disabled, purego, race
and purego+race full suites; vet; module verification/tidy diff; Linux amd64 and
arm64, Windows amd64, Darwin arm64 and Linux amd64 purego builds; comparison
policy tests; whitespace; four bounded fuzz targets (protobuf, tensor decode,
graph references, shape/attribute planning). Cross-builds prove compilation,
not performance on those architectures. No tolerance is relaxed.

All **45 comparison reports / 855 case executions / 85,500 timed calls per
runtime** pass with unchanged fingerprints and bitwise-identical Go outputs,
including the Pass 11 audit, initial baseline and both production measurement
sets. Maximum Go/ORT absolute error stays **1.9073486328125e-5**.

New focused tests compare eligible calls with preparation disabled in an
independently loaded control. They cover CNN families, Add/Sub/Mul/Div,
activations/Softmax, vector/batched/empty MatMul; initializer, Constant and
Identity-provenance Reshape targets; changing activation data; exact target
workspace acceptance/rejection; symbolic/shared and unknown dimensions;
changing batch/spatial geometry through runtime Reshape; defaults and overrides,
including malformed defaults replaced by valid targets; default → override →
default and override A → B; input shape errors; original load-limit fallback;
and concurrent sessions sharing a prepared default after Model.Close.

For these comparisons, output bits, error diagnostics and **context-check counts
match exactly**. Every successful control's cancellation boundary is exercised
on the prepared path. Existing suites also cover deadlines, exact live/workspace
bounds, malformed geometry/broadcast/product/Reshape errors and recovery,
original symbolic declarations refined by inference, output subsets/order,
ownership after later Runs/closure, ErrBusy/Close races and model/session limits.
Dynamic CNN geometry is tested with changed weights/bias/pads and spatial sizes.

Preparation errors do not become load errors: a failed descriptor attempt leaves
the original dynamic handler. An input default is never inspected as immutable
content. Only cancellation from preparation itself can stop model loading.
Load/compile validation already enforced by the original runtime stays in place.

## Candidate Ranking

This ranking uses demonstrated absolute opportunity, confidence and complexity,
without inventing numeric confidence scores or adding overlapping ceilings.

| Rank | Remaining candidate | Evidence after preparation | Confidence / complexity |
| --- | --- | --- | --- |
| 1 | Constant-only weight Reshape | About 1 µs incremental in the retained-computation diagnostic; 10,240-B copy/destination still present | High invariance confidence; moderate quota/error/lifetime work; +10,240-B payload if retained |
| 2 | Preserving ReLU investigation | 7.409 µs Go versus 2.000 µs coarse ORT; no new complete-contract recovery demonstrated | Large raw indication, low transfer confidence; backend investigation required |
| 3 | Remaining backend metadata | CNN bias-Add stride probe is 0.422 µs; ONNX selected setup is already prepared | Measured work, conditional recovery; would require an appropriate backend preparation API and guards |
| 4 | MatMul backend planning/arithmetic | 1.721 µs complete prepared adapter; isolated backend 1.644 versus 1.480 µs arithmetic-only; 1.003 µs coarse ORT | Small positive indication; most remaining Go time is arithmetic, not removable planning |
| 5 | Add+ReLU fusion | Backend Add still 3.668 µs; old raw fusion indication is not a post-metadata whole-Run saving | Potentially useful; high contract/destination coupling; overlaps backend Add planning |
| 6 | Destination/execution | 1.207-µs walk/check/setup remains, plus allocation/retirement/publication | Required semantics dominate; no new ownership-preserving whole-Run ceiling isolated |
| 7 | Pool | 3.539 µs prepared versus 3.018 µs coarse ORT | ~0–0.4 µs plausible; weak scope agreement |
| 8 | Conv | 34.997 µs Go prepared versus 36.460 µs coarse ORT | C, still good enough; no positive excess demonstrated; arithmetic remains closed |

## Selected Optimization

**Immutable, plan-owned operator-call metadata guarded by existing current-value
validation.** Prepared fields are resolved Step metadata, CNN geometry/configs,
output dtype/shape/bytes, numerical workspace decisions, and immutable Reshape
target/resolution metadata. Runtime fields are fresh argument slices/tensors,
context, destination, budget/workspace state, symbol bindings and storage lifetime.

The fallback is the original constructor/planner. Ownership is local-call copies
of immutable descriptors with new operands; lifetime is the existing shared
execution plan. A precomputed output shape is trusted only after the same fixed
input contract has been checked. Backend tensor/destination checks still run.

Prototype expectation was several microseconds and roughly 47 metadata
allocations without the diagnostic's argument-slice removal. Actual final
recovery is **2.8545 µs / 47 allocations**, a strong result under the requested
2–3-µs guidance. Key risks were conditional shape invariance, target overrides,
changed cancellation/quota ordering, early errors and shared mutation. The proof,
unchanged reservations/checks, dynamic fallback and differential/race tests
address them. There is no constant-computation folding or backend arithmetic change.

## Results

**65.646 µs before → 62.7915 µs after → 44.833 µs ORT → 1.4006×.**
Saving: **2.8545 µs / 4.35%**. The before set's own ORT median is 44.625 µs,
giving 1.4711×; ORT's 0.208-µs movement is not a Go optimization.
The earlier production set gives 3.229 µs; the final set is the primary result.

Ordinary ramp: **67.345 → 63.601 µs**, saving **3.744 µs / 5.56%** in its own
workload. All medians, ranges, CV and p95 are above. Benchmark averages do not
provide per-call p95. The comparison does not establish a tail improvement.

## Guard Hit / Miss Results

Five repetitions of a small Add model compare original source with the final
implementation. Static inputs are eligible; dynamic/alternating valid shapes
miss preparation eligibility and execute the general path. There is no mutable
signature cache or repeated-signature learning.

| Mode | Original µs | Prepared implementation µs | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| Fixed signature, hit | 0.9330 | 0.7747 | 1,336 → 808 | 26 → 23 |
| Repeated valid dynamic signature, fallback | 0.9218 | 0.9356 | 1,264 → 1,264 | 26 → 26 |
| Alternating valid dynamic signatures | 0.9279 | 0.9360 | 1,280 → 1,280 | 26 → 26 |

The separately built old/new sets show ~9–14 ns dynamic differences, about
0.9–1.5%, rather than a pathological miss penalty. Within the new binary,
prepared-disabled dynamic control is 0.9347 µs versus 0.9356 µs, and alternating
control 0.9374 versus 0.9360 µs. These small differences include measurement
variance; the allocation result is exact. Static declarations with invalid
signatures still fail original validation; they are not valid dynamic misses.

## Allocation Results

**112,763 → 103,691 B/op; 140 → 93 allocs/op**, down **9,072 B / 47 allocations**.
Added persistent metadata is **7,904 requested Go bytes / 39 objects**, with
**0 B added tensor payload**. Load allocation rises **10,048 B / 73 objects**,
including temporary preparation work. Logical metadata reservation is 12,800 B.
Runtime quota peaks, publication copies and two within-Run reuses are unchanged.

## Profile Delta

Rate-one allocation profiles remove the repeated newCNNCall/newNumericalCall/
newShapeCall sites, ONNX ResolveReshapeMode arrays, target Int64Values staging
and ONNX-side MatMul shape plan. Backend Conv2DShape/Pool2DShape remain, as do
broadcast stride/coordinate allocations and backend MatMul staging/plans.
The twelve current argument-slice allocations remain in evaluation.execute.

Setup-only scope falls **4.616 → 1.207 µs**, **9,232/59 → 160/12 B/allocs**.
Binding stays about 0.95 µs; publication about 0.132 µs. Prepared stripped graph
stays about **52.2 µs**. Kernel times above do not show a mysterious numerical
speedup. The full-stage census has **67.405 → 63.910 µs** median totals and
unchanged memory peaks/reuses. Individual stage means are noisy: constant
Reshape's measured stage rises 1.190 → 2.148 µs despite less setup. It is not
evidence that its arithmetic changed or a basis for additive stage accounting.

| Complete census stage | Before µs | After µs | B/Run before → after | allocs/Run before → after |
| --- | ---: | ---: | ---: | ---: |
| Evaluation/bind | 0.983 | 1.044 | 448 → 448 | 4 → 4 |
| Convolution28 | 9.648 | 10.373 | 28,304 → 27,456 | 11 → 6 |
| Plus30 | 4.657 | 3.085 | 28,216 → 27,640 | 15 → 12 |
| ReLU32 | 5.673 | 5.300 | 552 → 8 | 3 → 1 |
| Pooling66 | 4.910 | 3.113 | 7,480 → 6,712 | 9 → 6 |
| Convolution110 | 29.465 | 28.139 | 14,608 → 13,760 | 11 → 6 |
| Plus112 | 2.752 | 3.264 | 14,520 → 13,944 | 15 → 12 |
| ReLU114 | 3.000 | 2.725 | 552 → 8 | 3 → 1 |
| Pooling160 | 1.581 | 1.385 | 1,976 → 1,208 | 9 → 6 |
| Activation Reshape | 0.876 | 0.461 | 2,400 → 1,184 | 13 → 6 |
| Weight Reshape | 1.190 | 2.148 | 11,616 → 10,400 | 13 → 6 |
| MatMul | 2.210 | 1.982 | 960 → 336 | 12 → 8 |
| Plus214 | 0.682 | 0.487 | 1,088 → 544 | 16 → 13 |
| Publication | 0.132 | 0.132 | 208 → 208 | 6 → 6 |

Times are medians of five instrumented per-stage means; the median of their
complete totals need not equal the sum of stage medians. Allocation census is
separate from timing. This instrumented total is not the fixed-policy result.

CPU profiles retain the earlier Darwin sampling artifact: the candidate has
84.16% of samples in runtime.kevent. Run-focused samples show Conv, preserving
ReLU, Add and Pool, but are too sparse for precise setup attribution. Allocation
profiles, matched setup probes and whole-Run comparisons carry the quantitative
claims. Required work now dominates the non-kernel path: argument gathering,
current-value checks, destination allocation/reuse, budget/lifetime machinery
and publication. No unmeasured portion is assigned to GC or ORT setup.

## Post-Optimization Excess Ledger

Before and after scopes overlap and differ from ORT's integer-µs profiled full
nodes. The table is not an additive accounting identity. Unknown ORT runtime
costs are explicitly unknown, not zero.

| Region | Before Go µs | After Go µs | Coarse ORT µs | Recoverable excess / confidence |
| --- | ---: | ---: | ---: | --- |
| Conv, prepared | 34.988 | 34.997 | 36.460 | None demonstrated; closed C |
| Preserving ReLU, prepared | 7.594 | 7.409 | 2.000 | 5.409 raw indication; no new accepted production recovery |
| Pool, prepared | 3.547 | 3.539 | 3.018 | ~0–0.4 plausible; low confidence |
| Add, prepared | 3.559 | 3.668 | 3.004 | Generic backend planning/fusion investigation; net unknown |
| MatMul, prepared | 1.718 | 1.721 | 1.003 | ~0.3–0.7 plausible; contract-preserving transfer unproved |
| Both Reshapes, full census | 2.066 | 2.609 | 0.975 | Instrumented means noisy; constant computation ~1 µs diagnostic |
| Binding/evaluation | 0.983 | ~0.95–1.04 | Unknown | Constant cloning already removed in Pass 11 |
| Walk/check/call setup | 4.540 old probe | 1.207 matched prepared probe | Unknown | Main selected metadata work completed; residual is not all removable |
| Publication | 0.132 | 0.132 | Unknown | Required owned outputs; no removal demonstrated |
| Destinations/lifetime/runtime | Mixed into full stages | Mixed into full stages | Unknown | Unresolved, overlaps rows above |

Complete final gap: **62.7915 − 44.833 = 17.9585 µs**. The control's own gap is
21.021 µs. Only **2.8545 µs** of the 3.0625-µs gap change is Go recovery;
reference movement supplies 0.208 µs. Most remaining gap attribution is still
unresolved. Raw ReLU/MatMul differences are neither exact excess nor proven
whole-Run savings.

## Updated 1.25× Roadmap

At measured ORT 44.833 µs, **1.25× = 56.04125 µs**. Current Go 62.7915 µs
requires **6.75025 µs** further saving. At the historical 44.750-µs reference,
the comparable target is 55.9375 µs and remaining saving 6.854 µs.

The separate constant-Reshape diagnostic contributes about 1 µs in its own
scope, leaving roughly 5.7 µs. Conditional additions such as 1–3 µs preserving
ReLU, 0.3–0.7 MatMul, 0–0.5 additional fusion and 0–0.4 Pool would total only
about 1.3–4.6 µs beyond that. Even their optimistic transfer does not establish
the target. Fusion's old setup benefit is not counted after this pass.
Further destination/execution or kernel evidence is needed; Pass 11 cloning and
the accepted metadata saving are already in the baseline and are not added again.
The 0.422-µs stride probe overlaps Add/fusion work and is an alternative lead,
not another additive recovery line.

## Updated 1.10× Roadmap

**1.10× = 49.3163 µs**, requiring **13.4752 µs** more saving. The identified
constant computation and conditional ReLU/MatMul/fusion/Pool portfolio is far
short of that. Even the ~52.2-µs stripped numerical graph exceeds this target
before general Run semantics. Static/model-invariant cleanup alone is insufficient
on this workload and architecture.

Approaching 1.10× would require evidence for faster necessary work: preserving
activation, product planning/staging or arithmetic, complete-contract fusion,
destination/executor changes, architecture-specific implementation, or an
unidentified source. Current measurements neither require reopening Conv nor
justify promising SIMD or any other mechanism will close the gap.

## Cost of Generality

Three distinct ramp scopes remain useful:

| Scope | µs | Included semantics |
| --- | ---: | --- |
| Final general production Run | 63.601 | Current bindings, overrides, validation/symbols, budgets, context, destination handling, retirement, owned output |
| Strict fixed-model metadata diagnostic | 63.852 | Same principal Run operations, fixed prepared templates; current operands, both Reshape copies; different harness/allocation placement |
| Stripped prepared numerical graph | ~52.213 | Preallocated disjoint intermediates; excludes general binding/destination/lifetime/publication work |

The ~11.4-µs production/stripped difference is a broad scope comparison, not a
deployable saving or a cost assigned solely to dynamic shapes. A fixed model
now avoids repeated ONNX geometry construction, while still paying for current
input validation, override selection, symbol/declaration machinery, cancellation,
finite quotas, destinations, retirement and owned publication. Their separate
irreducible costs and ORT counterparts are not isolated. Dynamic fallback is
cheap in the small-model probe; generality is not synonymous with waste.

The older ~61.9-µs fixed diagnostic also retained constant computation, whereas
this pass's strict metadata-only diagnostic still runs both Reshapes. Its smaller
scope cannot be silently used as the new production target.

## Pass 13 Target

**Investigate guarded preparation of the constant-only weight Reshape**, selected
from the post-metadata profile and composition measurements. Its computation and
10,240-B destination remain after the metadata work is removed. The inherited
combined diagnostic changes **63.880 → 62.844 µs**, about **1.036 µs**, when that
computation is retained; this is an approximate direction, not a quota/error
equivalence proof. Do not carry over the pre-metadata 1–2-µs estimate unadjusted.

Pass 13 should first reproduce the incremental whole-Run saving with the exact
current workspace/reservation, error timing, pinning and output-ownership
contracts, price the +10,240-B resident payload and initialization, and verify
overrides and surviving/shared sessions. If that bounded optimization is not
worth its memory/complexity, the next evidence request is a standalone preserving
ReLU investigation. Conv arithmetic remains closed, and no metadata/fusion
benefit from Pass 12 may be counted again.
