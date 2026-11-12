# Performance Optimization Pass 11

2026-11-11, Apple M3. The production change reuses decoded constants as read-only
sources. **69.084 → 65.813 µs Go → 44.750 µs ORT → 1.471×**, saving
**3.271 µs / 4.73%** in the alternating fixed-policy comparison.
**138,608 → 112,763 B/op; 172 → 140 allocs/op.** Added persistent payload: **0 B**.
Constant-only Reshape and static call preparation remain measured diagnostics.

## Current Production Baseline

Started at ONNX `4d4b36d5cf710a36d3137932fd48020b005cffa5`, the merge of PR #31.
The real dependency is neuralnetwork **v0.10.7**, commit
`1f033b7d13788661a9002462151a997f30b620b4`, module sum
`h1:LPdDwSZz5MkiYGL1UAiMVZmpITvu5xVX4PHKoaf4BEw=`. There are no replacements.
The old audit uses v0.10.6 (`b0f1aa2f6716e32b3fae610528ec00ebd33ab19e`) and
frozen pre-integration numerical adapter/benchmark sources in temporary overlays.

Environment: Go 1.27.2, darwin/arm64, macOS 26.5.2 (25F84), Apple M3,
GOARM64=v8.0. Timing uses GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off, default
optimizations, no build tags/gcflags; comparison builds use `-buildvcs=false`.
The host's default CGO_ENABLED is 1 and GOMAXPROCS is unset; those defaults are
not the timing configuration. ORT 1.19.2, CPUExecutionProvider, ORT_SEQUENTIAL,
one intra-op and one inter-op thread, ORT_DISABLE_ALL, default CPU arena and
memory pattern. Reference Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2, protobuf 4.25.3.

The unchanged comparison contains upstream, zero, one and sixteen seed-0 uniform
random inputs: 19 cases, ten warmups and 100 timed calls per case; atol=1e-4,
rtol=1e-5. Owned outputs, input construction, timing boundary, execution order,
conversion/upstream/top-1 checks and ORT policy are unchanged. Loading, startup,
serialization and builds are excluded from inference timing. Model, input,
driver, runner and policy fingerprints match the archived v0.10.7 integration
across every report. Ordinary ramp remains its separate index%256 / 255 workload.
Ramp time is never divided by fixed-comparison ORT time.

Seven initial repetitions reproduce **68.146 / 44.541 = 1.530×**. Ramp median
is **69.972 µs**, range 69.164–72.628, CV 1.62%; 138,608 B/op and 172 allocations.
The later alternating control is slightly slower; both sets are retained.

| Seven-run set / engine | Median µs | Run-median range µs | SD µs / CV | Median p95 µs |
| --- | ---: | --- | --- | ---: |
| baseline / go | 68.1460 | 63.270–69.376 | 2.129 / 3.14% | 114.258 |
| baseline / ort | 44.5410 | 41.917–44.833 | 1.217 / 2.77% | 46.294 |

CV is sample SD / mean across repetitions. Ordinary medians summarize seven
benchmark ns/op averages; comparison medians summarize seven per-report medians
of 1,900 calls. p95 is the median of each report's within-run p95; ordinary
benchmarks do not provide per-call p95. No samples were
trimmed. Early lower ORT/Go observations and later high ramp observations remain.

Evidence and replay are retained locally under `tools/compare_runtime/pass11/`
(ignored and excluded from the PR).
`data/summary.json` contains all statistics; raw reports, census, CPU and rate-one
allocation profiles are retained. `prepare.py` reconstructs temporary overlays;
`measure.py`, `supplement.py`, `checks.py` and `summarize.py` reproduce the work.
Builds finish before timing, and measurements run sequentially. The preexisting
`.gitignore` excludes tools/; it is preserved byte for byte. This report, the
production changes, regression tests and preparation benchmark are outside that
ignored directory. Measurements and local checks were collected before publication;
subsequent CI results are recorded on the PR. No dependency release is needed.

## Pass 10 Prediction Audit

Fresh prepared real-intermediate measurements compare the actual v0.10.6
backend plus ONNX repair against released v0.10.7 plus its production adapter:

| Region | v0.10.6 µs | v0.10.7 µs | Saving µs | B/op before → after | Allocs before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReLU1 | 10.577 | 4.918 | 5.659 | 27,264 → 0 | 1 → 0 |
| ReLU2 | 5.365 | 2.458 | 2.907 | 13,568 → 0 | 1 → 0 |
| Total | 15.942 | 7.376 | **8.566** | **40,832 → 0** | **2 → 0** |

Whole ramp: 78.864 → 69.972 µs, saving **8.892 µs / 11.28%**. Fixed comparison:
76.4995 → 68.146 µs, saving **8.3535 µs / 10.92%**, ratios 1.7337× → 1.5300×.
Old ramp range/CV: 78.413–79.152 / 0.35%; old Go comparison range/CV:
74.729–77.292 / 1.09%. Old ORT median/range/CV/p95:
44.125 / 41.958–44.750 / 2.29% / 45.583 µs. Old Go p95 is 137.221 µs.

The **8–11 µs prediction transfers**: the fresh production saving is 8.354 µs,
0.213 µs below the prepared saving. The historical integration saved 7.687 µs
in its comparison and 8.887 µs on ramp. Those are separate observations with
different variance; this pass does not overwrite them. The earlier simplified
prototype omitted complete contracts and used a different loop. Released
bit-preserving logic, polling, validation and exceptional-path restoration
explain why its more optimistic result is not the production prediction.

Finite execution removes two snapshots and two independent NaN scans across
the graph. Per tensor, backend traversal + snapshot copy + scan becomes one
traversal: two complete tensor operations and three nominal full-array memory
streams disappear. Across both tensors S=37,632 B: 5S → 2S, eliminating
112,896 B of logical traffic, excluding allocator zeroing and metadata.

Contracts remain: finite success returns the supplied disjoint destination;
NaN success returns an independent correct clone and restores the supplied
private destination's legacy NaN-to-zero state. NaN signs/payloads/signaling
bits and positive-zero results are preserved. Cancellation/deadline errors
return nil output and restore the complete legacy destination before returning;
restoration is synchronous. Nonempty polling remains 3+ceil(n/1024) in the
backend, with the existing adapter checks; empty execution retains its legacy
path. Invalid destinations leave data unchanged. Existing exhaustive bit,
destination, cancellation and exact-budget tests all pass again.

## Fresh Operator Profile

Prepared calls use real ramp intermediates and disjoint destinations. They
exclude binding, call construction, destination allocation, retirement and
publication. Percentages use the corresponding ordinary ramp median, not ORT
comparison time. Independently timed costs are approximate and non-additive.

| Prepared region | Before µs | After µs | % of new 67.304-µs ramp | After B/op / allocs |
| --- | ---: | ---: | ---: | --- |
| Conv1 | 7.764 | 7.681 | 11.41% | 32 / 1 |
| Conv2 | 27.185 | 27.160 | 40.35% | 32 / 1 |
| Add1 | 1.991 | 1.959 | 2.91% | 216 / 7 |
| Add2 | 1.277 | 1.315 | 1.95% | 216 / 7 |
| ReLU1 | 4.918 | 4.906 | 7.29% | 0 / 0 |
| ReLU2 | 2.458 | 2.539 | 3.77% | 0 / 0 |
| Pool1 | 2.486 | 2.480 | 3.68% | 32 / 1 |
| Pool2 | 0.967 | 0.984 | 1.46% | 32 / 1 |
| MatMul | 1.708 | 1.705 | 2.53% | 144 / 3 |
| Final Add | 0.204 | 0.208 | 0.31% | 352 / 8 |

Post-change prepared family totals: Conv **34.841**, ReLU **7.445**, Pool
**3.464**, bias Adds **3.274**, final Add **0.208**, MatMul **1.705**, both
Reshape copies **0.273 µs**. Sum **51.209 µs**. Prepared whole graph is
**52.434 → 52.498 µs**, 1,088 B / 31 allocations: numerical work is unchanged.
Ramp minus prepared is **17.538 → 14.806 µs**, a mixed difference containing
setup, ownership, allocation and measurement-scope effects.

Five full-stage censuses per version time 3,000 Runs and count allocations
separately over 100 Runs. These include setup, allocation, checks and retirement.

| Full stage | Before µs | After µs | B/run before → after | Allocs before → after |
| --- | ---: | ---: | ---: | ---: |
| bind | 3.274 | 0.981 | 26,304 → 448 | 36 → 4 |
| Convolution28 | 9.912 | 9.389 | 28,304 → 28,304 | 11 → 11 |
| Plus30 | 4.875 | 4.687 | 28,216 → 28,216 | 15 → 15 |
| ReLU32 | 5.680 | 5.708 | 552 → 552 | 3 → 3 |
| Pooling66 | 3.782 | 4.979 | 7,480 → 7,480 | 9 → 9 |
| Convolution110 | 29.163 | 28.894 | 14,608 → 14,608 | 11 → 11 |
| Plus112 | 2.705 | 2.668 | 14,520 → 14,520 | 15 → 15 |
| ReLU114 | 3.023 | 2.999 | 552 → 552 | 3 → 3 |
| Pooling160 | 1.587 | 1.594 | 1,976 → 1,976 | 9 → 9 |
| Times212_reshape0 | 0.842 | 0.863 | 2,400 → 2,400 | 13 → 13 |
| Times212_reshape1 | 2.431 | 1.174 | 11,616 → 11,616 | 13 → 13 |
| Times212 | 2.259 | 2.220 | 960 → 960 | 12 → 12 |
| Plus214 | 0.769 | 0.683 | 1,088 → 1,088 | 16 → 16 |
| publish | 0.141 | 0.131 | 208 → 208 | 6 → 6 |

Census total medians: **70.682 → 67.675 µs**, ranges 69.836–70.983 /
66.870–68.103, CVs 0.62% / 0.74%. Peak charged live payload stays **101,328 B**
and two buffers are reused per Run. Large movement between unchanged Pool1 and
constant-Reshape stage means is not kernel attribution; allocator/GC timing and
instrumentation can move cost between stages. Prepared kernels and controlled
whole-run ablations are the stronger evidence. All original observations remain.

## Excess-Latency Ledger

Initial complete-comparison gap: **68.146 − 44.541 = 23.605 µs**. ORT entries
below are medians of three profiled-node mean sets, 1,000 retained ramp calls
per set. They have integer-microsecond event resolution and include different
machinery from unprofiled prepared Go calls. Unknown ORT work is never zero.

| Region | Initial Go µs | ORT µs | Recoverable excess / status | Confidence |
| --- | ---: | ---: | --- | --- |
| Conv | 34.949 prepared | 36.108 | No demonstrated positive excess | Strongly inferred status; exact difference uncertain |
| ReLU | 7.376 prepared | 1.999 | +5.377 indicative; no new production saving demonstrated here | Measured costs, recovery unproved |
| Pool | 3.453 prepared | 3.006 | +0.447 indicative; perhaps 0–0.4 | Weakly inferred recovery |
| Add, all three | 3.472 prepared | 2.896 | +0.576 indicative; fusion loop ceiling ~1.0 | Weak net attribution; measured diagnostic |
| MatMul | 1.708 prepared | 1.000 | +0.708 indicative; realistic 0.3–0.7 | Medium, differing scopes |
| Reshape | 0.281 copies; full stages ~3.27 | 0.897 | Constant preparation removes ~1–2 of Go work | Measured ablation, ORT scope imperfect |
| Initializer handling | 2.727 clone-only | Unknown | Production recovery belongs to 3.271 binding/clone bundle | High for eliminated work, unknown net ORT split |
| Binding | 3.729 with evaluation | Unknown | Includes initializer work; cannot add these rows | Measured Go; net excess unknown |
| Execution/setup | 4.501 including 1.171 walk/check | Unknown | Static call preparation ceiling ~3.66 after constants/Reshape | Fixed-shape diagnostic, medium generalizability |
| Destination handling | 9.030 all-fresh ceiling | Unknown | No isolated deployable net saving proved | Measured approximate scope |
| Publication | 0.118 prepared | Unknown | Required owned output; no demonstrated removal | Measured Go; excess unknown |
| Runtime/other | Mixed full/prepared remainder | Unknown | Unresolved | Unknown |

ORT profiled families sum to about 45.9 µs, already above unprofiled complete
ORT medians. No artificial zero/negative ORT setup cost is used to close the
ledger. Largest Go share and largest recoverable excess are different questions.

## Per-Run Work Classification

| Class | Work and evidence | Treatment |
| --- | --- | --- |
| A: input-dependent numerical work | Conv, ReLU, Pool, bias/final Add, MatMul, activation Reshape; prepared family total ~51.0 µs excluding constant Reshape | Execute every inference; avoid treating this as removable setup |
| B: model-dependent, input-independent | Eight clones 2.727 µs; constant weight Reshape ~0.19 µs copy and ~1–2 µs complete removable stage; fixed-shape call construction ceiling ~3.66 µs | Clones removed; transformation/metadata preparation remain diagnostics |
| C: Run bookkeeping | New evaluation, name/input binding, symbols, budget reservations, destination acquisition/retirement, cancellation, output copies | Bind+evaluation now 0.942 µs; walk/check 1.164; publication 0.118; scopes overlap |
| D: genuinely dynamic | Provided tensors/overrides, symbolic dimensions, input-derived shapes, requested-output validation, context state, live/workspace failures | Preserve runtime paths and validation |

This workload has fixed graph-input geometry; the runtime also accepts dynamic
models. The broader fixed-shape experiment is not a license to cache metadata
for arbitrary dynamic shapes. Constant validation alone is **0.4495 µs** and
allocation-free; declarations can bind shared symbols, so it remains at Run.
Reference slots, attributes, axes/defaults and CNN attributes are already
compiled. CNN output geometry, numerical broadcast/MatMul geometry, tensor
shape/type validation and shape-input snapshots are repeated. Some geometry is
static for this model; for dynamic inputs it needs guards or the existing path.
There is no evidence that separately resolving constant names is valuable.

## Initializer Ownership Analysis

The path is wire data → bounded decoder-owned tensor → graph.Constant → shared
operator/execution-plan constant reference → evaluation slot → read-only kernel
input. Decode owns independent shape/payload storage; neither caller model
bytes nor external-data buffers back the published tensor. Graph/compiler/plan
snapshots share constants instead of copying numerical payloads.

| Initializer | Shape | Payload B | Consumer / reason for the old clone |
| --- | --- | ---: | --- |
| Parameter5 | [8,1,5,5] | 800 | Conv1 weights; private binding copy |
| Parameter6 | [8,1,1] | 32 | Add1 bias; private binding copy |
| Parameter87 | [16,8,5,5] | 12,800 | Conv2 weights; private binding copy |
| Parameter88 | [16,1,1] | 64 | Add2 bias; private binding copy |
| Pooling160_Output_0_reshape0_shape | [2] int64 | 16 | Activation Reshape geometry; private binding copy |
| Parameter193 | [16,4,4,10] | 10,240 | Weight Reshape source; private binding copy |
| Parameter193_reshape1_shape | [2] int64 | 16 | Weight Reshape geometry; private binding copy |
| Parameter194 | [1,10] | 40 | Final Add bias; private binding copy |
| Total | Eight immutable initializers | **24,008** | **48,016 B nominal clone read/write traffic per Run** |

All eight are absent from graph inputs, so this model permits no initializer
overrides. The old destination belonged to the evaluation; it protected graph
state by granting a private mutable tensor at the graph/runtime boundary.
No supported kernel needs to mutate its source. Into functions write explicit
disjoint destinations; Identity only aliases; transforms copy into destinations;
ReLU restoration writes its private destination. Backend tensor data/shape
accessors copy, and the public Tensor facade exposes no writable backend value.

The lifetime planner already pins every producer-less value and Constant
output, including their Identity alias owners. None can retire into the free
buffer. Thus cloning is defensive isolation whose cost is unnecessary under
the current read-only source and lifetime contracts. The new internal Borrow
method makes that contract explicit; it is a trusted internal API, not a
compile-time read-only tensor type. Tests enforce the consumer invariant.

Models share one immutable plan across sessions. Separate sessions can run
concurrently; a single session returns ErrBusy for overlapping Run or Close.
Model.Close stops creation but existing sessions retain the plan. Session.Close
drops its plan reference; storage is reclaimed when references disappear.
Published outputs own copies and neither alias constants nor retain the plan.
No model/session cache or additional persistent payload is introduced.

## Override Semantics

The existing IR-8 runtime distinguishes non-input initializers from graph-input
defaults through Value.Input. It checks supplied input names first; a supplied
value wins only for a declared graph input. Otherwise a Default is selected;
a required input without either fails. A non-input initializer name is rejected
as an unexpected Run input. Supported standard opsets remain 13 and 17.

Borrowing occurs only after this selection, so defaults can be retained safely
without treating them as permanently constant dataflow. An override uses the
caller's validated tensor and the normal reservations. Every Run gets fresh
slots/symbols. No cached downstream value exists to invalidate, and override →
default, default → override and different overrides cannot leak across Runs.

A future prepared Reshape must exclude graph inputs even if defaulted, or use
an explicit guarded runtime fallback for every affected dependency. Default
shape initializers may be malformed at execution yet overridden successfully;
that existing behavior and error timing remain covered by the suite.

## Constant-Retention Experiments

The archived counterfactual independently reproduces the direction:
**70.923 → 67.572 µs** with retained bound values and validation, saving
**3.351 µs**, versus Pass 10's 3.479. Adding constant Reshape preparation gives
**65.589 µs**, saving **5.334 µs**, versus the old 4.691 bundle ceiling.
This scaffold omits some Session.Run entry/name/context work and is not production.

Five subsequent repetitions alternate mode order. They retain name validation,
input checks, constant/output validation, budgets, numerical kernels, dynamic
destinations, retirement and owned publication:

| Mode | Median µs | Range µs | CV | B/op / allocs |
| --- | ---: | --- | ---: | --- |
| Control | 70.849 | 69.986–73.842 | 2.08% | 138,608 / 172 |
| CloneRemovalOnly | 67.374 | 67.281–68.641 | 0.85% | 112,763 / 140 |
| ResolvedClones | 70.691 | 69.823–71.053 | 0.65% | 138,608 / 172 |
| ResolvedRetained | 67.315 | 66.950–67.346 | 0.25% | 112,763 / 140 |
| RetainedReshape | 65.583 | 65.401–66.317 | 0.58% | 101,162 / 127 |
| StaticMetadata | 61.922 | 61.517–62.647 | 0.69% | 93,162 / 76 |

Clone-only ceiling: **3.475 µs**. Resolving source indices while keeping clones
changes only 0.158 µs, within observed variation; combining resolution with
retention adds just 0.059 µs over retention. No useful independent reference
resolution opportunity is established. Validation remains **0.4495 µs / 0 B /
0 allocations**, rather than being counted as copying.

The first five contiguous samples and the later alternating samples are both
retained. These diagnostic binders are approximations, not byte-identical Run
entry paths; the production alternating comparison is the acceptance result.
Production saves **3.271 µs**, about 94% of the narrower 3.475 clone-only ceiling.
The remaining bundle ceiling was deliberately not implemented: its extra work
is constant-Reshape preparation, not missed memory-copy speed.

## Constant Reshape Analysis

Times212_reshape1 consumes immutable Parameter193 and a non-input int64 shape
initializer, copying **[16,4,4,10] → [256,10]**, **10,240 B**, every Run.
Both dependencies are immutable in this graph. MatMul expects an ordinary
contiguous row-major backend tensor, so a prepared owned tensor is compatible.

Prepared copy: **0.194 → 0.187 µs**, 16 B / one metadata allocation. The
instrumented complete stage moves from 2.431 to 1.174 µs with unchanged code;
use the controlled retention → retention+Reshape delta of **1.791 µs** as the
more useful ceiling, with a practical **~1–2 µs** range. Independent preparation
(construction, shape planning, kernel and value validation) costs **1.456 µs**,
range 1.422–1.605, **11,585 B / 12 allocations**. This is a preparation-component
benchmark, not a complete production cache initialization measurement.

An eager prepared copy is the simplest plausible follow-on: one owned result
per shared plan, pinned against reuse, charged before allocation and included
in persistent live/load accounting. It would add **10,240 B retained payload**
plus shape metadata, temporarily alongside its original tensor. Multiple sessions
should share that plan result instead of duplicating it. Close follows plan
lifetime. Overridable data or shape inputs must take the dynamic path. Existing
symbol checks must still run; malformed constants/errors must retain their
current observable behavior. Tight memory limits need an explicit fallback
policy before preparation is enabled.

Lazy initialization would add synchronization/failure/retry state without a
measured need. A general tensor view would require new backend shape/storage
identity and alias rules; no existing view API supplies that proof. This pass
introduces neither. Retention alone is Option A and is selected. Option B adds
this bounded prepared transform and accounting/fallback work for ~1–2 µs.
Option C, general constant-subgraph evaluation, adds operator coverage, error
ordering, dependency tracking and retained-memory policy with no demonstrated
extra workload benefit. It is not justified by these measurements.

ORT's disabled-optimization profile still contains this Reshape event for every
retained call (~0.880 µs mean-set median). Its cost is not assumed to be zero,
nor is ORT's payload-copy/view strategy inferred from that event alone.

## Prepared-Session Diagnostic

The fixed-shape diagnostic validates the graph's actual input and output
contracts on every iteration. It retains immutable bound values and the constant
Reshape result, and specializes CNN/numerical/shape call metadata derived from
this fixed model. It still resolves current tensor arguments, allocates/acquires
destinations, reserves workspace/live bytes, executes unchanged kernels, polls
context, checks values, retires values and publishes owned outputs.

**70.849 → 61.922 µs**, an **8.927-µs / 12.60%** broad ceiling. Decomposition
within the same experiment: retention **3.475**, Reshape **1.791**, then static
metadata **3.661 µs**. Do not add 8.927 to its components. It has **93,162 B /
76 allocations** versus control 138,608 / 172. Cached tensors are read-only in
this exact graph; its cached Reshape cannot be reused by a later compatible
destination. This does not establish generic graph cache safety or cover
arbitrary dynamic shapes/overrides/requested outputs.

The specialization proves an architectural direction, not a general optimizer.
Static call metadata alone is a more substantial follow-on than reference-slot
resolution. After the production retention gain, about **5.45 µs** of this
broader ceiling remains in the experiment. Applying that difference illustratively
to the fixed comparison projects ~60.36 µs; that is a projection across scopes,
not a measured fixed-policy result.

The empirical stripped graph is **52.498 µs**, or ~52.31 excluding the constant
Reshape copy. It omits ownership, construction, validation and budgeting, so it
is not an attainable Session.Run bound. The stronger complete-semantics
approximation is ~61.9 µs on ramp. Necessary input-dependent kernels, required
input/output ownership, dynamic validation, cancellation and memory lifetime
handling remain; their independent minimum costs are unresolved. This is an
empirical floor experiment, not a theoretical irreducible cost.

## Initialization / Memory Tradeoff

There is **no added preparation step**, model/session field, numerical copy,
retained tensor, metadata array or workspace in the selected optimization.
Existing decoded storage is the tensor that execution reads. Added persistent
payload and preparation allocations are **zero**. Initialization measurements
include the full cost rather than moving it outside accounting:

| Scope | Before median µs | After median µs | Before range µs | After range µs | B/op / allocs |
| --- | ---: | ---: | --- | --- | --- |
| LoadBytes + Close | 373.383 | 376.872 | 372.639–399.396 | 374.733–403.749 | ~325,171 / 2,487 → ~325,174 / 2,487 |
| NewSession + Close | 0.02612 | 0.02641 | 0.02611–0.02618 | 0.02615–0.02661 | 24 / 1 → 24 / 1 |

Load/session code is unchanged. The observed +3.489 µs load difference is within
the retained variation and is not an identified preparation operation. There
is no evidence of added initialization work. Added-work break-even is
0 / 3.271 = **0 runs**, so saving begins on the first Run. Even treating the
noisy positive load difference as a cost would give ~1.07 Runs; that is a
sensitivity calculation, not a demonstrated overhead.

Original resident initializer payload is **24,008 B**, already charged by
load/initializer accounting and newRunBudget. The selected change adds **0 B**
to that figure. To preserve exact tested memory-limit behavior, constantValue
keeps the historical per-use logical tensor reservation even though source and
bound value share physical payload. MaxTensorBytes, MaxLiveBytes, free-buffer
trimming, workspace release and failure ordering remain. This accounting is
conservative, not a hidden uncharged cache. Peak charged live payload remains
101,328 B. There is no new workspace/persistent-memory API distinction; Limits
continues to bound logical payload/reservations rather than Go object overhead
or outputs retained from earlier calls. Shared sessions duplicate no constants.

## Allocation Delta

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Whole Run B/op | 138,608 | 112,763 | **−25,845** |
| Whole Run allocs/op | 172 | 140 | **−32** |
| Bind + evaluation B/op | 26,288 | 448 | −25,840 |
| Bind + evaluation allocs/op | 36 | 4 | −32 |
| Added persistent payload | — | 0 | 0 |

Each nonempty clone constructs backend tensor, shape, identity and payload:
eight clones account for 32 allocations. Isolated clones measure 25,840 B;
whole-run accounting differs by 5 B, consistent with tiny-object packing
(the backend allocates one-byte tensor identities). The measured whole-run
number is reported rather than rounded to the clone microbenchmark.

Rate-one profiles show binding clones removed; output Clone remains required.
Remaining space is dominated by backend newTensor destinations (~90% of the
allocation profile), especially CNN and Add destinations and the constant weight
Reshape. Remaining object sites include tensor construction, per-step input
slices, cnnCall/numericalCall/shapeCall, backend Shape snapshots, broadcast
shape/stride plans and MatMul planning. The constant Reshape still contributes
~11,616 B / 13 allocations as a complete stage. No latency is inferred solely
from fewer allocations; the 3.271-µs saving is measured independently.

GC diagnostic, five alternating on/off pairs, fixed 2,000 calls: before
69.793 / 69.944 µs; after 66.724 / 66.607 µs. Ranges overlap (after on
66.230–67.351; off 66.111–66.884). This does not demonstrate meaningful
GC recovery. No arena, unsafe allocator, global scratch pool or sync.Pool
architecture is introduced.

## Fusion Reassessment

Current prepared bias Add1/2 are **1.959 / 1.315 µs** and ReLU1/2
**4.906 / 2.539 µs**: **10.719 µs** combined. The old 10–14 µs total-improvement
estimate included ReLU work already removed and is not reusable.

Finite real-Conv-output diagnostics preserve float32 Add rounding and compare
output bits against the graph. Raw loops include chunked context checks but
exclude generic validation, graph matching/fanout rules and exceptional adapter
restoration. They are an incremental loop experiment, not a production API.

| Scope | First pair µs | Second pair µs | Total µs |
| --- | ---: | ---: | ---: |
| Backend separate Add + preserving ReLU | 7.246 | 3.832 | 11.078 |
| Raw separate loops | 7.539 | 4.154 | 11.693 |
| Raw fused loops | 7.099 | 3.561 | 10.660 |

Raw incremental saving is **1.033 µs**; comparing raw fused to complete separate
backend calls gives only **0.418 µs**, with different validation scopes. Evidence
supports a modest ~0–1 µs kernel opportunity, not a demonstrated 1–3 µs complete
runtime win. Fusion might also remove call setup, which overlaps metadata
preparation and must not be counted twice.

For S=37,632 B across both epilogues, separate Add + ReLU uses four nominal
read/write streams; fusion uses two, saving 75,264 B logical traffic. Both
current ReLUs already reuse retired Conv destinations. Consequently, fusing
these pairs does **not automatically remove a full backing allocation per pair**:
both disjoint designs can need two buffers. Avoid counting eliminated
intermediate lifetime as 37,632 B of guaranteed allocation saving. Exact graph,
NaN, cancellation, alias and allocation benefits need a later complete prototype.

## Conv Status

**C — still good enough.** Initial prepared Conv is 7.764 + 27.185 =
34.949 µs; post-change is 7.681 + 27.160 = **34.841 µs**. Coarse ORT is
9.013 + 27.095 = **36.108 µs**. Different scopes prevent claiming Go is faster,
but show no demonstrated positive Conv excess or regression. Conv remains
~51.8% of the optimized ramp because surrounding work shrank. It is left alone.

MatMul remains the host tensor MatMulInto path, including backend matrix/plan
construction, ~**1.705 µs / 144 B / 3 allocations**, versus coarse ORT 1.000 µs.
A realistic independent saving is ~0.3–0.7 µs, not the entire kernel time.

## Candidate Ranking

Rank reflects recoverable absolute latency × confidence / complexity; these
are bounded judgment calls, not false precision in a numeric scoring system.
Projections use 65.813 µs / 44.750 µs after retention unless marked otherwise.
Unknown ORT cost and unknown preparation cost remain explicitly unknown.

| Rank / candidate | Current cost µs / comparable ORT | Diagnostic ceiling µs | Realistic further saving µs | Projected Go µs / ratio | Confidence / complexity |
| --- | --- | ---: | ---: | --- | --- |
| Completed: initializer retention | Old clones 2.727; ORT unknown | 3.475 | **3.271 realized** | **65.813 / 1.471×**, from 69.084 | High / small |
| 1: guarded static call metadata | 4.474 setup+walk; ORT unknown | 3.661 after constant bundle | 2–3.5 | 62.3–63.8 / 1.39–1.43× | Medium / moderate |
| 2: constant-only Reshape | ~1.17 complete post stage; ORT ~0.880 imperfect scope | ~1.8; observed ~1–2 | 1–1.8 | 64.0–64.8 / 1.43–1.45× | High invariant, medium transfer / moderate |
| Combined materialization | Retention completed; remaining Reshape above | 5.266 total from old runtime | Remaining 1–1.8 | Same as Reshape; do not add retention again | High/medium / moderate |
| 3: remaining preserving ReLU | 7.445 prepared / 1.999 coarse | No new complete-contract prototype | Unknown; 2–3.5 exploratory | 62.3–63.8 / 1.39–1.43× if proved | Low recovery confidence / backend work |
| 4: MatMul | 1.705 / 1.000 coarse | ≤0.705 indicative | 0.3–0.7 | 65.1–65.5 / 1.46× | Medium-low / moderate |
| 5: Add+ReLU fusion | 10.719 combined prepared / ~4 coarse | 1.033 raw-loop incremental | 0–1 kernel; setup overlap excluded | 64.8–65.8 / 1.45–1.47× | Low/medium / high |
| 6: destination/execution | 14.806 mixed full/prepared; ORT unknown | All-fresh construction ~9.2, not a saving | Unknown | No defensible projection | Low / potentially high |
| 7: Pool | 3.464 / 3.006 coarse | ~0.46 indicative | 0–0.4 | 65.4–65.8 / 1.46–1.47× | Low / backend work |
| Closed: Conv | 34.841 / 36.108 coarse | None demonstrated | 0 demonstrated | No change | Strong status / no justified implementation |
| Rejected independent reference resolution | Source walk within bind / ORT unknown | ~0.16, within variation | No proved saving | No change | Low / small but unnecessary |

| Candidate | B/op / alloc impact | Persistent / initialization cost | Repository / correctness and ownership risk |
| --- | --- | --- | --- |
| Selected retention | −25,845 / −32 whole Run | **0 added payload; 0 added preparation** | ONNX only; internal borrow contract, pinned sources and owned publication; regression-tested |
| Constant Reshape / combined follow-on | Diagnostic additional ~11.6 KB / 13 fewer allocations | +10,240 B payload plus shape; component preparation 1.456 µs / 11,585 B / 12 allocations; complete integration cost unknown | ONNX; quota, lifetime pinning, symbols, override dependency and error/fallback rules need design |
| Static metadata | Diagnostic additional ~8 KB / 51 fewer allocations after constant bundle | Retained call/shape descriptors and full initialization cost not measured | ONNX, some backend planning remains; dynamic guards/fallback and shared descriptor immutability required |
| ReLU tuning | Finite prepared path already 0 B / 0 allocations | Expected no payload cache; exact design/cost unknown | neuralnetwork plus adapter audit; NaN bits, cancellation and destination semantics are constraints |
| MatMul | Current 144 B / 3 allocations; removable portion unknown | No measured new cache; unknown | neuralnetwork; numeric ordering and zero-contraction/destination semantics |
| Fusion | 216 B / 7 allocations per prepared Add; exact graph delta unproved | No retained payload required by raw loop; graph preparation cost unknown | Both repositories; fanout, alias, NaN/rounding, cancellation and lifetime rules; high risk/complexity |
| Destination/execution | Potentially reduces allocations; no proved safe delta | Reuse can retain extra storage; cost unknown | ONNX/backend; output ownership, quotas, complete initialization and reuse proof |
| Pool | Current 64 B / 2 prepared allocations across both | No measured new persistent cost | neuralnetwork; small gain, edge/NaN rules |
| Conv | Current 64 B / 2 prepared allocations across both | No proposed initialization or memory change | neuralnetwork; leave closed without new excess evidence |
| Reference resolution | No measured allocation delta | Additional index metadata/cost unmeasured | ONNX; rejected as an independent optimization |

Destination components, all from separate controlled diagnostics:

| Component | µs | B/op / allocs | Interpretation |
| --- | ---: | --- | --- |
| All twelve fresh tensor destinations | 9.196 | 142,892 / 48 | Overstates actual Run, which reuses two buffers |
| Twelve backing allocations | 8.461 | 141,408 / 12 | Includes safe allocation/zeroing; not pure allocation overhead |
| Zero twelve reused backing slices | 1.476 | 0 / 0 | Variable (1.175–1.646), no allocation; overlaps prior row |
| Twelve shape snapshots | 0.309 | 320 / 12 | Copying only |
| Empty rank-four tensor | 0.054 | 129 / 3 | Object/shape/identity approximation with no payload |
| Twelve budget reserve/releases | 0.352 | 0 / 0 | Includes ByteSize validation |
| One compatible free-buffer acquisition | 0.050 | 0 / 0 | Shape/type checks and accounting |
| Lifetime schedule lookup | 0.013 | 0 / 0 | Lookup only; full retirement remains in stage census |

Tensor.New promises zero-filled storage and Go make supplies initialized backing.
Many Into kernels then overwrite it, but this is not proof that safe allocation
can omit zeroing. The existing two reuses already avoid new backing on those
paths. There is no clean, isolated safe destination optimization proved here;
no uninitialized memory or allocator redesign is introduced.

## Path Toward 1.25×

At the fresh ORT median, **1.25× = 55.938 µs**. The optimized runtime needs
**9.875 µs** more saving (initial baseline needed ~12.5 µs). Use post-retention
costs; never count the completed 3.271 µs again.

| Scenario | Non-overlapping further work | Saving µs | Projected Go µs / ratio | Remaining to 1.25× |
| --- | --- | ---: | --- | ---: |
| Conservative | ~1 constant Reshape + ~1.5 guarded metadata | 2.5 | 63.31 / 1.415× | 7.38 µs |
| Expected, conditional on implementation | ~1.5 Reshape + ~2.5 metadata + ~0.5 MatMul + ~0.5 fusion loop | 5.0 | 60.81 / 1.359× | 4.88 µs |
| Aggressive but unproved | Constant/metadata ceilings plus bounded ReLU tuning, small MatMul/Pool and only incremental fusion arithmetic | 8–11 | 54.81–57.81 / 1.225–1.292× | Could bracket target; not established |

The aggressive range requires new complete-contract kernel evidence, especially
ReLU. Metadata and fusion setup overlap: the table only counts fusion's raw
arithmetic component. Broad prepared-session, full/prepared and destination
ceilings are not added to these rows. Conservative describes a cautious transfer
assumption from measurements, not an already implemented guarantee.

## Path Toward 1.10×

**1.10× = 49.225 µs**, requiring **16.588 µs** more saving. The identified
constant-Reshape+metadata experiment supplies only ~5.45 µs of that direction,
and production may recover less. It leaves ~11.1 µs. Even an additional 3 µs
of ReLU tuning, 0.5 MatMul, 0.5 fusion and 0.3 Pool would leave ~6.8 µs without
an independently demonstrated source. These are conditional estimates.

The ~52.3-µs stripped ramp graph is already above 49.2 µs, before required
Run semantics. For this experiment, that target needs faster necessary work as
well as invariant-work removal. Its transfer to the fixed input distribution
still needs measurement. That could involve native SIMD/assembly for
preserving activation, carefully bounded fusion, or executor/destination changes.
Current evidence does not require reopening Conv. Scalar implementation,
validation/planning and native kernels need matched experiments before assigning
any specific gap to SIMD. ORT's runtime/binding/memory machinery remains partly
unisolated.

Needed evidence: a fixed-policy comparison of guarded metadata/constant
preparation; complete-contract ReLU/fusion prototypes; matched allocation/reuse
experiments without ownership bypass; ORT node/session scopes without coarse
clock/profiler distortion; and measurements on additional architectures. This
pass supplies no promise that 1.10× is realistic.

## Updated 1× Question

The new complete gap is **21.063 µs**. Demonstrated remaining Go work removable
in fixed-shape diagnostics is ~5.45 µs for Reshape+metadata, before production
fallback/guard costs. Plausible small kernel/fusion gains and a remaining
~5.45-µs raw ReLU difference are not demonstrated whole-run recovery.

Required Go/session work includes validated current inputs/overrides and shared
symbols, cancellation, finite memory reservations, disjoint destinations and
owned outputs. Their irreducible cost is unknown. ORT's native implementation
has an observed activation/MatMul advantage in imperfect node scopes; the
portion caused by SIMD, numerical contracts or runtime geometry is unresolved.
Measurement mismatch includes prepared versus profiled/full calls, input
workload differences, integer-µs events, GC/allocator placement and Darwin CPU
sampling artifacts. The remainder stays **unknown**.

The earlier optimistic 50–56 µs range is not strengthened by these data. A
concrete complete-semantics diagnostic is ~61.9 µs on ramp, with cross-scope
projection ~60.4 µs for the comparison. Additional conditional kernel gains
suggest roughly mid-50s to low-60s, with low confidence. The stripped graph at
~52.3 µs is not a production floor. Matching 44.75 µs is not demonstrated;
no unknown costs are assigned to a fictional native advantage to make it fit.

## Selected Optimization

**Option A: read-only reuse of the already retained decoded constants**, covering
non-input initializers, selected graph-input defaults and Constant nodes through
one existing materialization boundary. The invariant is that all such tensors
are immutable sources whose lifetime groups are pinned. Overrides are selected
before borrowing; downstream work remains dynamic.

This is the smallest correct abstraction supported by the measurements: one
internal Borrow accessor and replacement of evaluation's clone operation with
read-only binding. It adds no graph rewrite, transformed payload, cached
operator descriptor, session synchronization, invalidation state or memory
policy. Its expected ~3–3.5 µs production saving transfers to 3.271 µs. Choosing
A deliberately leaves the separately measured ~1–2 µs Reshape opportunity for a
bounded preparation design with its own memory/fallback review.

## Implementation

[internal/graph/constant.go](internal/graph/constant.go) documents immutable
ownership and adds Borrow. Its writable-consumer API Clone remains available.
[evaluation.go](evaluation.go) replaces cloneConstant with constantValue at the
two existing initializer/Constant binding points. It checks context before and
after borrowing, preserves the existing reservation/workspace sequence, then
runs the same value/declaration/symbol checks. All backend kernels are unchanged.

The existing lifetime planner and within-Run storage manager need no changes:
source and Constant Identity groups were already pinned. Publication still
clones every requested output separately. No writable destination is borrowed
from a constant. No cached result depends on an override; therefore invalidation
is unnecessary. Session/Model locking and closure behavior remain as before.
The backend version, go.mod/go.sum and public ONNX API are unchanged.

## Correctness

All **23 recorded checks** pass: default, cgo-disabled, purego, race and
purego+race full suites; vet; module verification/tidy diff; Linux amd64/arm64,
Windows amd64, Darwin arm64 and Linux amd64 purego builds; comparison policy;
formatting/whitespace; four bounded fuzz targets; three focused MNIST constant
immutability probes (default, purego without cgo, and race). Cross-builds prove compilation,
not other-architecture performance. The PR carries subsequent CI results.

All **28 reports / 532 comparison case executions / 53,200 timed calls per
runtime** pass. This includes the old audit and seven alternating production
before/after pairs. All Go output bits match each other and the archived
v0.10.7 integration. Maximum Go/ORT absolute error remains
**1.9073486328125e-5**. No tolerances, ownership or timing policy changed.

New evaluation tests execute initializer and Constant Identity aliases through
shape kernels with float NaN payload/signed-zero, exact extreme int64 values and
zero-size tensors. They verify pointer retention, absence from writable free
storage, independent publication, and unchanged source bits across repeated
execution even after internal writes to returned copies. Constant aliases are
not declared outputs in that plan, so source pinning is tested independently of
output pinning.

An additional optional MNIST test snapshots every initializer and verifies all
eight payloads bitwise after eight different finite/NaN-input Runs through the
actual CNN and MatMul pipeline. It skips cleanly when the optional model is absent.

New Session tests alternate different overrides/defaults across 100 Runs in
each of two concurrent sessions after Model.Close, compare bits, mutate only
published copies and verify retained default bits afterward. Existing tests
cover ordinary inference, different inputs and dynamic shapes, different-sized
overrides, missing/invalid inputs, default → override → default, symbols,
requested output ordering/subsets, outputs after closure, session overlap and
Close returning ErrBusy, model/session creation/closure races, exact live and
workspace bounds, MaxInitializerBytes/MaxLoadBytes, cancellation at every
boundary, malformed/overridden shape defaults and malformed constants.

## Results

**69.0835 µs before → 65.8125 µs after → 44.750 µs ORT → 1.4707×.**
The control's corresponding ORT median is 44.250 µs (1.5612×). The measured
Go saving is **3.271 µs / 4.73%**; the extra 0.500 µs difference in reference
medians must not be called a Go optimization. The initial 68.146-µs baseline is
a different repetition set, not the numerator of the paired improvement.

| Seven-run set / engine | Median µs | Run-median range µs | SD µs / CV | Median p95 µs |
| --- | ---: | --- | --- | ---: |
| final-baseline / go | 69.0835 | 67.646–69.708 | 0.802 / 1.17% | 121.877 |
| final-baseline / ort | 44.2500 | 41.416–44.625 | 1.102 / 2.51% | 45.627 |
| final-candidate / go | 65.8125 | 65.291–66.250 | 0.311 / 0.47% | 120.883 |
| final-candidate / ort | 44.7500 | 41.917–45.334 | 1.475 / 3.35% | 46.292 |

Ordinary ramp medians: **70.255 → 67.304 µs**, saving **2.951 µs / 4.20%**.
Ranges 68.143–74.969 / 66.577–72.083; CVs 2.93% / 3.34%. A first candidate ramp
sample is slower than its control and remains included. Median-of-set savings
are not paired-median savings or per-call tail predictions.

**138,608 → 112,763 B/op; 172 → 140 allocs/op; 0 B additional persistent
payload; 0 added preparation operations; first-Run break-even.** The full
initialization observations and their noise are reported above.

## Profile Delta

Binding+evaluation drops **3.729 → 0.942 µs**, 26,288 → 448 B and 36 → 4
allocations. Prepared graph is unchanged at ~52.5 µs, confirming attribution
to eliminated repeated materialization rather than faster numerical kernels.
Conv now occupies ~51.8% of ordinary ramp, ReLU ~11.1%, Pool ~5.1%, bias Adds
~4.9%, MatMul ~2.5%. The remaining metadata setup is ~4.474 µs including the
~1.164-µs walk/check scope. Constant weight Reshape remains a clear invariant
transform and allocation site.

Raw whole-process CPU samples place ~87% in runtime.kevent, as earlier passes
also observed. These sample percentages cannot quantify useful runtime overhead
or GC excess. The Session.Run-focused subtree still contains Conv, preserving
ReLU, broadcast Add, validation and construction; it is too sparse for precise
per-region latency. Raw and focused profiles are both retained. Separate process
accounting gives before 3.624 s wall / 3.524 user / 0.031 system (~98.1% of one
CPU); after 3.648 / 3.564 / 0.023 (~98.3%). This is neither per-kernel utilization
nor hardware-counter evidence. Operator/census and controlled ablations carry
the quantitative claims. Allocation profiles retain destinations and metadata;
binding clone stacks disappear while required output cloning remains.

## Post-Optimization Excess Ledger

Final complete gap: **65.8125 − 44.750 = 21.0625 µs**. The final control gap was
24.8335 µs; changing ORT reference medians accounts for 0.500 µs of that gap
reduction. Only **3.271 µs** is the measured Go saving.

| Region | Post Go µs | ORT µs | Remaining recoverable work | Confidence |
| --- | ---: | ---: | --- | --- |
| Conv | 34.841 prepared | 36.108 profiled | None demonstrated; closed C | Strong status, imperfect scopes |
| ReLU | 7.445 prepared | 1.999 profiled | +5.446 raw indication; no new accepted recovery | Low for reclaimable amount |
| Pool | 3.464 prepared | 3.006 profiled | ~0–0.4 plausible | Weak |
| Add | 3.482 prepared | 2.896 profiled | Raw fusion incremental ~1.0, complete runtime unproved | Weak/medium diagnostic |
| MatMul | 1.705 prepared | 1.000 profiled | ~0.3–0.7 plausible | Medium-low |
| Reshape | 0.273 copies; ~2.037 full stages | 0.897 profiled | Constant preparation ~1–2 Go µs | High invariant, medium transfer |
| Initializer handling | Clones eliminated; validation retained | Unknown | Completed; no repeated payload copy left | High |
| Binding | 0.942 incl. new evaluation | Unknown | Required binding/symbols/validation; no isolated new saving proved | Measured Go; excess unknown |
| Execution/setup | 4.474 incl. 1.164 walk/check | Unknown | Static metadata ~3.661 diagnostic ceiling | Medium, needs guarded production design |
| Destination handling | 9.196 all-fresh diagnostic | Unknown | Not actual overhead or a deployable saving | Unknown recovery |
| Publication | 0.118 prepared / 0.131 census | Unknown | Required owned copy; no removal proved | Unknown net excess |
| Runtime/other | Mixed 14.806 ramp full/prepared difference | Unknown | Unresolved; do not add to rows above | Unknown |

The controlled remaining constant-Reshape + metadata portfolio is **~5.45 µs**
of Go work in its own complete-execution approximation. Using it illustratively
against the 21.06-µs comparison gap leaves ~**15.61 µs unresolved by that
portfolio**; this is not exact cross-scope attribution or proof all 5.45 is net
ORT excess. ReLU/MatMul/fusion indicators may explain some of it, but are not
additional demonstrated whole-run gains. The honest explained *production*
recovery so far is 3.271 µs; the remaining exact ORT bookkeeping split is unknown.

## Pass 12 Target

**Guarded preparation of static operator-call metadata**, beginning with the
fixed-shape eligibility and fallback proof. Post-change setup still costs
4.474 µs; the controlled incremental ceiling after the constant bundle is
3.661 µs, larger than the remaining constant Reshape and measured fusion loop
opportunities. Reference resolution alone is negligible. Conv stays closed,
and GC/destination allocation do not justify an arena.

Pass 12 should first reproduce that metadata ceiling against this retained-
constant baseline, distinguish static output/broadcast/CNN/MatMul geometry from
required dynamic validation and backend planning, and test guards on symbolic
shapes and overrides. Constant Reshape preparation is the independent smaller
candidate (~1–2 µs, +10,240 B persistent payload) if metadata guards cost away
the demonstrated advantage. Remaining preserving-ReLU kernel/native disparity
is a backend investigation, not an assumed saving to bundle into the executor
change. This target is chosen from post-change evidence; no Pass 12 production
implementation or 1.25× promise is included here.
