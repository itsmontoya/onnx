# Performance Optimization Pass 17 — remaining runtime costs

Pass 17 is an **investigation with no production change**. Fresh released Go
inference measures **55.250 µs**, fresh ORT **41.875 µs**, ratio **1.3194×**,
with **92,826 B/op and 72 allocs/op**. The largest measured runtime mechanism is
private destination allocation: preallocating the ordinary destination topology
removes **91,449 B / 36 objects** and gives an **8.003-µs diagnostic difference**.
Its complete-contract production recovery is not established. Pass 18 should
investigate bounded destination reuse and its accounting/lifecycle obligations.

No NeuralNetwork production change is required for that investigation. No
backend optimization was selected, no backend production source was modified,
and no local backend replacement is used. Neither MatMul nor Pool passes the
gate for a new kernel optimization. Required output ownership is preserved by
the real runtime.

The [evidence/replay index](tools/compare_runtime/pass17/README.md),
[complete allocation/ownership ledger](tools/compare_runtime/pass17/ALLOCATION_LEDGER.md),
[production decision](tools/compare_runtime/pass17/DECISION.md), and
[machine-readable summary](tools/compare_runtime/pass17/data/summary.json)
contain the supporting details. All valid observations, including outliers,
remain in the lossless `data/evidence.json.gz` bundle. No component differences
below are summed into a production savings claim.

## 1. Fresh baseline

Measured November 13, 2026, before production changes; none were subsequently
made. Host: Apple M3, Mac15,13, eight cores (four performance/four efficiency),
24 GB; macOS 26.5.2 build 25F84, Darwin 25.5.0, arm64. Toolchain: Go 1.27.2,
`GOARM64=v8.0`, `GOMAXPROCS=1`, `CGO_ENABLED=0`, `GOWORK=off`, empty GOFLAGS.
Cache: `/private/tmp/onnx-pass6/go-cache`. This is the default host path, not
Metal execution.

ONNX base/measurement commit: `dbbf8bc75a7d8030ba7cd68b671c484bccb9d244`;
branch `optimization-pass-17`. Backend: published NeuralNetwork **v0.10.10**,
tag commit `2ffa5bd9fd2034382c49d14efc732ea4d026aa09`, module sum
`h1:CcoQQN8w9aGodz61FQGKSWtbnKzAHf/gtXbukwVUkj0=`. Direct protobuf dependency
v1.36.12; simd v1.3.0 and x/sys v0.45.0. The full module list is in the summary.

Reference environment: Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2, ORT 1.19.2,
Python protobuf 4.25.3. ORT uses CPUExecutionProvider, sequential execution,
one intra/inter-op thread, graph optimizations disabled, default enabled CPU
arena and memory patterns. Conversion/loading/building/serialization remain
outside timing.

Converted model SHA-256:
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
Unchanged driver SHA-256:
`9f32bdc790f2b323c53ac2d8fa1a8f777be8daa8d9f73aac3cd32f6f63b2bdae`.
Runner hashes:

| File | SHA-256 |
| --- | --- |
| `run_result.go` | `f1357afe5986ff398f752f9cdc7d7415eb32b0d13e7740c161ec63f1539bd64f` |
| `runner.go` | `9d1f4a83da63fd38ae8609ffc45b40d3bd81aa30b77d04085f7d4296a8832eec` |
| `tensor_data.go` | `e29a7e66c44068b91f73c200f5ad04aaf3230a47475966f3c6a3c4965cd200ac` |

Every report uses upstream, zeros, ones and sixteen seed-0 uniform-random cases,
nineteen sorted cases, ten warmups, 100 timed calls per case, independently
owned outputs, Go then ORT, atol=1e-4 and rtol=1e-5. Conversion, upstream output
and top-1 checks are unchanged. All seven reports pass all checks.

| Measurement | Median µs | Range µs | CV | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: | ---: |
| Fixed-policy Go | 55.250 | 50.958–57.979 | 3.63% | — | — |
| Fixed-policy ORT | 41.875 | 41.250–43.084 | 1.24% | — | — |
| Separate ordinary ramp | 54.788 | 53.403–59.349 | 3.21% | 92,826 | 72 |

Go report medians: 55.250, 55.3545, 57.979, 50.958, 55.958, 53.7705,
55.125 µs. ORT: 41.916, 43.084, 41.250, 41.667, 41.750, 42.000,
41.875 µs. The ratio is the ratio of those engine medians; ramp is not divided
by ORT. Model loading/session construction medians from the reports are
2,146.0/0.416 µs, separately measured with substantial load-time variance.

There is no production candidate: Go control 55.250 µs, Go candidate/difference/
percentage N/A; ORT control 41.875 µs, ORT candidate/movement N/A; control ratio
1.3194×, candidate ratio N/A. The previous released 54.625/42.0835-µs observation
is historical context, not a paired control or evidence of regression. Fresh ORT
movement is visible in the retained range; ratio movement cannot be attributed
entirely to Go.

## 2. Complete allocation ledger

The [detailed ledger](tools/compare_runtime/pass17/ALLOCATION_LEDGER.md) gives
23 allocation rows, sites/types, per-node payloads, owners/lifetimes, payload vs
metadata, immutability, reuse potential, and output/dynamic/concurrency/atomicity
obligations for every semantic group. Its disjoint major groups are:

| Purpose | B/Run | Objects/Run |
| --- | ---: | ---: |
| Nine private destinations: payloads, headers, shapes, identities | 91,449 | 36 |
| Run/evaluation/budget/value table/symbol map | 448 | 4 |
| Node input argument arrays | 160 | 12 |
| Conv/Pool/Reshape validation geometry | 144 | 5 |
| MatMul plan/result shape/staged output | 144 | 3 |
| Final equal-shape Add adapters | 288 | 6 |
| Independently owned output and publication | 193 | 6 |
| **Total ordinary accounting prediction** | **92,826** | **72** |

Internal payloads alone are 90,336 allocator B, representing 83,664 logical
bytes. Tensor headers are 864 B, shapes 240 B, storage identities nine amortized
tiny B. Returned output storage/metadata is separate.

Ordinary benchmarks independently agree with the prediction. Allocation-rate-one
profiles locate the sites, but their tiny-object handling produces about 62.6
Run-focused profile objects per invocation, **not** the ordinary 72. The profile
includes 1,001 invocations including warmup. Per-node MemStats barriers also
change tiny packing and catch a few background allocations; their raw results
remain in the bundle. No sampled profile was treated as an exact ordinary census,
and no unknown remainder was invented to force a balance. Per-slab tiny ownership
and physical retained/RSS overhead remain unattributed.

There are no successful-path heap allocations here for initializers, broadcast
strides, prepared operator geometry, operator output-index arrays, context errors,
or new cancellation infrastructure. Immediate-use Shape() copies do not escape
on this Go 1.27 path; multiplying the separately escaping accessor benchmark
by validation calls would double-count nonexistent heap allocations.

## 3. Copy/traffic ledger

These are logical copy sizes, not allocator-rounded sizes. Source inspection,
the current destination census and current backend scopes agree on the significant
payload copies. This model has **no multi-kilobyte per-Run initializer/output
clone** left.

| Source → destination | Bytes × frequency/Run | Why / ownership / lifetime | Newly allocated destination? | Atomicity / output obligation | Alias or remove/move possibility |
| --- | ---: | --- | --- | --- | --- |
| Pool07 activation → Reshape08 destination | 1,024 × 1 | Backend `ReshapeInto` promises a disjoint copied tensor; temporary invocation value | Yes | Validation before copy preserves destination on errors; not the final public ownership boundary | Read-only view could be a future design, but no current backend view contract; cannot move value-dependent work to load |
| Two final Add tensor operands → Matrix inputs | 40 × 2 | `FromSlice` owns copied values; invocation adapters | Yes, each | Constructor ownership contract; not required by public output ownership | Current API copies; removing adapters is a separate backend design, not selected |
| Add Matrix result → tensor destination | 40 × 1 | `ValuesInto` copies directly into existing output | Matrix result newly allocated; tensor destination allocated earlier | Adapter extraction into disjoint tensor | Cannot remove by deleting required final ownership; tiny separate opportunity |
| MatMul staged product → destination | 40 × 1 | Commit complete result after successful batches | Staging newly allocated; tensor destination allocated earlier | Current all-errors atomicity across batches/build paths | Host kernel-only diagnostic skips staging; general removal unproved |
| Final private tensor → caller-owned clone | 40 × 1 | Output survives future Run/session/model Close | Yes | Required current public ownership; final context checks remain | Borrowing internal storage is invalid; unique-storage transfer would need proof, not implemented |
| Tensor constructor shapes → owned shapes | 240 + 16 total | Nine private shape arrays plus returned clone shape | Yes | Tensor owns metadata independent of source shape slices | Reusable only with retained compatible tensors; public constructors still copy |
| Reshape target shape → temporary checked shape | 16 × 1 | Current backend validation helper returns owned shape | Yes | Payload remains unchanged on invalid arguments | Internal temporary could be redesigned; tiny unselected backend opportunity |
| MatMul batch prefix → result-shape prefix | 0 here | Empty batch prefix; two dimensions appended (16 B metadata stores) | Yes, shape array above | Checked planning | No batch stride payload or Matrix adaptation on this host path |
| Storage value table → node argument arrays | 160 B of pointer stores across 12 slices | Current tensor references in fixed input order, including optional slots | Yes | Run-dependent bindings and context checks | Topology can be prepared; contents cannot be shared mutably in the plan |
| Tensor shape → ONNX check/destination scratch | Up to 32 B per immediate-use shape read here | Validate actual dimensions/compatibility; stack/local metadata | No observed heap destination | Validation/symbol/limit obligations remain | Exact physical copy traffic not instrumented; do not treat this as zero work merely because B/op is zero |

The first five payload rows total **1,224 copied bytes/Run**. This excludes
ordinary arithmetic loads/stores and zero-initialization, and is not a memory-bus
counter. The nine newly constructed destinations logically start with 83,664
zero-valued payload bytes before successful kernels overwrite their results.
Actual zeroing/page/store traffic was not instrumented; allocator/kernel cache
effects cannot be inferred from that logical byte count.

The retained weight Reshape09 copies 10,240 bytes at preparation, outside Run,
and is already included in existing resident accounting. Binding borrows
immutable defaults; it does not clone them. Neither is a new Pass 17 saving.
Metadata literal construction, shape accessor copies, reference clearing and
budget counters also create traffic; their physical byte total remains unknown.

## 4. Runtime cost decomposition

Fresh node diagnostics use real ramp intermediates, prepared calls and disjoint
destinations, except the explicitly described current-topology traversal below.
They include backend validation/planning and ONNX wrapper context checks, not
just arithmetic. Destination allocation and Session dispatch are excluded.

| Node / prepared numerical call | µs | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Conv00 | 7.951 | 32 | 1 |
| Add01 | 1.887 | 0 | 0 |
| ReLU02 | 0.8606 | 0 | 0 |
| Pool03 | 2.455 | 32 | 1 |
| Conv04 | 27.192 | 32 | 1 |
| Add05 | 1.176 | 0 | 0 |
| ReLU06 | 0.4493 | 0 | 0 |
| Pool07 | 0.9657 | 32 | 1 |
| Activation Reshape08 | 0.08595 | 16 | 1 |
| Weight Reshape09 | Prepared retained value; numerical work omitted | 0 | 0 |
| MatMul10 | 1.662 | 144 | 3 |
| Add11 | 0.1832 | 288 | 6 |

The node medians are not forced to sum to a full traversal. Hot-node cache state,
ordinary reuse, call order, allocation and GC differ.

Controlled machinery evidence:

| Diagnostic scope | µs | B/op | allocs/op | Meaning / overlap |
| --- | ---: | ---: | ---: | --- |
| New evaluation + bind | 0.9501 | 448 | 4 | Value table, budget, symbols, input/default resolution/validation; excludes Session/name checks |
| Output checks across twelve values | 0.9203 | 0 | 0 | Actual rank/shape/type/declaration/limit checks; overlaps full traversal |
| Construct/fill twelve argument arrays | 0.2433 | 160 | 12 | Heap allocation plus pointer extraction and per-argument context checks |
| Synthetic same-shape destination selection across twelve requests | 0.8842 | 0 | 0 | Includes setup accessors and reservations; not ordinary allocation or an exact Run attribution |
| Empty-table retirement traversal | 0.0361 | 0 | 0 | Lower-bound loop/clearing traversal after values have retired; excludes actual release work |
| 100 background-context Err checks | 0.1031 | 0 | 0 | Microdiagnostic only; not a measured count of checks in Run |
| Complete publication | 0.1166 | 193 | 6 | Includes clone, wrapper, names and final context boundaries |
| Same full traversal with operator bodies disabled | 10.089 | 92,250 | 58 | Retains binding/dispatch/destination construction/value checks/retirement/publication; removes backend validation/planning as well as arithmetic/copies |
| Disabled operator bodies + preallocated destinations | 3.210 | 801 | 22 | Residual execution shell under that deliberately incomplete contract |

The matched disabled-operator control is 55.961 µs; operator-body removal changes
it by 45.872 µs. **This is not an arithmetic-only estimate:** backend planning,
validation, staging/adapters and their allocation disappear too. The 10.089 →
3.210-µs difference independently exposes about 6.879 µs associated with fresh
destinations in that altered workload. It is not additive to other differences.

Initializer/default resolution, budget updates, shape resolution, dispatch,
wrapping, context checks, lifetime accounting and cleanup overlap these scopes.
Their remaining individual latencies are not reliably isolated; no invented
residual is assigned to a function. Dynamic shape planning is not exercised by
the static MNIST timing; existing dynamic correctness tests remain necessary.

The separate CPU profile is retained but unsuitable for a percentage-based cost
ranking: 88.06% of samples appear in `runtime.kevent`/world restart while the
profiled benchmark measures 58.056 µs. It neither establishes a numerical kernel
regression nor an 88% recoverable runtime cost. Allocation-rate-one timing
(133.032 µs) is likewise excluded from ordinary latency estimates.

## 5. Updated stripped/preallocated diagnostic

The fresh current prepared traversal is **44.505 µs, 576 B, fourteen objects**,
range 44.174–44.998 µs. It retains the ordinary intra-Run reuse topology, uses
pre-resolved inputs and preallocated destinations, and skips the already retained
constant Reshape09. Repeated finite-ramp output bits match ordinary execution.

The older generic `BenchmarkMNISTPipeline/PreparedGraph` freshly measures
45.422 µs, 592 B, fifteen objects. Source inspection shows it still executes
the constant weight Reshape, adding a 10-KiB copy and one 16-B shape allocation.
It is retained as evidence but is not the current numerical floor. The new
diagnostic corrects that scope mismatch without changing production.

| Diagnostic variant | Matched control µs | Variant µs | Difference µs | Go change | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control A: ordinary Run | 56.943 | 56.943 | — | — | 92,826 | 72 |
| Skip backend output clone | 56.943 | 56.919 | -0.024 | -0.042% | 92,665 | 68 |
| Skip publication entirely | 56.943 | 56.950 | +0.007 | +0.012% | 92,633 | 66 |
| Evaluation-local argument array | 56.943 | 56.681 | -0.262 | -0.460% | 92,698 | 60 |
| Preallocated ordinary destinations | 56.943 | 48.940 | -8.003 | -14.054% | 1,377 | 36 |
| Destinations + evaluation argument array | 56.943 | 48.933 | -8.010 | -14.067% | 1,249 | 24 |
| Previous row + no publication | 56.943 | 48.839 | -8.104 | -14.232% | 1,056 | 18 |
| Control B: independent ordinary Run group | 55.597 | 55.597 | — | — | 92,826 | 72 |
| Destinations + fully pre-resolved arguments | 55.597 | 48.191 | -7.406 | -13.321% | 1,217 | 24 |
| Previous row + no publication | 55.597 | 48.030 | -7.567 | -13.610% | 1,024 | 18 |

Seven repetitions per row; order reverses on alternate repetitions. Every raw
sample remains available. Primary differences are differences of medians;
paired differences remain secondary (for example destination-only -7.845 µs,
argument-array -0.440 µs). Control A ranges 56.473–63.676 µs, CV 4.48%; its
outlier is retained. Destination-only ranges 48.450–49.603 µs, CV 0.69%.
Control B is a separate group and is never substituted for A to improve a claim.

These are Go-only diagnostics. For **every row**, ORT control/candidate,
ORT movement, control ratio and candidate ratio are N/A. They cannot establish
a released milestone or be normalized by fixed-policy ORT.

The broad outside-prepared difference is about 10.745 µs against fixed-policy
Go (different input scopes), or 11.092 µs against ordinary diagnostic control B
(same ramp fixture, separate benchmark scopes). These are broad indications,
not a sum of recoverable mechanisms. The retained-contract preallocated shell
still measures roughly 48 µs versus the 44.505-µs prepared traversal.
Overlaps/cache/GC prevent exact subtraction-based attribution.

Preallocated buffers are constructed outside timing and retained by the serial
diagnostic, including all nine unique ordinary buffers: 83,664 logical payload
bytes / 91,449 allocator bytes, plus fixture argument/reference/closure metadata.
They are **not free memory**. The setup executes an ordinary graph and builds
fixture closures; this initialization is excluded from inference timings, not
amortized into a production claim. No production retained-memory or initialization
change occurs. The diagnostic uses globals and ordinary logical reservations;
its retained-storage/concurrency contract is explicitly invalid for production.

## 6. Owned-output analysis

Required current ownership boundary: one `[1,10]` clone, **40 logical payload
bytes**, allocating a 48-B payload, 96-B backend header, 16-B shape and one
amortized identity byte: **161 B / four objects**, 0.07456 µs in isolation.
The ONNX wrapper adds 8 B/one object; `cloneOutput` plus reservation/context work
measures 0.09637 µs, 169 B/five objects. Named publication adds 24 B/one object;
the complete scope is **0.1166 µs, 193 B/six objects**. The standalone 40-byte
copy measures 0.00289 µs. These are nested, not additive scopes.

The final internal Add destination is another independently allocated 161-B,
four-object tensor before its result is cloned. Public output ownership does
not intrinsically demand two owned tensors. A proven unique-storage transfer
could theoretically remove that intermediate clone boundary, but general graph
outputs may alias inputs/defaults or one another and current publication charges
must remain sound. The no-clone ablation deliberately violates the general
ownership contract and yields no durable whole-Run gain. Do not return internal
activation storage, remove final copying generally, or spend Pass 18 on this
small ceiling before the larger destination question.

## 7. Execution-argument analysis

Exactly **twelve slices, 160 B, twelve objects** per Run: nodes 00/01/04/05/
08/09/10/11 each have two pointers (16 B); 02/03/06/07 each have one (8 B).
The weight-constant Reshape09 still constructs arguments despite retaining its
result. No operator output slice is constructed on this single-output path.

Lengths, input indices and optional-slot topology are model-invariant. Elements
are current Run-specific tensor pointers, including input overrides and actual
destinations. Static topology does not make mutable element arrays safe to
share in the execution plan. A caller-owned table could work only with a precise
nonretaining contract; none is added here.

An evaluation-embedded three-pointer array with generic heap fallback removes
all twelve argument objects in this fixture. The evaluation allocation grows
96 → 128 B, so the net saving is **128 B / twelve objects**, not 160 B. It is
heap-owned Run-local storage, **not proved stack allocation**. One-node local
stack storage across generic escaping call constructors was not established.
Preparing mutable values outside Run is valid only in the artificial fixed-input,
preallocated diagnostic; it does not model overrides or concurrent calls.

Construction alone measures 0.2433 µs. The full-Run array indication is 0.262 µs,
small against control movement. Added to preallocated destinations it changes
48.940 → 48.933 µs, showing overlap/no durable additional recovery. This remains
a secondary architectural/allocation opportunity, not shared mutable plan state.

## 8. Destination/staging analysis

The detailed per-node destination table is in the allocation ledger. All twelve
operators have prepared static output geometry; nine allocate fresh tensors,
two ReLUs reuse already retired same-shape tensors, and the weight Reshape09
returns its immutable retained tensor. `runStorage.destination` selects/charges;
the numerical/CNN/shape wrapper allocates `tensor.New` when no compatible tensor
is available. Private storage retires within Run; public clones survive.

Preallocated destinations retain the ordinary two ReLU reuses and the same
disjoint operator-input topology. Merely summing output sizes would count those
buffers twice. Retaining every ordinary unique destination across calls would
keep more memory alive than the current one-free-buffer strategy; the measured
91,449-B allocation removal is not a retained-memory reduction.

Large activation construction `[1,8,28,28]` separately measures 1.044 µs,
27,393 B/four objects. Clearing an existing payload is 0.2695 µs and copying it
0.3237 µs. These are cache-sensitive component scopes, not nine independent
production savings estimates. Capacity reuse must still validate actual shapes,
handle inputs/defaults/output aliases, preserve failure recovery and charge
retained/live/workspace bytes visibly.

MatMul stages ten floats (40 logical / 48 allocator B, one object), then copies
to the private destination. Its plan is 80 B and result shape 16 B. Host MatMul
uses direct slice kernels; **there is no Matrix adapter in this path**. No batch
stride allocations occur for the empty batch prefix. The Metal build is different
and can fail during device work. Current all-errors atomicity is not optional.

## 9. MatMul gate result

Current real ONNX prepared call: **1.662 µs, 144 B, three objects**.
Same-shape backend decomposition uses the explicit finite deterministic Pass 16
fixture against current v0.10.10; it is a separate workload, not a subtraction
from the ONNX number:

| Backend scope | µs | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Complete MatMulInto | 1.603 | 144 | 3 |
| Planning alone | 0.06073 | 96 | 2 |
| Batches including staging/allocation/final copy | 1.556 | 48 | 1 |
| Direct product arithmetic | 1.481 | 0 | 0 |
| Validated direct-product ceiling | 1.552 | 0 | 0 |
| Existing-destination ten-float copy | 0.00183 | 0 | 0 |

Most of the synthetic complete path is arithmetic. About 0.122 µs separates
complete call and unchecked product, while validated product recovers only
about 0.051 µs. Planning is about 0.061 µs; staging/block handling/allocation/
copy overlap the other scopes. Allocation alone is not reliably separated from
staging/plan construction. Destination allocation is outside all these scopes;
destination validation is retained by the validated ceiling. Host matrix
adaptation cost is zero by source path. These numbers must not be forced to
sum to 1.603 or 1.662 µs.

**Gate: no new meaningful demonstrated MatMul production mechanism selected.**
Kernel work is reserved for a later dedicated pass if needed; no staging buffer
is removed and no failure-atomicity contract is weakened.

## 10. Pool gate result

Real ONNX complete prepared calls: Pool03 2.455 µs; Pool07 0.9657 µs, each
32 B/one validation shape. Current complete backend fixture calls are
2.448/0.8634 µs. Current Pass 9 small-window kernels alone are 2.353/0.7593 µs;
validation is 0.09763/0.09710 µs, including shape helpers at
0.06531/0.06514 µs. The scopes overlap and use deterministic synthetic values
for backend gates; real intermediate call costs are retained separately.

The observed metadata mechanism is small and already understood. No new excess
in the optimized small-window arithmetic was demonstrated. **Gate: do not reopen
Pool production work.** No NaN, first-equal signed-zero or window behavior changes.

## 11. Ranked opportunities by demonstrated recovery

Ranked first by the size of a demonstrated mechanism, with production-contract
uncertainty explicit. Current cost alone does not determine priority.

| Candidate | Current cost | Demonstrated avoidable cost | B/op opportunity | alloc opportunity | Confidence | Complexity |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Bounded private destination reuse | Nine destinations; 56.943-µs complete diagnostic | 8.003 µs with destinations prepared; 6.879 µs in disabled-body control; complete-contract recovery unknown | Up to 91,449, traded for retained storage | Up to 36 | High mechanism; medium/low production transfer until accounting proved | High: retention, quotas, shapes, failure/Close |
| Run-local argument structure | 0.2433-µs construction | Twelve objects removed; 0.262-µs whole-Run indication, almost zero additional gain with destinations prepared | 128 net in embedded-array probe | 12 | High allocation; low durable latency | Low/medium; escape/fallback/optional slots |
| Other execution shell | 3.210-µs disabled-body/preallocated shell | No isolated production recovery proved | Unknown | Unknown | Low attribution | Depends on validation/dispatch proof |
| Pool geometry metadata | ~0.10-µs validation each | ~0.065-µs shape scope each; required checking still remains | At most 64 | 2 | High current cost; low net recovery | Backend-owned; not selected |
| MatMul planning/staging | 1.603-µs synthetic complete call | ~0.051-µs validated-product ceiling; full all-errors proof absent | At most 144 | 3 | High scope; low universal transfer | Backend/device/atomicity |
| Output clone boundary | 0.1166-µs total publication | No durable whole-Run recovery; tiny intermediate-ownership ceiling | At most 161 while keeping wrapper/list | 4 | High allocation/copy; low latency | Alias/quota/ownership proof |
| Activation Reshape08 | 0.08595-µs prepared call plus destination construction | 1,024-byte copy exists; alias-safe production removal unproved | 1,137 destination + 16 checked shape, only with a valid replacement | 5 ceiling | High existence; low transfer | Backend view/identity/lifetime design |
| Final Add adapters | 0.1832 µs | No new production mechanism tested in this pass | At most 288 | 6 | High current cost; no new recovery claim | Backend; explicitly deprioritized |

Upper bounds overlap; for example the Reshape destination is already in the
destination-reuse row. No backend prompt is justified by those small unselected
ceilings. The shell row is deliberately unranked within its own cost: there is
no demonstrated unnecessary work to assign it a recoverable value.

## 12. Production decision

**Make no production change and recommend Pass 18.** The
[ten-question decision record](tools/compare_runtime/pass17/DECISION.md)
explains the unnecessary-work hypothesis, timing/bytes/objects, semantic
obligations, missing smallest-correct-fix proof, expected recovery uncertainty
and owning repository.

Destination allocation is demonstrated; a contract-preserving retention policy
is not. The global diagnostic fixture cannot be a production mode. This pass
does not add pooling, an arena, unsafe access, unbounded caching, shared mutable
plan state, or uncharged retained memory. Smaller backend metadata mechanisms
do not justify crossing the repository boundary. No backend implementation
branch, commit, PR, or local modified production result exists.

## 13. Production result

No production candidate, API/dependency change, initialization change or retained
production memory change. Complete released Run remains **92,826 B / 72 objects**.
Component before/after and complete production transfer are N/A; the ablation
tables are diagnostic results only. Artifacts are documentation, replay scripts,
test-only overlays, evidence and the ignore-rule exception for their directory.

## 14. Correctness validation

All recorded commands exit successfully; detailed logs and environments are in
`data/validation.json` and the compressed evidence:

- Full ONNX suites: default CGO-enabled, CGO-disabled, purego, race, purego+race.
  These cover existing numerical/IEEE semantics, dynamic/symbolic shapes,
  initializer defaults/overrides, aliases, ownership/survival, cancellation and
  deadlines, failed-Run recovery, Busy/Close, quotas/accounting and concurrency.
- Finite-ramp bitwise repeated-traversal checks for ordinary, argument-array,
  preallocated/pre-resolved and publication-stripped diagnostics in all five
  configurations. These limited checks **do not validate the deliberately
  stripped variants as general production contracts**.
- Backend test-only MatMul/direct-product and Pool/kernel finite equivalence,
  input-preservation checks, CGO-disabled, purego, race and purego+race. Backend
  production/module-cache/sibling-checkout source remains unchanged.
- `go vet ./...`, module verification, `go mod tidy -diff`, formatting/diff checks.
- CGO-disabled default and purego cross-builds for darwin/arm64, darwin/amd64,
  linux/arm64, linux/amd64, windows/amd64 and linux/386. Compilation, not native
  execution on those target systems.
- Seven fresh fixed-policy default reports plus a separate purego report; all
  conversion/upstream/top-1/numerical checks pass with unchanged tolerances.
- Complete serialized native Metal ONNX and backend suites: zero failures,
  **zero skips**, 2,010 and 5,705 passing test/subtest records respectively.
  Five isolated native `Test_SequentialResidentLongRunningMixedStress`
  repetitions pass. Sandbox device-unavailable skips remain in the earlier logs.

Pass 16's one unrestricted Metal stress failure remains **unexplained**. These
new successes do not resolve it and do not attribute it to v0.10.10. No new
failure was silently discarded. No production lifetime/device code changed.

## 15. Updated excess-latency signals

ORT nodes were freshly profiled in three separate complete-graph runs of ten
warmups plus 1,000 calls, same disabled optimizations/one-thread CPU policy.
The table uses median of per-profile mean node durations. Individual durations
are integer microseconds, include profiling/dispatch overhead and have different
scopes from Go prepared calls. These are **coarse signals, not recoverable cost**.
Every profile, including the slower first profile, is retained.

| Node | Go prepared µs | ORT coarse µs | Go − ORT signal µs |
| --- | ---: | ---: | ---: |
| Conv00 | 7.951 | 9.043 | -1.092 |
| Add01 | 1.887 | 1.025 | +0.862 |
| ReLU02 | 0.8606 | 1.009 | -0.1484 |
| Pool03 | 2.455 | 1.030 | +1.425 |
| Conv04 | 27.192 | 27.367 | -0.175 |
| Add05 | 1.176 | 1.012 | +0.164 |
| ReLU06 | 0.4493 | 1.004 | -0.5547 |
| Pool07 | 0.9657 | 2.019 | -1.0533 |
| Activation Reshape08 | 0.08595 | 0.055 | +0.03095 |
| Constant weight Reshape09 | Retained; dispatch not separately isolated | 0.692 | Not comparable |
| MatMul10 | 1.662 | 1.007 | +0.655 |
| Add11 | 0.1832 | 0.901 | -0.7178 |

Whole fixed-policy Go − ORT is **13.375 µs**. The broad runtime-vs-prepared
indication is about **10.7–11.1 µs**, depending on the explicitly different
control scope. Destination ablation demonstrates 6.9–8.0 µs in its controlled
variants; other shell costs remain about 3.2 µs when both operator bodies and
fresh destination allocation are removed. These signals overlap and must not
be added, reconciled to 13.375 µs, or interpreted as pure excess over ORT.
Pool03's coarse positive signal does not override its current mechanism gate.

## 16. Updated 1.25× roadmap arithmetic

Fresh ORT **41.875 µs × 1.25 = 52.34375 µs**. Current Go **55.250 µs**;
nominal difference **2.90625 µs**. This is target arithmetic only. The destination
diagnostic exceeds that number but is not a released comparison or a demonstrated
complete-contract production recovery. No milestone is claimed.

## 17. Updated 1.10× roadmap arithmetic

Fresh ORT **41.875 µs × 1.10 = 46.06250 µs**. Current Go **55.250 µs**;
nominal difference **9.18750 µs**. That recovery has not been demonstrated under
the public runtime contract. The stripped/preallocated floor is not a production
forecast and is not normalized into a Go/ORT ratio.

## 18. Recommendation for Pass 18

Investigate **bounded private destination reuse owned by ONNX**, beginning with
an explicit retained-memory/lifetime/accounting design. Use the nine destination
allocations and 6.9–8.0-µs controlled signals as the mechanism to validate, not
as a promised result. Prove limits, dynamic-shape fallback, optional/defaulted
inputs, output ownership, failure recovery, cancellation, session isolation and
Close behavior before promoting a prototype. Measure initialization and retained
bytes/objects, then run unchanged fixed-policy control/candidate reports with
separate Go/ORT movement and complete correctness validation.

If the complete contract erases the benefit or requires an unjustified retention
policy, reject that candidate. Run-local argument structure is a smaller fallback
allocator objective. MatMul, Pool, Conv, ReLU and another tiny matrix wrapper are
not predetermined targets. Any later backend-owned production requirement must
stop at a standalone NeuralNetwork handoff and await a published release.
