# Performance optimization Pass 19: the post-retention critical path

Pass 19 selects one ONNX change: reuse a cleared, evaluation-local argument array
for synchronous calls with at most three inputs, with ordinary allocation for
larger arities. **No NeuralNetwork production change is required.**

The final seven-pair unchanged-policy comparison measures **51.2915 → 50.9170 µs**
for Go, a **0.3745-µs / 0.7301%** reduction. ORT separately moves
**44.3125 → 44.3750 µs / +0.1410%**. The ratio moves **1.15750× → 1.14743×**.
Warm allocation changes **1,409 → 1,265 B/op; 36 → 24 allocs/op**.
This is a small improvement, not a 1.10× result.

[Decision gate](tools/compare_runtime/pass19/DECISION.md),
[allocation ledger](tools/compare_runtime/pass19/ALLOCATION_LEDGER.md),
[replay instructions](tools/compare_runtime/pass19/README.md), and
[all summary inputs](tools/compare_runtime/pass19/data/summary.json) accompany
this report. The compressed evidence preserves every valid sample, raw reports,
logs, source overlays, fingerprints, allocation/CPU listings and generated code.
No pre-Pass-18 ranking was used to select this change.

## 1. Fresh merged baseline

Start: `optimization-pass-19`, merged ONNX
`2445b7920781d6cd91e6635749f6e023044a979b` (PR #39). Production change:
`ae7850b9f2d551908b54da4a4eb72591e88cdb2c`. NeuralNetwork is published `v0.10.10`,
commit `2ffa5bd9fd2034382c49d14efc732ea4d026aa09`, with no replacement.
Go 1.27.2, darwin/arm64; Apple M3, 8 cores (4 performance/4 efficiency), 24 GB;
macOS 26.5.2 build 25F84, Darwin 25.5.0. Measurement environment:
`GOARM64=v8.0`, `GOMAXPROCS=1`, `CGO_ENABLED=0`, `GOWORK=off`, empty `GOFLAGS`
except explicit diagnostic/control overlays. Compilation is outside timing.

Relevant modules: protobuf v1.36.12, simd v1.3.0, x/sys v0.45.0. Python 3.9.6,
NumPy 1.26.4, ONNX 1.16.2, ORT 1.19.2. ORT uses CPUExecutionProvider,
ORT_SEQUENTIAL, one intra/inter-op thread, ORT_DISABLE_ALL, default CPU arena
and memory-pattern behavior. Full environment/module publication data are archived.

| Fresh initial scope | Median µs | Range µs | CV |
| --- | ---: | --- | ---: |
| Fixed-policy Go | 51.3330 | 50.5840–52.0000 | 0.809% |
| Fixed-policy ORT | 41.9170 | 41.6670–44.4580 | 2.131% |
| Separate ordinary Go ramp | 48.6880 | 47.9550–51.3990 | 2.199% |

Initial ratio: **1.22463×**. All seven ramps have **1,409 B/op / 36 allocs/op**.
The later interleaved production controls have their own reference; they do not
replace these initial observations. ORT reference drift is material.

SHA-256 fingerprints:

| Artifact | SHA-256 |
| --- | --- |
| Converted IR8/opset13 model | `310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d` |
| Original pinned model | `5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd` |
| Comparison driver | `9f32bdc790f2b323c53ac2d8fa1a8f777be8daa8d9f73aac3cd32f6f63b2bdae` |
| `runner.go` | `9d1f4a83da63fd38ae8609ffc45b40d3bd81aa30b77d04085f7d4296a8832eec` |
| `run_result.go` | `f1357afe5986ff398f752f9cdc7d7415eb32b0d13e7740c161ec63f1539bd64f` |
| `tensor_data.go` | `e29a7e66c44068b91f73c200f5ad04aaf3230a47475966f3c6a3c4965cd200ac` |

The fixed policy remains nineteen sorted cases: upstream, zeros, ones and sixteen
seed-0 uniform-random inputs; ten warmups and 100 timed calls per case; Go then
ORT; independently owned outputs; atol=1e-4, rtol=1e-5; unchanged conversion,
upstream and top-1 checks. Loading, building and serialization are excluded.

## 2. Retention-state verification

Fresh merged measurements, seven observations per scope:

| State | Median µs | Range µs | CV | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: | ---: |
| Cold Run, unpopulated Session | 56.272 | 56.008–56.856 | 0.527% | 92,864 | 72 |
| Second Run, first reuse | 49.499 | 48.957–50.530 | 1.087% | 1,424 | 36 |
| Ordinary warm ramp | 48.688 | 47.955–51.399 | 2.199% | 1,409 | 36 |

Cold/second scopes exclude Session preparation and Close with timer barriers.
Those barriers change tiny-allocator packing: the isolated second-Run counter
is 15 B higher than ordinary ramp, with the same object count. Do not substitute
that number into the ordinary ledger.

Runs 1, 2, 3 and 1,000 show nine retained tensors / **83,664 logical bytes**,
zero workspace balance, two intra-Run reuses and final live accounting
**155,376 B**. No retained pointer changes in 1,000 Runs. The limit is 121,296
logical B; actual retained tensor allocator storage is approximately **91,449 B**
plus the 96-B pointer table and Session metadata, not an RSS measurement.

Balanced final accounting is 2×34,248 resident/default/prepared reservations +
83,664 retained + 3,136 input + 40 internal output + 40 published clone. Fresh
profiles show no warm private-destination payload construction or unexpected
fallback. The final candidate repeats the same retention/pointer/accounting test.

Candidate cold and second-Run observations are retained separately: **55.436 µs,
92,720 B / 60 objects**, and **50.200 µs, 1,280 B / 24 objects**. They were collected
later, not interleaved with the merged state experiment; the higher second-Run
time is not hidden or interpreted as a matched regression/improvement.

## 3. Fresh allocation ledger

The allocation-rate-one profile, 1,000-invocation MemStats census, ordinary
benchmarks, current source sizes and removal controls agree on the ordinary
**1,409 B / 36 objects**. The complete row-by-row semantic ledger is linked above.

| Current group | B/Run | Objects |
| --- | ---: | ---: |
| Evaluation, budget, value table, symbol map | 480 | 4 |
| Node argument arrays | 160 | 12 |
| Conv validation shapes | 64 | 2 |
| Pool validation shapes | 64 | 2 |
| Activation Reshape validation shape | 16 | 1 |
| MatMul plan, result shape and staging | 144 | 3 |
| Final Add matrix adapters | 288 | 6 |
| Owned output publication | 193 | 6 |
| **Total** | **1,409** | **36** |

This was measured anew, not obtained by deleting historical destination rows.
Profiles include cold population and tiny-object coalescing; their object totals
are not exact ordinary allocs/op. MemStats barriers also perturb tiny packing:
publication's barrier census is about 209 B, versus 193 B in ordinary components.
Both observations remain archived.

## 4. Fresh node decomposition

Prepared complete calls use captured actual arguments/destinations and current
retained/intra-Run topology. Operator validation, backend staging and numerical
context polling remain. Setup, dispatch, acquisition, value checks and publication
are excluded. ns/element uses output elements, not arithmetic operations.

| Node | µs/op | ns/output element | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| Conv00 | 7.7950 | 1.2428 | 32 | 1 |
| Add01 | 1.8900 | 0.3013 | 0 | 0 |
| ReLU02 | 0.8674 | 0.1383 | 0 | 0 |
| Pool03 | 2.4860 | 1.5855 | 32 | 1 |
| Conv04 | 27.3050 | 8.7070 | 32 | 1 |
| Add05 | 1.1760 | 0.3750 | 0 | 0 |
| ReLU06 | 0.4554 | 0.1452 | 0 | 0 |
| Pool07 | 0.9707 | 3.7918 | 32 | 1 |
| Reshape08 | 0.0857 | 0.3346 | 16 | 1 |
| MatMul10 | 1.6790 | 167.9000 | 144 | 3 |
| Add11 | 0.1838 | 18.3800 | 288 | 6 |

Node 09 is the actual immutable-reshape prepared dispatch, not a 10-KB copy:
**0.0305 µs, 0 B / 0 objects**. Its budget/context/retained-return path remains
in ordinary execution; the numerical traversal omits its nonexistent body.

## 5. Fresh prepared numerical floor

The current prepared traversal is **44.783 µs**, range 44.613–45.503 µs,
CV 0.653%, **576 B / 14 objects**. It runs eleven operator bodies in graph order
with real intermediate tensors and the ordinary alias topology, skipping only
the already-retained constant reshape. Repeated traversal matches captured
ordinary output bits.

It includes complete ONNX numerical/CNN/reshape calls, backend validation,
MatMul staging, Add adapters and backend polling. It excludes evaluation creation,
binding, construction of argument slices, destination lookup/reservations,
per-value checks, the outer traversal/context boundaries, retirement, publication
and cleanup. It uses pre-resolved closures and hot resident tensors.

The contemporaneous broad warm control is **49.304 µs**, a descriptive difference
of **4.521 µs**. Other ordinary controls differ; this subtraction is neither a
matched ablation nor a demonstrated recoverable budget. The shell is separately
3.764 µs. Do not force these scopes to sum.

## 6. Fresh execution-shell decomposition

The full retained traversal with operator bodies disabled measures **3.764 µs**,
range 3.726–3.995, CV 2.271%, **833 B / 22 objects**. It retains Session locking,
evaluation/binding, argument construction, actual prepared dispatch/acquisition,
accounting, value validation, retirement, owned publication and cleanup. It drops
arithmetic, operator/backend validation, operator allocations and backend polling.
Its outputs are not valid numerical results. Warm population is excluded.

Fresh separate components, with setup excluded:

| Scope | µs | B/op | Objects |
| --- | ---: | ---: | ---: |
| Evaluation creation | 0.1170 | 480 | 4 |
| Binding existing evaluation | 0.7864 | 0 | 0 |
| Construct/bind twelve argument slices | 0.2492 | 160 | 12 |
| Twelve produced-value checks | 0.9073 | 0 | 0 |
| Acquisition/retirement/accounting cycle approximation | 0.8468 | 0 | 0 |
| Complete owned publication | 0.1170 | 193 | 6 |
| 337 interface context checks | 0.3468 | 0 | 0 |

The acquisition cycle uses current static shapes and real lifetime retirement,
but zero per-operator workspace and no body, value validation or node-09 special
return. It is an attribution probe, not an exact shell decomposition. Whole-shell
measurement preserves those real paths. No allocation belongs to warm retained-slot
bookkeeping. Dispatch, cleanup and accounting are not independently established
recoverable quantities; they remain represented by the controlled whole shell.

## 7. Destination-retention overhead

Nine successful lookups using pre-extracted requested geometry take **0.3020 µs**;
the direct table traversal takes **0.0105 µs**. The approximately **0.2915-µs**
difference includes nil/bounds/type/shape compatibility work and the backend
metadata accessors' validation/copying. It excludes reservation and retirement.
Both allocate zero. Actual complete-Run compatibility bypass indicates 0.473 µs
in the noisy broad matrix, but removes a required guard and is not a valid fix.

The 0.8468-µs acquisition-cycle probe includes reservations, free-candidate
comparison, lookup, installation, reference clearing and retirement/release.
It is not added to lookup or shell costs. Warm `keep` insertion is dormant except
for rejected/no-slot cases; retained bytes are charged once at Run entry in
addition to invocation accounting. Payload reset/clear cost is **zero**: kernels
fully overwrite destinations; no new preclear was introduced.

The initial lookup probe took 0.5423 µs because it also computed requested dtype
and shape through validating accessors. Its result is archived, superseded for
attribution by the corrected probe, not discarded as an outlier.

## 8. Argument-array result

Fresh construction is 160 B / twelve objects, approximately 0.24–0.25 µs.
MNIST has maximum arity two; supported BatchNormalization has five, normalization
and optional Conv/Gemm have up to three, and Concat is variadic. Topology belongs
to the immutable plan; dynamic dimensions change tensors, not the storage policy.

The diagnostic inline-three overlay removes twelve objects and adds 16 B to the
evaluation allocation size class: **144 B net saved**. Broad complete-Run
indication: 49.304 → 49.024 µs. Focused ordinary control/overlay: 49.124 → 48.851 µs;
focused fixed-policy medians: 51.6455 → 51.1250 µs, with overlapping ranges and
approximately 1.94%/1.98% CV. These were selection signals, not the final headline.

Production uses one helper for execute and normalization. Each slice is cleared
before filling, preserving nil optional slots. Wider calls use the original fresh
allocation. Every evaluation owns its own array; no shared plan/Session cache,
mutable global, new output alias or cross-Run state is introduced. Synchronous
handlers finish before the next argument set is filled. Failed evaluations die
with their arrays; existing context checks keep their positions.

The actual production argument component is **0.1348 µs / 0 B / 0 objects**,
versus corrected control 0.2492 µs; the approximately **0.1144-µs** component
recovery excludes all other Run effects. Complete-runtime transfer is reported
separately below.

## 9. Validation/geometry result

| Fresh backend geometry | µs | B/op | Objects |
| --- | ---: | ---: | ---: |
| Conv00 shape helper | 0.0759 | 32 | 1 |
| Conv04 shape helper | 0.0764 | 32 | 1 |
| Pool03 shape helper | 0.0663 | 32 | 1 |
| Pool07 shape helper | 0.0664 | 32 | 1 |
| Reshape08 shape helper | 0.0316 | 16 | 1 |

All are static for MNIST, and ONNX prepared descriptors already contain output
geometry. Public Into operations independently validate actual tensor metadata,
configurations, aliases and storage before writing; no prepared backend interface
exists to consume that proof. Dynamic shapes, overridable targets and invalid
inputs still need runtime validation. Helper-owned result shape arrays look
structurally avoidable internally, but these measurements include required
geometry work and do not prove an equivalent complete-Run bypass.

Initial Conv geometry probes allocated an extra weight-shape literal per call
because the error formatting path makes it escape. Moving test inputs outside
timing yields the corrected 32-B result above. Both logs remain. No validation
or geometry production change was selected.

## 10. MatMul gate

Prepared ONNX MatMul is **1.679 µs / 144 B / 3 objects**. Separate backend complete
call: 1.690 µs; plan/result shape: **0.0622 µs / 96 B / 2 objects**;
batches with staging: 1.580 µs / 48 B / 1 object; direct product: **1.516 µs**;
output copy: 0.0019 µs. A validated-product diagnostic takes 1.588 µs.

Arithmetic dominates. Those overlapping scopes are not additive savings. The
finite host diagnostic does not prove all-errors atomicity without staging across
all batches/devices. No current production-valid complete-Run mechanism or kernel
rewrite is demonstrated. Keep MatMul closed in this pass.

## 11. Pool gate

Backend complete / validation / kernel scopes are **2.524 / 0.0981 / 2.428 µs**
for Pool03 and **0.8767 / 0.0990 / 0.7814 µs** for Pool07. Each validation allocates
32 B once. Prepared ONNX complete calls differ slightly because of wrapper,
working-set and measurement scope. The existing small-window kernels dominate;
validation is required. No new arithmetic mechanism or equivalent validation
removal is demonstrated. Coarse ORT signals do not reopen Pool.

## 12. Add adapter gate

Final Add prepared call: **0.1838 µs / 288 B / 6 objects**. Backend complete:
0.1685 µs; matrix adapter: **0.0913 µs**; validated direct arithmetic diagnostic:
0.0798 µs. The approximately 0.09-µs component ceiling is tiny, belongs in
NeuralNetwork, and lacks production-valid complete-Run evidence. Leave it alone.
The two channel-broadcast Adds allocate zero.

## 13. Publication result

Complete publication is **0.1170 µs / 193 B / 6 objects**. ONNX clone including
wrapper: **0.0968 µs / 169 B / 5 objects**; backend clone: **0.0727 µs / 161 B /
4 objects**; payload-only copy: **0.0029 µs**. Required independent storage and
metadata dominate the tiny copy. Names borrow immutable strings; no name-copy
allocation exists. Reservations and context boundaries remain in complete scopes.

The no-publication diagnostic indicates 0.462 µs in the noisy matrix, but discards
the entire output contract; it is not 0.462 µs of avoidable work. No unnecessary
surrounding mechanism is established. Outputs still survive later Runs and Close.

## 14. Context/cancellation result

A counting delegating context observes **337 Err calls per warm Run**, before and
after production. Removing only ONNX checks diagnostically leaves **15 backend
checks**; thus 322 observed checks belong to ONNX. A homogeneous 337-call interface
loop costs 0.3468 µs. The full ONNX-check-removal diagnostic indicates 0.388 µs,
with different compiler/control-flow effects.

Binding/declaration/axis checks, node boundaries, reservations, copying and
publication have observable cancellation/error order; ReLU additionally polls
inside chunks. The ablation weakens these semantics. No equivalent redundant
layer is proved, and no check or cancellation responsiveness changed.

## 15. Dispatch/escape/generated-code findings

Fresh CPU samples attribute 61.2% flat to the two Conv eight-filter arithmetic
routines. They also contain OS/GC work such as madvise/kevent; process sample
percentages are not removable runtime latency. There is no measured dispatch
kernel opportunity from these percentages.

ARM64 control code calls `runtime.makeslice` for each argument set. The new helper
inlines into both execute and normalize (compiler cost 18), branches on arity
three, clears pointer storage through `runtime.memclrHasPointers`, and reaches
`makeslice` only on the wider fallback. Heap allocation of the evaluation remains;
its size class grows 128 → 144 B. Dynamic constructors may escape transient call
objects; the warm prepared path does not allocate those constructors.

Metadata accessor Shape copies used immediately by validation/lookup stay off the
ordinary heap, but validation/copy CPU work remains. Interface context calls,
prepared indirect calls and ordinary bounds checks remain. No assembly, unsafe,
backend API or abstraction rewrite was selected.

## 16. Fresh ORT signal table

**Signals, not recoverable costs.** Values are the median of three repetition
means, with 1,000 profiled events per node after ten warmups per repetition.
ORT events have integer-microsecond resolution, profiling overhead and scheduling
outliers; Go numbers are separate unprofiled prepared calls. Do not sum columns,
node differences, or use this table as the complete Go/ORT gap.

| Node | Go prepared µs | ORT profiled signal µs |
| --- | ---: | ---: |
| Conv00 | 7.7950 | 10.914 |
| Add01 | 1.8900 | 1.321 |
| ReLU02 | 0.8674 | 1.119 |
| Pool03 | 2.4860 | 1.314 |
| Conv04 | 27.3050 | 32.977 |
| Add05 | 1.1760 | 1.252 |
| ReLU06 | 0.4554 | 1.034 |
| Pool07 | 0.9707 | 2.246 |
| Reshape08 | 0.0857 | 0.287 |
| Reshape09 | 0.0305 dispatch | 1.078 |
| MatMul10 | 1.6790 | 1.157 |
| Add11 | 0.1838 | 1.107 |

ORT may schedule immutable Reshape09 ahead of Conv00. Its ordering/event scope
is not ONNX's retained dispatch. These signals do not nominate an optimization.

## 17. Controlled ablation matrix

One common merged control per repetition; variant order reverses on alternating
repetitions. Seven observations each; all are retained. Differences below are
**control median minus variant median**, not medians of paired differences.
These are Go-only diagnostics: no ORT movement or Go/ORT ratio is assigned.

| Variant | Median µs | Range µs | CV | Difference µs | B/op | Objects |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| control | 49.3040 | 48.9460–63.0860 | 9.352% | 0.0000 | 1409 | 36 |
| arguments | 49.0240 | 48.8460–55.5230 | 4.547% | 0.2800 | 1265 | 24 |
| no-publication | 48.8420 | 48.7420–51.8120 | 2.099% | 0.4620 | 1216 | 30 |
| shell | 3.7640 | 3.7260–3.9950 | 2.271% | 45.5400 | 833 | 22 |
| direct-retention | 48.8310 | 48.5560–49.0540 | 0.299% | 0.4730 | 1409 | 36 |
| no-output-validation | 47.5110 | 47.3740–50.8760 | 2.449% | 1.7930 | 1409 | 36 |
| no-onnx-context | 48.9160 | 48.7930–51.6820 | 1.981% | 0.3880 | 1409 | 36 |

The control's 63.086-µs observation and argument variant's 55.523-µs observation
remain in their ranges/CVs. Contracts: arguments preserve the complete contract;
publication removes owned outputs; shell removes numerical work/validation;
direct-retention removes compatibility checks; no-output-validation actually
removes `checkValue` for **both binding and produced values**, including declarations,
symbols and limits; no-onnx-context removes ONNX cancellation boundaries. None of
the invalid variants was combined to manufacture a fast result.

The original no-context transformer failed compilation by matching an unrelated
receiver member. It was corrected before any timing; the exit and explanation
are archived. Corrected component diagnostics and their initial attribution
mistakes are retained separately. No valid timing observation was filtered.

## 18. Ranked opportunities

Rank by demonstrated avoidable complete-contract work. Only rank 1 passes that
test; the remaining rows are unresolved hypotheses, not additive opportunities.

| Candidate | Current cost µs | Demonstrated avoidable cost | B/op involved | Objects | Confidence | Complexity | Owner |
| --- | ---: | --- | ---: | ---: | --- | --- | --- |
| **1. Run-local arguments** | 0.2492 component | 0.1144 component; 0.153 ramp / 0.3745 fixed-policy transfer | 160 gross / 144 net saved | 12 removed | High allocation; moderate latency | Low | ONNX |
| Retention compatibility | 0.3020 nine lookups | Unknown; direct lookup removes guard | 0 | 0 | Measured cost, no equivalent fix | Medium | ONNX |
| Evaluation structures | 0.1170 creation | Unknown; reuse needs lifecycle proof | 480 | 4 | No complete-Run ablation | Medium | ONNX |
| Value validation | 0.9073 produced checks | Unknown; removal violates contract | 0 | 0 | Required checks measured | Medium/high | ONNX/backend API |
| Conv/Pool/Reshape geometry | 0.0316–0.0764 each helper | Unknown; checked shape allocation is not all validation | 144 combined | 5 | Fresh component evidence only | Medium | NeuralNetwork |
| MatMul plan/staging | 1.690 complete | No valid complete-Run recovery; plan 0.0622 | 144 | 3 | Arithmetic dominates | Medium/high | NeuralNetwork |
| Final Add adapters | 0.0913 adapter | About 0.09 component ceiling, no full-Run proof | 288 | 6 | Tiny mechanism | Medium | NeuralNetwork |
| Output publication | 0.1170 complete | None demonstrated | 193 | 6 | Ownership required | High | ONNX/backend |
| Context checks | 0.3468 homogeneous loop | None demonstrated | 0 | 0 | Equivalence not established | High | ONNX/backend |

## 19. Production decision

**Implement one focused ONNX optimization.** The ten-question gate was recorded
before editing production source. Only `evaluation.go` changes production logic;
`evaluation_test.go` adds explicit optional/fallback/normalization regressions.
No NeuralNetwork production source, module, dependency version, module-cache source or sibling
checkout is modified. Backend diagnostics overlay tests onto a byte-identical
copy of the published module, checked again during packaging. No backend handoff
or implementation branch is selected.

## 20. Production result

Seven fresh alternating/interleaved matched control/candidate pairs:

| Engine/version | Median µs | Range µs | CV |
| --- | ---: | --- | ---: |
| Go merged control | 51.2915 | 51.0000–51.4580 | 0.291% |
| Go candidate | 50.9170 | 50.7080–51.1455 | 0.278% |
| ORT control | 44.3125 | 43.7920–44.4160 | 0.446% |
| ORT candidate | 44.3750 | 43.9170–45.1660 | 0.791% |

Primary estimator: **difference of medians of per-report engine medians**.
Absolute Go reduction **0.3745 µs**, percentage **0.7301%**. ORT movement
**+0.0625 µs / +0.1410%**. Control ratio **1.15750×**, candidate **1.14743×**.
Paired Go reductions are all positive, range 0.0205–0.6465 µs, median 0.2285 µs;
these are secondary evidence and do not replace the primary estimator.

Separate ordinary ramp: **49.067 → 48.914 µs**, reduction **0.153 µs**.
Ranges 48.741–50.909 / 48.595–50.774, CVs 1.413% / 1.441%. The first high samples
remain. Component recovery 0.1144 µs is consistent with a small mechanism;
allocator/cache/measurement effects prevent assigning the larger fixed-policy
movement entirely to the pointer-construction instructions.

## 21. Allocation result

Every production ordinary-ramp sample: **1,409 → 1,265 B/op**,
**36 → 24 allocs/op**. Twelve 8/16-B pointer arrays disappear; a 16-B larger
allocation class for the existing evaluation offsets part of the 160-B gross
saving. No retained destination bytes, Session tables or output allocations
change. The final allocation profile confirms removal at execute while operator
and publication sites remain.

## 22. Correctness validation

Pass: default, CGO-disabled, purego, race, purego+race complete ONNX suites;
vet, module verification, tidy-diff; twelve default/purego cross-builds for
Darwin arm64/amd64, Linux arm64/amd64/386 and Windows amd64; four existing fuzz
targets, three seconds each with two workers; unchanged-policy purego comparison.

All fixed-policy reports pass numerical tolerance, top-1 and upstream checks.
Existing tests cover dynamic/symbolic shapes, initializer overrides/defaults,
aliases, NaNs/IEEE behavior, output ownership/survival, budgets, cancellation,
deadlines, failed-Run recovery, Busy/Close, Session isolation and concurrency.
New graph tests specifically detect stale optional Gemm bias and normalization
bias after a populated call, and exercise a five-input Concat fallback.
Both merged and final diagnostics verify retention through 1,000 Runs and
prepared-traversal output bits. No correctness assertion was weakened.

## 23. Metal validation

Native serialized ONNX and published-backend Metal suites pass with **zero skips**;
five isolated repetitions of `Test_SequentialResidentLongRunningMixedStress`
also pass. Native test/subtest pass counts: ONNX 2,036; backend 5,705; isolated
stress 5. Sandbox ONNX passes, while sandbox backend has 46 device-unavailable
skips; its log remains. These are correctness checks, not latency claims.

The historical Pass 16 unrestricted stress failure remains **unexplained**.
New passing runs do not retroactively explain it.

## 24. Updated 1.25× status

Fresh production ratio **1.14743×**, below 1.25×. At fresh candidate ORT 44.375 µs,
1.25× is 55.46875 µs; Go is 50.917 µs. Initial fresh ratio was 1.22463×, also
below 1.25×. Reference movement is reported, not normalized away.

## 25. Updated 1.10× status

**Not attained.** Fresh candidate arithmetic target: **48.8125 µs**, leaving
**2.1045 µs** relative to 50.9170 µs. That is arithmetic context, not demonstrated
recoverable work. No collection of component ceilings establishes that budget.

## 26. Recommendation for Pass 20

Investigate whether immutable retention metadata can avoid repeated validating
backend accessors while preserving exact dtype/shape compatibility and all
accounting/lifecycle contracts. The nine-lookup component is only about 0.302 µs,
with a direct-table lower probe around 0.0105 µs; any realistic benefit is small
and requires a new matched complete-contract measurement. Reject the idea if
metadata duplication/complexity outweighs transfer.

Reprofile the merged Pass 19 runtime first. Repeated backend metadata validation
may deserve a separate API/geometry investigation if fresh evidence establishes
a larger equivalent mechanism. No backend kernel rewrite or dependency handoff
is currently justified. The remaining shell cost is largely contractual; this
pass does not prove the runtime has reached a fundamental architectural floor,
or that its remaining gap can be recovered under the present design.
