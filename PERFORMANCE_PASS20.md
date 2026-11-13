# Performance optimization pass 20: metadata proof and repeated validation

One focused ONNX change reuses the actual tensor dtype within each value check.
It removes thirteen repeated backend validations per measured warm MNIST Run,
adds no state, and preserves every declaration and runtime guard. **No
NeuralNetwork production change is required.**

Seven final fixed-policy pairs show **51.1045 -> 50.9170 us**, a **0.1875 us /
0.3669%** Go reduction. ORT moves **44.4170 -> 44.6670 us** independently;
ratios are **1.15056x -> 1.13992x**. Allocations remain **1,265 B / 24 objects**.
The latency effect is small, and paired observations include reversals.

The primary question has a narrow positive answer: actual metadata proof can be
reused within one synchronous validation call. An existing prepared descriptor
can also support a retained-slot admission proof, demonstrated diagnostically,
but that alternative adds an invariant and state for less stable complete-Run
transfer. It remains an experiment. Prepared expected metadata alone cannot
prove that an actual produced tensor satisfies its declaration.

[Replay and artifacts](tools/compare_runtime/pass20/README.md),
[pre-implementation decision](tools/compare_runtime/pass20/DECISION.md), and
[all observations and statistics](tools/compare_runtime/pass20/data/summary.json)
form the evidence record. Times below are microseconds unless marked otherwise.

## 1. Fresh merged baseline

Measured merged ONNX `06ae89fbaba9f46c0e111fc7ef3b896eccb8e899` on branch
`optimization-pass-20`, with published NeuralNetwork **v0.10.10**, commit
`2ffa5bd9fd2034382c49d14efc732ea4d026aa09`. No backend replacement was used
in baseline or production measurements. Go **1.27.2 darwin/arm64**;
`GOARM64=v8.0`, `GOMAXPROCS=1`, `CGO_ENABLED=0`, `GOWORK=off`, `GOFLAGS=''`.
CPU: Apple M3, 8 cores (4 performance/4 efficiency), 24 GB; macOS **26.5.2
(25F84)**, Darwin **25.5.0**. Build cache: `/private/tmp/onnx-pass6/go-cache`.

Modules: NeuralNetwork v0.10.10, protobuf v1.36.12, simd v1.3.0, x/sys v0.45.0;
the complete module graph and published module sums are recorded in summary.json.
Reference: ORT **1.19.2**, NumPy 1.26.4, ONNX 1.16.2, Python 3.9.6,
Python protobuf 4.25.3. ORT uses CPUExecutionProvider, sequential execution,
one intra/inter-op thread, disabled graph optimizations.

Converted IR8/opset13 model SHA-256:
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
Unchanged driver SHA-256:
`9f32bdc790f2b323c53ac2d8fa1a8f777be8daa8d9f73aac3cd32f6f63b2bdae`.
Runner fingerprints:

| File | SHA-256 |
| --- | --- |
| runner.go | `9d1f4a83da63fd38ae8609ffc45b40d3bd81aa30b77d04085f7d4296a8832eec` |
| run_result.go | `f1357afe5986ff398f752f9cdc7d7415eb32b0d13e7740c161ec63f1539bd64f` |
| tensor_data.go | `e29a7e66c44068b91f73c200f5ad04aaf3230a47475966f3c6a3c4965cd200ac` |

Fresh initial measurements, seven reports and seven separate ramp observations:

| Scope | Median | Range | CV | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: | ---: |
| Fixed-policy Go | 51.0000 | 49.9375–52.6460 | 1.6264% | separate ramp | separate ramp |
| Fixed-policy ORT | 44.1670 | 41.8750–45.0000 | 2.8176% | not measured | not measured |
| Ordinary warm ramp | 48.5830 | 48.0080–49.1630 | 0.6866% | 1265 | 24 |

Initial ratio: **1.15471x**. Initial Go observations:
50.833, 50.667, 49.9375, 51.000, 51.1875, 52.646, 52.042.
ORT: 42.208, 41.875, 42.083, 44.208, 44.167, 44.500, 45.000.
No Pass 19 latency is substituted for a fresh measurement.

## 2. Fresh post-Pass-19 profile

Fresh all-allocation and five-second CPU profiles were collected before selecting
production work. The two Conv eight-filter arithmetic routines account for
**60.28% flat CPU samples**; output validation accounts for 2.99% cumulative,
backend Tensor.Validate 3.19%, and validateShapeSize 4.19%. OS/GC work appears too;
profile percentages are sampled attribution, not recoverable latency.

Ordinary counters confirm 24 objects. Allocation stacks still include five
Conv/Pool/Reshape geometry arrays, MatMul planning/staging, final Add matrix
adapters, evaluation/budget/value/symbol structures and owned publication. The
argument-array allocations removed in Pass 19 remain absent. Sampled/profile
totals include cold setup and calibration; they are not the ordinary B/op ledger.

Prepared traversal: **44.639 us / 576 B / 14 objects**. Individual current calls:

| Node | Prepared call | B/op | allocs |
| --- | ---: | ---: | ---: |
| Conv00 | 7.7330 | 32 | 1 |
| Add01 | 1.8890 | 0 | 0 |
| ReLU02 | 0.8730 | 0 | 0 |
| Pool03 | 2.4370 | 32 | 1 |
| Conv04 | 27.1330 | 32 | 1 |
| Add05 | 1.1670 | 0 | 0 |
| ReLU06 | 0.4561 | 0 | 0 |
| Pool07 | 0.9593 | 32 | 1 |
| Reshape08 | 0.08559 | 16 | 1 |
| MatMul10 | 1.6850 | 144 | 3 |
| Add11 | 0.1836 | 288 | 6 |

Reshape09 is an immutable retained result outside the eleven numerical calls.
Execution-shell diagnostic: **3.705 us / 689 B / 10 objects**, range
3.696–4.212, CV 4.634%. It removes numerical work and backend validation, so its
gap is not a production opportunity. Fresh complete component scopes: binding
0.8044, twelve produced-value checks 0.9100, acquisition 0.8513, publication
0.1175. Different scopes overlap and must not be summed into savings.

## 3. Metadata-access inventory

This inventory covers one complete finite, successful warm MNIST Run. Failed
Runs, NaN restoration, dynamic operators and serialization have different paths.
Public output inspection occurs outside inference timing.

| Caller -> callee | Object/fact | Frequency | Copy/allocation/validation | Mutability and purpose |
| --- | --- | ---: | --- | --- |
| bind -> ONNX Tensor.Validate -> backend Validate | supplied input type/shape/storage | 1 | validates; no copy/heap on success | actual caller value, Run dependent |
| bind -> ByteSize | supplied input payload size | 1 | full backend validation | actual reservation before declaration checks |
| constantValue -> Constant.ByteSize -> ByteSize | eight defaults' payload sizes | 8 | full validation; no copies | model immutable payload, legacy logical reservation retained |
| checkValue -> Rank | nine bindings and twelve produced tensors | 21 | full validation | actual rank/limit contract |
| checkValue -> Shape | same actual tensors | 21 | validates and copies; these shapes stay off heap | axes/limits/declarations/symbol binding |
| checkValue -> DataType | same actual tensors | 34 -> 21 | each read fully validates; no copy | actual dtype shared across declarations after this change |
| checkInfo -> prepared declarations/Info | expected dtype/rank/axes | 34 info checks | no shape extraction; context and symbols checked | expectations immutable, symbol bindings Run dependent |
| preparedCall.run -> call geometry/config/bytes/workspace | twelve descriptors | 12 | no reconstruction or shape copy | model immutable after preparation |
| runStorage.destination -> DataType/Shape | free intra-Run tensor | 9 each | validates twice; shape copy is local | actual free value differs by retirement; two matches/seven misses |
| destinationStorage.get -> DataType/Shape | retained destination | 9 each | validates twice; local shape copy | private geometry immutable; requested compatibility still required |
| runStorage.destination/trim -> ByteSize | discarded/free tensors | 7 + 1 | validation; no copy | actual legacy live-charge release |
| backend Into -> validateTensorType | inputs/destinations | 27 | actual type/shape/storage checks | backend's independent public contract |
| backend Into -> validateDestination | eleven mutable outputs | 11 | geometry and storage identities checked | actual value/destination ownership contract |
| backend shape helpers | Conv x2, Pool x2, Reshape x1, MatMul x1 | 6 helper calls | checked geometry; five small shape copies plus MatMul plan | actual inputs/configuration may differ outside ONNX prepared path |
| backend Add/broadcast helpers | ranks/axes/strides/scratch capacity | 3 Adds, 6 stride plans | bounded stack metadata for current broadcast ranks | geometry, not payload values; public backend checks remain |
| final Add -> matrix adapters | row/column/storage lengths | 3 adapters | owns copied payload; validates geometry | backend layer's storage contract |
| matrix adapter -> Matrix.Shape | output matrix geometry | 1 | two scalar loads, no validation/copy | adapter private geometry |
| matrix adapter -> ValuesInto | output values length | 1 | private validate and payload copy | publication into backend tensor |
| cloneOutput -> ByteSize/Clone | requested output | 1 each | actual validation; owned shape/payload allocation | independent public output ownership |

Backend numerical kernels also read private shape extents and payload lengths to
bound arithmetic and index storage; those are necessary operational loads, not
repeated public accessor calls. Capacity checks bound scratch buffers, not a
public tensor capacity promise. Tensor destinations require **exact storage
length**, not merely sufficient capacity. There is no separate Tensor capacity
accessor to remove. Error construction reads input shapes only on failure.

## 4. Accessor-frequency table

Instrumentation counted two successive warm Runs with identical results.
Original module-cache and sibling backend source remain unchanged. Go initially
rejected a module-cache overlay; the successful counter binary uses a verified
byte-identical published-module copy, a temporary modfile and diagnostic source
overlays. **Instrumented latency is never performance evidence.**

Costs below divide a nine-operation retained-tensor probe by nine, including loop
overhead. ShapeLocal also includes exact comparison. These measured shapes really
occur in the Run; costs are not extrapolated to other ranks or added together.

| Accessor / operation | Calls/Run before -> after | ns/call probe | Warm heap allocations | Copies | Purpose |
| --- | ---: | ---: | ---: | --- | --- |
| Tensor.DataType | 52 -> 39 | 14.57 | 0 | none | actual type and full storage validation |
| Tensor.Shape | 39 -> 39 | 18.40 incl comparison | 0 for current local uses | shape axes | compatibility and declaration checking |
| Tensor.Rank | 21 -> 21 | 14.20 | 0 | none | actual rank limit |
| Tensor.Len | 0 -> 0 | 14.62 | 0 | none | not called in this warm prepared graph |
| Tensor.ByteSize | 18 -> 18 | 14.91 | 0 | none | reservation/release |
| Tensor.Validate, including accessor internals | 161 -> 148 | 14.30 | 0 | none | identity/type/dimensions/product/storage lengths |
| validateShapeSize, including Validate | 204 -> 191 | not separately isolated | 0 itself | none | representability and exact element/byte counts |
| ValidateShape, excluding accessor internals | 42 -> 42 | scope dependent | caller owned | helper dependent | Conv/Pool/Reshape/broadcast/MatMul geometry |
| validateTensorType | 27 -> 27 | not isolated | 0 | none | backend actual argument type |
| validateDestination | 11 -> 11 | Pool full validation ~98 ns | 0 itself | none | actual geometry and alias identities |
| backend shapesEqual | 20 -> 20 | scope dependent | 0 | none | exact shapes and broadcast checks |
| Matrix.Shape | 1 -> 1 | not isolated | 0 | two scalars | actual adapter geometry |
| Matrix private validate / matrixSize | 4 / 7 unchanged | not isolated | 0 | none | exact adapter storage/representability |
| Matrix Rows/Cols/public Validate/Values | 0 unchanged | not measured | 0 | none | absent on this warm path |
| Tensor.Clone / newTensor | 1 / 1 unchanged | publication scope 117.5 ns | ownership allocations remain | shape/payload | independently owned output |

Of the original 52 dtype reads, 34 are checkValue, nine retained lookup, nine
intra-Run lookup. The original 39 shape reads are 21/9/9 in those same scopes.
The selected change reduces checkValue dtype reads **34 -> 21** and nothing else
in the accessor inventory. Full caller/callee counters are archived.

## 5. Mutability classification

| Class | Facts | Reuse boundary |
| --- | --- | --- |
| A: model immutable | declarations/Info, prepared output geometry/configuration/bytes/workspace, immutable initializers and proven constant targets | model lifetime; expectations still require actual-value checks |
| B: Session immutable | retained slot association after admission, private destination dtype/shape/length/identity | admitted slot until Close; payload is rewritten each Run |
| C: Run dependent | supplied values/default choice, override choice, requested outputs, symbol bindings, free candidate, budgets/cancellation | fresh each Run |
| D: value dependent | actual dtype/rank/shape/storage length and identity of bound/produced tensor | actual object; payload/NaNs can affect returned object/ownership |
| E: contract validation | actual value against original declarations and inferred Info; backend arguments/config/destination aliases/storage; accounting/error order | preserved at every responsible layer |

Some facts occupy multiple classes: a retained tensor's shape is immutable for
that object, but deciding that it satisfies the current request is contract work.
An expected static shape does not identify the actual produced storage. The
production scalar lives only within one checkValue call; no cross-Run or
value-dependent result is cached.

## 6. Prepared-metadata duplication map

Every benchmark node has concrete prepared Float32 inputs/outputs, except Reshape
target tensors which are Int64 `[2]`. All listed output geometries are exact:

| Node | Prepared input geometry | Prepared output | Override/alias/storage notes |
| --- | --- | --- | --- |
| Conv00 | `[1,1,28,28]`, `[8,1,5,5]` | `[1,8,28,28]` | actual backend inputs/config/aliases checked |
| Add01 | `[1,8,28,28]`, `[8,1,1]` | `[1,8,28,28]` | channel broadcast; private retained destination |
| ReLU02 | `[1,8,28,28]` | `[1,8,28,28]` | intra-Run destination; NaN/cancellation restoration unchanged |
| Pool03 | `[1,8,28,28]` | `[1,8,14,14]` | actual pool config and destination checked |
| Conv04 | `[1,8,14,14]`, `[16,8,5,5]` | `[1,16,14,14]` | no packed weights or arithmetic changes |
| Add05 | `[1,16,14,14]`, `[16,1,1]` | `[1,16,14,14]` | channel broadcast |
| ReLU06 | `[1,16,14,14]` | `[1,16,14,14]` | intra-Run destination |
| Pool07 | `[1,16,14,14]` | `[1,16,4,4]` | actual backend geometry checked |
| Reshape08 | `[1,16,4,4]`, target `[2]` | `[1,256]` | target provenance immutable; actual source/destination still validated |
| Reshape09 | `[16,4,4,10]`, target `[2]` | `[256,10]` | immutable retained result, never writable destination |
| MatMul10 | `[1,256]`, `[256,10]` | `[1,10]` | backend plan/staging unchanged |
| Add11 | `[1,10]`, `[1,10]` | `[1,10]` | matrix adapters unchanged |

The benchmark initializers are not graph inputs, so cannot be overridden. The
general API allows declared defaulted inputs to be overridden; those cases retain
their existing checks and preparation rules.

| Fact | Already prepared? | Re-read at Run? | Why? | Can proof be reused? |
| --- | --- | --- | --- | --- |
| requested output dtype/geometry | yes | descriptor loads only | destination reservation/selection | already reused; no second requested cache needed |
| requested logical byte/workspace size | yes | descriptor loads only | original accounting contract | already reused |
| actual dtype for each declaration | first actual read is sufficient within checkValue | repeated per declaration | separate checkInfo arguments | **yes, selected local reuse** |
| actual output rank/shape | expected shape known | actual Rank/Shape | detect actual contract violations and bind original symbols | keep actual checks |
| retained actual geometry | prepared expectation plus admission validation | actual metadata each get | conservative compatibility | admission proof works diagnostically; not selected |
| free candidate geometry | output descriptors exist, actual free owner varies | nine actual reads | retirement/reuse compatibility | not established as equivalent reusable proof here |
| backend Conv/Pool/Reshape output geometry | ONNX descriptor knows expected geometry | recomputed from actual inputs/config | public backend validates its own arguments and failure semantics | backend API hypothesis only |
| initializer dtype/shape/bytes | immutable defaults known | checkValue and ByteSize | bindings, reservations, original declarations | unchanged; no separate measured complete-contract candidate |
| output storage identity/ownership | expected shape known | backend identities and Clone | alias safety and output survival | geometry proof cannot replace ownership |

## 7. Retained-destination compatibility decomposition

Fresh nine-successful-lookup scope: **0.2977 us**, direct-table traversal
**0.01094 us**. The **0.28676-us difference is a diagnostic lower-bound gap**,
not automatically removable work. No allocation occurs in the actual lookup.

| Part | Fresh evidence | Contract |
| --- | --- | --- |
| slot/table traversal, bounds, nil | direct scope 0.01094 us is a traversal floor | disabled retention, invalid slot, not-yet-admitted candidate |
| dtype determination/comparison | nine DataType calls 0.1311 us | actual validated type vs request |
| requested shape | direct prepared descriptor slice | no validating requested accessor to eliminate |
| stored shape extraction/comparison | nine ShapeLocal calls 0.1656 us | actual shape/product/storage validation, copied axes, exact comparison |
| shape comparison alone | nine equal distinct slices 0.03610 us | exact rank/axes; not merely equal element count |
| shape copy alone | nine local copies 0.02042 us | defensive metadata ownership |
| storage/capacity | inside both backend Validate calls | exact typed lengths and initialized identity |
| reservation/accounting | before get in runStorage.destination | original logical reservation plus independent retained residency |

These scopes share traversal, validation and comparison work; **do not add them**.
The first comparison probe used differing final-output geometry and included rank
mismatches (0.02454 us). It is preserved and labeled; the separate CompareExact
probe above is used for equal-shape reasoning.

The static proof diagnostic validates the actual admitted tensor against the
existing immutable prepared descriptor, never replaces a populated slot, compares
the current requested dtype/shape with that descriptor, and leaves backend Into
storage/alias validation in place. It needs a borrowed descriptor-table reference
(24 bytes of struct fields before allocator size-class effects), not copied shapes.
It does not demonstrate compatibility for a runtime-mutable geometry API or
arbitrary internal replacement that bypasses admission. Public API users cannot
modify these private fields. The ordinary full checker remains the production
path; malformed or incompatible candidates therefore retain the existing fallback.

## 8. Dynamic/symbolic analysis

Preparation requires concrete known fixed input dimensions and valid operator
geometry. Unknown/symbolic input signatures decline preparation. Retention also
requires a whole-plan conservative live bound and actual prepared geometry equal
to fixed output Info. Symbols that inference refines away still remain in original
declarations and bind/check against the actual Run's symbol table.

Dynamic Reshape shape values must be read each Run, even if the target tensor's
own length is static. Defaulted/overridable shape tensors do not count as immutable
target provenance. Changing input dimensions, zero-sized dimensions, empty values,
incompatible reuse and fallback allocation keep their established paths.

The selected local dtype reuse needs **no static eligibility test**. It reads each
actual tensor afresh and retains all rank/axis/symbol checks, including scalars
and zero dimensions. Existing dynamic/symbolic/empty suites plus new direct
declaration tests pass in all required configurations.

## 9. Override analysis

bind validates supplied wrappers and reserves their actual bytes before checking
original declarations and inferred Info. Missing defaulted inputs borrow the
immutable default. Compatible overrides can change values; incompatible dtype or
geometry fails with the existing argument/model distinction and ordering.

Prepared expectations remain valid only when those checks succeed. Immutable
constant propagation explicitly excludes overridable graph inputs; malformed
defaults remain runtime errors so valid overrides can recover them. The local
dtype scalar is extracted after binding selects the actual default/override and
again for every subsequent value. No prepared expected type replaces that read.

Tests cover defaults, compatible/incompatible overrides, malformed defaults,
override-sensitive Reshape targets and subsequent successful Runs.

## 10. Alias/ownership analysis

Geometry equality does not prove disjoint storage. Backend validateDestination
compares actual storage identities to every non-nil validated input, including
empty shallow aliases. All eleven calls and their identity checks remain.

The lifetime plan pins inputs, defaults, Constant results and Identity alias
groups. Immutable Reshape09 storage cannot become writable. ReLU's NaN path may
produce distinct storage while preserving legacy destination state; its behavior
is unchanged. Public outputs clone shape and payload, remain independent of
retained private storage and each other, and survive later Runs/Close. Session
retention remains nine tensors / **83,664 payload bytes**, bounded by the existing
121,296-byte limit for this model. Local dtype reuse introduces no ownership state.

## 11. Validation-responsibility map

| Layer | What/when/inputs | Error/ownership responsibility | Reusable proof |
| --- | --- | --- | --- |
| compile/preparation | declarations, immutable target provenance, operator attributes and expected geometry at load | model support/limits; failed optional preparation falls back | expected geometry/configuration |
| ONNX input binding | actual supplied tensor/default and actual byte reservation per Run | argument errors, original declarations, inferred Info and symbols; reservation precedence | actual dtype only within same checkValue |
| ONNX produced check | actual kernel result per node | original/inferred output declarations, symbols/limits, node/value error context | actual dtype only within same checkValue |
| ONNX retention/reuse | actual private candidate vs current request | exact compatibility and residency/legacy accounting | prototype admission invariant only |
| backend accessors | actual Tensor on every accessor | invalid receivers/types/shapes/lengths return documented invalid/zero/nil values | repeated identical actual dtype result can be shared locally |
| backend public Into | actual inputs, config, destination each call | public validation order, disjoint storage, all-errors destination preservation | ONNX expectation alone is insufficient |
| publication | actual requested source, budget and context | independent storage and survival | no metadata proof replaces Clone |

ONNX and backend see different contracts and different expectations. A backend
argument may be shape-valid yet violate a model declaration; an expected model
shape cannot certify actual backend storage. Only deterministic repeated dtype
validation **inside the same unchanged-object check** disappears in production.
All checkInfo calls, context calls and error branches remain.

## 12. Backend accessor findings

Published tensor/tensor.go implements DataType, Shape, Rank, Len and ByteSize by
calling full Validate first. Validate checks initialized identity, supported dtype,
each dimension's host representability, checked dimension product/byte size and
exact storage length/type. Shape returns an owned copied slice. Caller mutation
or retention of that slice cannot mutate backend geometry.

Constructors own the metadata. Searches of non-test tensor source find geometry,
dtype and identity initialized in newTensor; Into operations alter payloads and
do not resize or replace metadata. Matrix.Shape is a separate two-scalar accessor
without validation. There is no existing general Tensor prepared/read-only
metadata snapshot API that ONNX can use to combine all these validations.

No backend production file, sibling checkout, module-cache file, dependency
version, implementation branch or release was changed. Counter overlays contain
diagnostic instrumentation only; production measurements use the published module.

## 13. Metadata-copy findings

Nine retained shapes contain 30 axes, **240 bytes copied per lookup traversal**.
Their immediate-use Shape copies have zero ordinary heap allocations. Keeping
Shape results alive in an owned probe forces **240 B / 9 objects**, at 0.2337 us;
that deliberate ownership probe is not the warm runtime's allocation behavior.

Local shape copy scope is 0.02042 us, much smaller than validating/extracting and
comparing the same shapes (0.1656 us). Defensive ownership and encapsulation
explain the copies. Error formatting can force otherwise-local shapes to escape
on failed paths. The selected dtype change removes **no shape copies or bytes**.

Backend Conv/Pool/Reshape helpers return owned resolved shapes even when ONNX
already owns expected shapes. Those five warm allocations are real, but replacing
them requires equivalent backend public validation and ownership semantics.

## 14. Generated-code/escape findings

ARM64 objdump and `-m=2` listings were collected for merged and candidate source.
DataType/Shape wrappers inline; Tensor.Validate remains a call. Merged checkValue
contains dtype validation inside the declaration loop and again for final Info;
candidate extracts it once before that loop and passes the scalar to every
unchanged checkInfo call. Compiler inline-cost estimates drop **770 -> 707**;
checkValue itself remains uninlined. No interface/API/assembly rewrite occurs.

Shape's current local scratch is 32 bytes: ranks up to four use stack storage;
larger ranks still have a makeslice fallback. memmove remains for axis copying.
Successful checkValue Shape copies do not escape, while failure diagnostic shape
lists may escape. Retained lookup retains both actual Validate calls, exact shape
loop, bounds/nil checks and the stack/copy machinery in production.

Generated-code findings explain a small CPU reduction; they do not establish
additional heap savings or a sum of future savings.

## 15. Controlled ablations

All variants used merged Pass 19 evaluation source and a common fixture. Variant
order reverses in alternating repetitions; every valid observation is retained.
These are Go-only complete-Run diagnostics; no ORT ratio is assigned to them.
The requested-geometry hypothesis is an identity with current production:
preparedCall.run already supplies immutable requested dtype/shape/bytes, so no
second purported requested-geometry optimization was manufactured.

| Variant | Median | Range | CV | Control minus variant | B/op | allocs |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| current full compatibility | 49.052 | 48.993–49.177 | 0.127% | 0 | 1265 | 24 |
| local actual dtype reuse | 48.907 | 48.729–49.026 | 0.188% | 0.145 | 1265 | 24 |
| prepared retained admission proof | 48.920 | 48.792–53.348 | 3.592% | 0.132 | 1265 | 24 |
| unconditional direct retention, **invalid** | 48.790 | 48.614–50.514 | 1.262% | 0.262 | 1265 | 24 |
| execution shell, **invalid** | 3.705 | 3.696–4.212 | 4.634% | 45.347 | 689 | 10 |

All latency observations, ordered by repetition:

| Variant | Observations |
| --- | --- |
| control | 49.012, 48.993, 49.109, 49.052, 49.177, 48.993, 49.062 |
| local dtype | 48.729, 48.949, 48.962, 48.888, 48.811, 49.026, 48.907 |
| prepared proof | 48.792, 48.920, 53.348, 48.833, 49.014, 48.794, 52.224 |
| invalid direct | 50.514, 48.810, 48.787, 48.790, 48.662, 48.614, 48.988 |
| invalid shell | 3.825, 4.212, 3.703, 3.697, 3.696, 3.720, 3.705 |

Local dtype preserves all contracts. Prepared proof establishes actual dtype/shape
on admission, keeps nil/bounds/requested-geometry checks and preserves every
reservation and backend identity check; it depends on immutable slot geometry
and admission-only writes. Its nine-lookup scope becomes 0.05233 us, acquisition
0.6278 us; produced checks remain 0.9209 us. No copied geometry or payload is
added, but a Session table reference and admission invariant are new complexity.

Invalid direct removes compatibility without equivalent proof. Invalid shell
removes kernels and backend validation. Neither is eligible production evidence;
none of these deltas is combined. Initial overlay/tooling failures and the
corrected comparison probe are retained in the evidence notes/logs.

## 16. Broader metadata opportunity

The repeated-access mechanism is broader than retention: output validation alone
performed 24 dtype extractions for twelve nodes, and input/default binding another
ten for nine values. The selected local reuse handles both through their shared
checkValue implementation and avoids any second metadata architecture.

Conv, Pool, Reshape and MatMul already use prepared expected geometry in ONNX,
but backend Into validates actual geometry independently. Add still needs actual
broadcast/stride proof; publication still needs actual byte reservation and owned
storage. No additional production work is selected in those paths. No evidence
establishes that all repeated backend validation can safely be bypassed.

## 17. Backend geometry result

Fresh backend component evidence from test-only overlays on the exact publication:

| Scope | Time | B/op | allocs |
| --- | ---: | ---: | ---: |
| Conv00Shape | 0.07638 | 32 | 1 |
| Conv04Shape | 0.07595 | 32 | 1 |
| Pool03Shape / full validation | 0.06639 / 0.09792 | 32 | 1 |
| Pool07Shape / full validation | 0.06633 / 0.09816 | 32 | 1 |
| Reshape08Shape / complete Into | 0.03250 / 0.08446 | 16 | 1 |

An eventual general backend prepared-operation concept would need to own expected
geometry/configuration and preserve validation of actual dtype/shape/storage,
aliases, zero/empty dimensions, checked arithmetic, device behavior and all-error
destination preservation. ONNX descriptors are expectations, not backend-issued
validation capabilities. No existing general API consumes that proof and no
equivalent backend complete-Run diagnostic was established here.

**No NeuralNetwork implementation handoff is justified in this pass.** Helper
costs alone do not establish equivalent recovery. No backend shortcut or prompt
asserting a production-ready mechanism is supplied. Future backend investigation
must independently establish a general contract before implementation.

## 18. Ranked opportunity table

Ranked by demonstrated equivalent recovery, with complexity breaking the small
latency tie. Costs/scopes overlap and are not additive.

| Candidate | Current cost | Demonstrated equivalent recovery | B/op / allocs removed | Confidence | Complexity | Owner |
| --- | ---: | --- | --- | --- | --- | --- |
| **local dtype reuse** | 0.9100 produced checks, plus binding | 0.1199 produced component; 0.145 diagnostic Run; 0.1875 final fixed-policy | 0 / 0 | high mechanism; modest latency evidence | one local scalar | ONNX |
| prepared retained admission proof | 0.2977 nine lookups | 0.24537 lookup; 0.132 diagnostic Run with high CV | 0 / 0 | component clear; full transfer less stable | Session reference + admission invariant | ONNX |
| requested-geometry cache | already prepared | **zero additional mechanism** | 0 / 0 | high | unnecessary state | ONNX |
| combined backend metadata extraction | 161 actual Validate calls before change | unknown; layers protect different contracts | unknown | hypothesis | new general backend API | NeuralNetwork |
| Conv/Pool/Reshape prepared backend geometry | helper times above | **no equivalent complete-Run recovery shown** | 144 B / 5 objects involved, not demonstrated removed | component only | general backend proof/failure contract | NeuralNetwork |
| unconditional compatibility bypass | invalid diagnostic 0.262 Run movement | **none equivalent** | 0 / 0 | invalid | weakens guard | rejected |

## 19. Production decision

**Choose exactly one: focused ONNX optimization.** The fourteen-question gate is
recorded in DECISION.md before production edits. Actual dtype is extracted once
per unchanged-object check and reused across original declarations and inferred
Info. No field, slice, cache, prepared-descriptor modification, retained-memory
increase or backend dependency change is introduced.

Implementation commit: `cc995e6` (Reuse actual dtype within runtime value validation).

No static destination fast path, argument-array follow-up, arithmetic/operator
work, output-ownership change or unrelated allocation cleanup is bundled.

## 20. Production result

The unchanged policy uses upstream/zeros/ones plus sixteen seed-0 uniform cases,
nineteen sorted cases, ten warmups and 100 timed calls per case, independently
owned outputs, Go then ORT, atol=1e-4 and rtol=1e-5. Conversion/upstream/top-1
checks are unchanged. Loading/building/serialization stay outside inference timing.

Seven fresh alternating matched control/candidate pairs; primary estimator is
**difference of medians of per-report engine medians**:

| Engine | Median | Range | CV |
| --- | ---: | --- | ---: |
| Go merged control | 51.1045 | 50.3330–51.6660 | 0.7452% |
| Go candidate | 50.9170 | 50.7500–51.4375 | 0.4239% |
| ORT control | 44.4170 | 42.0000–45.1045 | 2.2001% |
| ORT candidate | 44.6670 | 41.7080–45.0420 | 3.1002% |

Go reduction: **0.1875 us / 0.3669%**. ORT movement: **+0.2500 us / +0.5628%**.
Control ratio **1.15056x**, candidate **1.13992x**. ORT movement contributes to
the ratio change; it is not attributed to the Go implementation.

| Pair | Go control | Go candidate | ORT control | ORT candidate | Go reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 51.4170 | 51.0420 | 42.0000 | 42.0000 | 0.3750 |
| 2 | 51.6660 | 50.7500 | 43.8750 | 41.7080 | 0.9160 |
| 3 | 50.3330 | 50.9170 | 44.2705 | 44.8340 | -0.5840 |
| 4 | 51.1045 | 50.8750 | 45.1045 | 42.9795 | 0.2295 |
| 5 | 51.0415 | 51.4375 | 44.4170 | 44.6670 | -0.3960 |
| 6 | 51.1670 | 50.7915 | 44.8330 | 44.9160 | 0.3755 |
| 7 | 51.0830 | 51.0830 | 44.8330 | 45.0420 | 0 |

Paired reductions are secondary: median 0.2295, range -0.5840–0.9160,
four positive/two negative/one tied. Their large relative CV (359.76%) and
reference movement limit precision; this is not uniform per-pair improvement.

Separate ordinary ramp: **49.152 -> 48.932 us**, reduction **0.220 us**;
ranges 48.812–49.865 / 48.708–50.269, CVs 0.6575% / 1.2196%. All samples,
including high initial/candidate ramp observations, remain in summary/evidence.
The simple repeated-validation mechanism, component recovery and diagnostic Run
agree on a small effect; the fixed-policy difference is not a precise instruction
cost or a promise across machines.

## 21. Allocation result

Every ordinary baseline/control/candidate sample reports **1,265 B / 24 objects**.
Removed allocation sites: **none**. Sampled final allocation stacks retain the
same metadata-geometry, adapter, evaluation and output sites. Shape scratch
remains local in the measured graph. Session retention stays nine tensors /
83,664 payload bytes; production introduces no metadata or retained-payload state.

## 22. Correctness validation

Pass: complete default, CGO-disabled, purego, race and purego+race ONNX suites;
vet, module verification and tidy-diff; twelve default/purego builds covering
Darwin arm64/amd64, Linux arm64/amd64/386, Windows amd64; four existing fuzz
targets for three seconds each with two workers; unchanged-policy purego
comparison. All 21 default fixed-policy reports pass numerical/upstream/top-1
checks. No correctness assertion or tolerance is weakened.

New table-driven checkValue cases preserve original symbolic declarations plus
inferred fixed geometry, empty Int64 symbolic geometry, scalar/unknown dtype,
actual dtype violations against either declaration layer, and original-rank-error
precedence over inferred dtype errors. All actual metadata is still validated.

Existing suites cover dynamic/symbolic shapes, changing shape targets, empty
values, defaults/overrides, aliases, IEEE/NaN behavior, destinations/accounting,
public output independence/survival, cancellation/deadlines, failed-Run recovery,
Busy/Close, Session isolation and concurrency. Both control and local-dtype
diagnostics verify 1,000 Runs with stable retained identities/accounting and
prepared traversal output bits. Delegating-context diagnostics remain **337 Err
calls per Run**. Build/escape snapshots and hashes identify measured source.

## 23. Metal validation

Native serialized ONNX and published-backend Metal suites pass with **zero skips**.
Pass/subtest counts: ONNX **2,043**, backend **5,705**. Five isolated repetitions
of Test_SequentialResidentLongRunningMixedStress pass. Sandbox ONNX passes;
sandbox backend passes with **46 device-unavailable skips** (5,628 pass/subtests).
All failures/skips/logs are retained; none is silently relabeled as flaky.

The historical Pass 16 unrestricted stress failure remains **unexplained**.
New passing checks do not retroactively explain it. These are correctness checks,
not Metal latency claims.

## 24. Updated 1.25x status

Fresh initial ratio **1.15471x** and final candidate ratio **1.13992x** are below
1.25x. At fresh candidate ORT 44.6670 us, the arithmetic 1.25x threshold is
**55.83375 us**; measured Go is 50.9170 us. Reference movement remains visible.

## 25. Updated 1.10x status

**Not attained.** Fresh arithmetic threshold: **44.6670 x 1.10 = 49.13370 us**.
Measured Go 50.9170 us leaves **1.78330 us**. This is arithmetic context only,
not demonstrated recoverable work. No component ceilings are summed to claim it.

## 26. Recommendation for Pass 21

Reprofile the merged small change first. The strongest unresolved local metadata
question is whether retained admission proof transfers reliably enough to justify
its extra invariant; the existing diagnostic and exact geometry checks provide
a bounded starting point, not an instruction to ship it. Stop if repeatable
complete-contract recovery does not justify that state.

A broader backend snapshot/prepared-operation API deserves a separate diagnosis
only if it can combine actual metadata extraction while retaining each public
validation/error/alias contract. Current helper costs do not justify a backend
implementation handoff, kernel work or renewed MatMul/Pool/Add optimization.
Keep required actual-value validation and use evidence to choose the next pass.
