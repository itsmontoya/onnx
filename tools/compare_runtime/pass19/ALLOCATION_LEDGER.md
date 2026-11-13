# Fresh post-retention allocation ledger

Ordinary merged Run: **1,409 B / 36 objects** in all initial and production-control
ramps. This ledger uses fresh rate-one profiles, current source, independent
component counters and controlled removal. Logical bytes and allocator-rounded
bytes are different. Tiny one-byte identities contribute amortized ordinary bytes;
profile coalescing and MemStats barriers cannot attribute each tiny slab exactly.

## Allocation sites and objects

| Site | Object | Allocator B/Run | Objects/Run |
| --- | --- | ---: | ---: |
| `newEvaluation` | evaluation, size class 128 B | 128 | 1 |
| `newRunBudget` | limits/live/workspace object | 128 | 1 |
| `newRunStorage` | 22-element value pointer backing array | 176 | 1 |
| `newEvaluation` | empty symbol map header, no buckets here | 48 | 1 |
| `evaluation.execute` | eight two-pointer and four one-pointer arrays | 160 | 12 |
| `tensor.Conv2DShape` | two rank-four result-shape arrays | 64 | 2 |
| `tensor.Pool2DShape` | two rank-four result-shape arrays | 64 | 2 |
| `tensor.ReshapeShape` | rank-two copied target for activation Reshape08 | 16 | 1 |
| `tensor.newMatMulPlan` | MatMul plan header | 80 | 1 |
| `tensor.newMatMulPlan` | rank-two result-shape array | 16 | 1 |
| `tensor.matMulBatchesInto` | ten-float temporary result, 40 logical B | 48 | 1 |
| `matrix.FromSlice` twice / `matrix.New` once | three matrix headers | 144 | 3 |
| same final Add adapter constructors | three ten-float arrays, 120 logical B | 144 | 3 |
| `cloneOutput → Tensor.Clone → FromFloat32 → newTensor` | owned output payload, 40 logical B | 48 | 1 |
| same output clone | backend tensor header | 96 | 1 |
| same output clone | rank-two copied shape | 16 | 1 |
| same output clone | one-byte storage identity, amortized tiny bytes | 1 | 1 |
| `evaluation.cloneOutput` | public ONNX Tensor wrapper | 8 | 1 |
| `evaluation.publish` | one-element NamedTensor backing array | 24 | 1 |
| **Total** | | **1,409** | **36** |

Grouping: evaluation construction 480 B / four objects; arguments 160 B / twelve;
operator metadata/staging/adapters 576 B / fourteen; publication 193 B / six.
There is **no warm allocation row** for retained destination payloads/headers,
slot bookkeeping, borrowed initializers, retained constant Reshape09, channel Add
broadcast geometry, prepared descriptors, output-name strings, context.Background,
or successful error wrappers. Immediate-use Shape accessor copies remain off the
heap with this compiler, although their validation/copy CPU work remains.

## Purpose, ownership and semantic necessity

| Group | Purpose, owner and lifetime | Payload / mutability | Dynamic behavior | Ownership, failure and concurrency obligations | Heap/avoidability finding |
| --- | --- | --- | --- | --- | --- |
| Evaluation/budget | Current invocation's context, plan, limits, live/workspace state | Mutable metadata; context/plan references fixed | Actual shapes and reservations determine counters | Independent per evaluation; failed state must not reach another Run; no public escape required | Returned constructor pointers escape. Reusing/embedding them is not proven by this pass; no production change |
| Value table | Evaluation resolves graph slots to actual current tensors, then retires references | Mutable pointer metadata | Topology fixes length; runtime values/signatures vary | No stale inputs after failure; no shared mutable plan table across Sessions | Variable backing array escapes through storage; keeping old arrays would need cleanup/lifetime proof |
| Symbol map | Evaluation binds consistent symbolic dimensions | Mutable metadata; empty in this fixed graph | Symbolic models need actual bindings and possibly buckets | Conflicts must fail; bindings must not leak across Runs/Sessions | Header escapes with evaluation; lazy construction is unmeasured, not selected |
| Arguments | One synchronous call's ordered current inputs including omitted nils | Mutable pointer metadata | Actual arity determines length; large/variadic calls retain fallback | No output owns the slice; optional slots must clear; original cancellation/binding order remains; each evaluation private | Runtime-sized makeslice escapes through calls. Fresh backing objects are unnecessary for up to three inputs; selected fix uses existing evaluation allocation |
| Conv/Pool shape arrays | Backend Into validator's checked output geometry during one call | Immutable metadata after construction | Depends on actual input/configuration; not universally static | Validate before writing, maintain alias/storage/error behavior; local temporaries isolate concurrent calls | Public shape helpers return owned arrays and escape; internal allocation may be avoidable, but equivalent complete-Run behavior not demonstrated |
| Reshape shape | Backend compares element counts and validates copied target geometry | Immutable metadata | Current source/target counts matter | Disjoint destination and all-errors atomicity remain required | Public helper owns copied shape; Into only needs transient geometry, but no backend change selected |
| MatMul plan/result shape | Checked promotions, batch broadcast and result geometry for one call | Immutable metadata | Vectors, broadcast batches, overflow, empty contraction/output and signatures remain | All shape/offset/alias failures must occur before writes; concurrent calls cannot share mutable plans | Plan and owned result shape escape. Fresh planning costs 0.0622 µs; no production-valid complete-Run savings established |
| MatMul staging | Complete product before committing to supplied destination | Mutable numerical payload, private to one call | Depends on full actual output/batch size and device path | Preserves all returned errors leaving destination unchanged across batches/devices | Variable result array escapes; finite host direct-product ceiling does not prove staging unnecessary |
| Add headers/payloads | Tensor-to-matrix adapter copies equal-shape inputs and stages matrix output | Mutable metadata and temporary payload | Only equal-shape branch; broadcasts already allocate zero here | Preserve Matrix copying/device contracts, tensor alias rejection and error behavior; temporary storage stays private | Constructors return heap-owned copies. About 0.09-µs component ceiling; tiny and backend-owned, no complete-Run proof |
| Output payload/header/shape/identity | Independently owned returned tensor, caller lifetime | Payload plus fixed metadata and alias identity | Actual requested output type/shape controls allocation | Must survive later Runs and all Close calls; cancellation/failure must not publish partial results; cannot borrow retained internal storage | Clone constructors escape to caller. Current independent storage contract remains required |
| Output wrapper/publication slice | Returned API values and names, caller lifetime | Stable metadata; names borrow immutable plan strings | Output selection determines slice count/order | Previously returned results cannot be mutated by later Runs; no shared publication buffers | Public result necessarily escapes; no unnecessary surrounding allocation demonstrated |

The production change removes twelve argument objects (160 B gross) while the
existing evaluation moves from allocation class 128 to 144 B. The resulting
ordinary total is **1,265 B / 24 objects**: 496 B construction + 576 B operators +
193 B publication. Actual argument construction/binding with the helper is
0.1348 µs / zero allocations; required assignments/context checks remain.

## Census and profile limitations

Merged MemStats census, 1,000 Runs with no logging inside measured scopes:
construction 480 B/four objects; binding zero; node scopes show argument slices
plus exactly the operator categories above. One incidental 16-B allocation is
visible in the first Conv scope. Publication averages 209.232 B / 6.005 objects
under barriers; ordinary component publication is 193 B/six objects. The barrier
runs affect tiny packing and background allocation, so their raw bytes are not
used to invent an ordinary-counter remainder.

The rate-one profile contains one cold population plus 1,000 timed invocations;
its large tensor.newTensor flat bytes include initial retained payloads and owned
output clones. Its tiny objects coalesce. Do not divide every line by 1,000 and
call the result ordinary bytes/objects. The archived listings and binary pprof
files retain that scope. Cold/second tests with StopTimer also alter tiny packing:
1,424 B / 36 objects for isolated second Run, versus ordinary 1,409 B / 36.

Retained destination allocation storage is approximately 91,449 B across nine
payload/header/shape/identity groups, with 83,664 logical payload B charged to the
Session on top of unchanged Run reservations. These objects persist until Close
and are **not** warm B/op. Pointer stability through 1,000 merged and candidate
Runs, ordinary allocation counters and profiles establish absence of fixed-shape
replacement churn. No RSS or global physical-memory accounting claim is made.
