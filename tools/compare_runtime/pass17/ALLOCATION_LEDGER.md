# Fresh allocation and ownership ledger

Ordinary released Run: **92,826 B/op, 72 allocs/op** in every one of seven ramp
samples. This ledger was rebuilt against v0.10.10, from current source, an
allocation-rate-one profile, independent component benchmarks, per-scope ordinary
counters, and controlled removal of destinations, argument arrays and publication.
It is not copied from Pass 16.

Bytes below are allocator-rounded ordinary benchmark bytes, except identity
bytes, which are the amortized tiny-allocator contribution. Logical tensor
payload bytes differ from allocator size classes. Ordinary benchmark counters
include tiny objects; the heap profile coalesces/samples tiny allocations and
does **not** report exactly 72 objects. Its Run-focused object listing is about
62.6 objects per invocation. The profile contains a warm invocation plus 1,000
timed invocations. Do not divide all profiles by 1,000 or convert profile samples
into an exact ordinary allocation census.

MemStats barriers between nodes alter tiny-object packing and include occasional
background allocations. The final census records about 72.006 objects and more
rounded tiny bytes than the ordinary benchmark. Those discrepancies are retained,
not assigned to invented mechanisms. The table's source-size prediction sums to
the independently observed ordinary counters, and removal controls confirm its
major boundaries. No unidentified ordinary counter remainder is needed to
balance this prediction; allocator attribution of individual tiny slabs remains
unresolved, as does physical retained/RSS overhead.

## Exact ordinary accounting prediction

| Purpose | B/Run | Objects/Run | Allocation site and object |
| --- | ---: | ---: | --- |
| Internal destination payloads | 90,336 | 9 | `tensor.newTensor`: nine `[]float32` backing arrays |
| Internal tensor metadata | 864 | 9 | `tensor.newTensor`: nine 96-byte `Tensor` headers |
| Internal tensor shapes | 240 | 9 | `tensor.newTensor`: six 32-byte and three 16-byte `[]int64` arrays |
| Internal storage identities | 9 | 9 | `tensor.newTensor`: `new(byte)`, amortized tiny allocation |
| Public output payload | 48 | 1 | `cloneOutput` → `Clone` → `FromFloat32` → `newTensor`; 40 logical bytes |
| Public output backend header | 96 | 1 | same clone, `tensor.Tensor` |
| Public output shape | 16 | 1 | same clone, `[1,10]` shape array |
| Public output identity | 1 | 1 | same clone, `new(byte)` |
| Public Tensor wrapper | 8 | 1 | `evaluation.cloneOutput`: ONNX `Tensor` containing backend pointer |
| Output publication slice | 24 | 1 | `evaluation.publish`: one-element `[]NamedTensor` |
| Evaluation | 96 | 1 | `newEvaluation`: heap-owned `evaluation` |
| Run budget | 128 | 1 | `newRunBudget`: limits/live/workspace object |
| Value reference table | 176 | 1 | `newRunStorage`: 22-element `[]*tensor.Tensor` |
| Symbol map header | 48 | 1 | `newEvaluation`: `map[string]int64`; no buckets for this fixed graph |
| Execution input arrays | 160 | 12 | `evaluation.execute`: eight 16-byte and four 8-byte pointer arrays |
| Conv validation shapes | 64 | 2 | `tensor.Conv2DShape`: four-element shape literals |
| Pool validation shapes | 64 | 2 | `tensor.Pool2DShape`: four-element shape literals |
| Reshape validation shape | 16 | 1 | `tensor.ReshapeShape`: copied destination shape at activation node 08 |
| MatMul plan | 80 | 1 | `tensor.newMatMulPlan`: `matMulPlan` |
| MatMul result shape | 16 | 1 | `tensor.newMatMulPlan`: rank-two result-shape array |
| MatMul staging | 48 | 1 | `tensor.matMulBatchesInto`: ten-float output staging array |
| Add matrix headers | 144 | 3 | `matrix.FromSlice` twice and `matrix.New`: 48-byte `Matrix` objects |
| Add matrix payloads | 144 | 3 | same adapters: three ten-float backing arrays, 40 logical bytes each |
| **Total** | **92,826** | **72** | Ordinary counters independently agree |

Internal destinations total **91,449 B / 36 objects**. Owned output publication
totals **193 B / six objects**. Run construction is **448 B / four objects**.
Execution arguments are **160 B / twelve objects**. Operator-internal metadata
and staging/adapters are **576 B / fourteen objects**. These five disjoint groups
also sum to the ordinary counters. `Clone` alone is 161 B/four objects;
`cloneOutput` including wrapper is 169 B/five objects.

## Destination details

Payload allocations use actual allocator size classes, confirmed by the total
destination-removal control and individual constructor measurements. Their
logical values are not additional copies. All shapes here are static; dynamic
models must choose actual current shapes.

| Node | Operator | Shape | Logical payload B | Allocator payload B | New private destination? |
| ---: | --- | --- | ---: | ---: | --- |
| 00 | Conv | `[1,8,28,28]` | 25,088 | 27,264 | Yes |
| 01 | Add | `[1,8,28,28]` | 25,088 | 27,264 | Yes |
| 02 | ReLU | `[1,8,28,28]` | 25,088 | 0 | Reuses node 00 after its last consumer |
| 03 | MaxPool | `[1,8,14,14]` | 6,272 | 6,528 | Yes |
| 04 | Conv | `[1,16,14,14]` | 12,544 | 13,568 | Yes |
| 05 | Add | `[1,16,14,14]` | 12,544 | 13,568 | Yes |
| 06 | ReLU | `[1,16,14,14]` | 12,544 | 0 | Reuses node 04 after its last consumer |
| 07 | MaxPool | `[1,16,4,4]` | 1,024 | 1,024 | Yes |
| 08 | Reshape | `[1,256]` | 1,024 | 1,024 | Yes; copied activation |
| 09 | Reshape | `[256,10]` | 10,240 | 0 | Prepared retained immutable weight reshape |
| 10 | MatMul | `[1,10]` | 40 | 48 | Yes, plus separate 48-B staging above |
| 11 | Add | `[1,10]` | 40 | 48 | Yes, plus separate 288-B adapter above |
| **New private payloads** | | | **83,664** | **90,336** | **Nine** |

`runStorage.destination` chooses a same-shape freed tensor or returns nil;
`cnnCall.run`, `numericalCall.run` and `shapeCall.run` perform `tensor.New` when
nil. The two successful reuse decisions are already in the baseline. The single
free slot discards incompatible shapes; no destination survives an ordinary Run.
Node 09 still resolves arguments and makes the same logical reservations, but
returns the plan-retained tensor without a numerical allocation/copy.

## Lifetime, ownership and necessity by semantic group

The following properties apply to every row in each named group above. “Required”
means the obligation must survive, not that the current heap object is the only
implementation. No proposed reuse below is a production implementation.

| Group | Lifetime / owner / reason | Payload or metadata; immutable? | Reusable? | Public ownership requirement | Dynamic shapes | Concurrency | Failure atomicity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Internal payloads | Invocation / evaluation storage / writable operator results | Payload; mutable | Two intra-Run reuses already occur; across-Run reuse measured only diagnostically | Private until final clone; final node's intermediate boundary is not inherently required by public ownership | Actual signature determines capacity/shape; no persistent shape cache exists | Must be private to the active session/Run; plan sharing prohibited | Disjoint private destinations prevent exposing partial Run results; fresh allocation each Run is not itself proved necessary |
| Internal headers/shapes | Invocation / each backend tensor / type, shape and slices | Metadata; shape/type immutable after construction, payload slice references owned mutable storage | Reusable with the same buffer and shape; different shapes need validation and accounting | Tensor constructor owns copied shape; no direct public output escape here | Current shapes determine header contents | Header/slices must not be concurrently mutated | Validation and destination identity must remain correct; rebuilding metadata is not intrinsically needed for every write |
| Internal identities | Tensor lifetime / tensor storage / alias detection | Metadata; identity immutable | Only with its corresponding buffer; never shared across independent storage | Establishes disjoint-storage contract, not an output clone | Independent of shape values | Distinguishes independent concurrent destinations | Required for shallow-copy/empty alias rejection; new identity per Run is not required if the same storage is safely retained |
| Output payload/header/shape/identity | Caller lifetime / returned backend tensor / independent output | Payload mutable through backend operations; shape and identity fixed | Must survive later Run and all Close calls; reuse by runtime prohibited while caller owns it | Required independent storage under current clone contract; transfer of a proven unique private buffer might avoid an intermediate clone, but was not implemented | Actual output shape always applies | Caller output cannot share mutable runtime workspace | Publication happens only after successful execution and final context checks; output ownership cannot be removed |
| Output wrapper/slice | Caller lifetime / returned ONNX values / names and API handles | Metadata; names reference immutable plan strings | Cannot mutate previously returned results for later calls | Required API representation; no string copy required | Output selection determines slice length | Distinct returned results must remain stable | Errors return no published partial result |
| Evaluation / budget | Invocation / Session.Run / context, plan, limits and counters | Mutable metadata | Embedding/reusing structures possible in principle; not measured as a correct implementation | Not required to escape to callers | Budget depends on actual sizes | Must remain invocation/session private | Current reservation ordering, limits, rollback/discard and error precedence remain required |
| Value table | Invocation / run storage / slot-to-current-tensor resolution | Mutable metadata | Topology static, values Run-dependent; retaining references requires cleanup | Does not own returned clones | Value shapes/inputs can change | Shared mutable plan table forbidden | Failed Run must not bind stale values into the next Run |
| Symbol map | Invocation / evaluation / consistent symbolic dimensions | Mutable metadata; empty in this fixed graph | Lazy creation is plausible, unmeasured as production | No direct output ownership obligation | Required when a symbol is bound; no buckets here | Bindings must not leak across concurrent sessions or calls | Conflicting symbols must still fail at the same checks |
| Execution input arrays | One node call / execute / ordered current input pointers including optional nils | Mutable metadata; lengths/topology static here, contents Run-dependent | Embedded Run-local array demonstrates twelve fewer objects; generic heap fallback retained | No public-output requirement | Topology need not change with dimensions; fallback call constructors still accept actual tensors | Reuse within synchronous Run possible; plan-owned mutable arrays forbidden | Must clear optional slots and preserve per-argument cancellation and validation order |
| Conv/Pool shape arrays | One backend call / backend validator / owned helper result used to validate destination | Immutable metadata for this call | Shape helper public ownership remains required; internal stack/prepared geometry is hypothetical | No public output payload obligation | Geometry depends on actual input and config | Stack/local scratch could isolate callers; no shared mutable geometry | Pre-write validation remains required, allocation itself is not a failure-atomicity requirement |
| Reshape validated shape | One backend call / ReshapeInto / validate equal count and destination shape | Immutable metadata | Public helper returns owned copy; internal temporary allocation potentially avoidable | Output remains disjoint under current backend contract | Validates actual counts and target each call | Local shape isolation required | All errors must preserve destination; copying shape is not itself required |
| MatMul plan/result shape | One call / MatMulInto / promotions, batches, checked dimensions and result shape | Immutable metadata | Could be temporary/prepared in a separate investigation | No public ownership requirement beyond destination shape validation | Vectors, broadcast batches, changing signatures require guards/fallback | Must never store mutable Run values in shared geometry | Overflow and geometry errors must occur before writes |
| MatMul staging | One call / matMulBatchesInto / complete product before commit | Mutable payload | Caller/local scratch possible only with a preserved contract; not implemented | Temporary, not returned | Full result size depends on actual signature | Must be private | Required by current all-returned-errors atomicity across batches/devices; host success-path diagnostic alone does not prove all build paths can omit it |
| Add matrix headers/payloads | One equal-shape Add / tensor-to-matrix adapter / copied inputs and matrix output | Mutable adapter payloads and metadata | Synchronous adapter is measured; no new design selected | Existing Matrix constructors/accessors promise copies; public ONNX output ownership does not require these intermediate copies | Equal-shape branch only; broadcasts already allocate zero | Temporary storage must be disjoint | Preserve Matrix/device and tensor alias/error contracts if ever revisited |

There are **no invocation allocation rows** for strides/broadcast metadata on
the two channel Adds, fixed operator output-index slices, immutable prepared call
geometry, initializer payloads, retained node-09 weights, cancellation/deadline
objects for `context.Background`, or successful-path error wrappers. Shape()
metadata copies in ONNX validation/destination checks do occur, but the measured
Go 1.27 compiler keeps these immediate-use results off the ordinary heap on this
path. The separately escaping Shape accessor benchmark allocates 32 B; it must
not be multiplied by call counts and added to the ledger.
