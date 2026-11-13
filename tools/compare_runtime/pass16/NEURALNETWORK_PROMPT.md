# Implement the NeuralNetwork-owned optimization identified by ONNX Pass 16

Work only in `github.com/itsmontoya/neuralnetwork`. This prompt authorizes the
scoped backend production work below, with maintainer review through a PR.
Do not modify the ONNX repository, integrate an unreleased dependency there,
release a version, or create a tag. You have no prerequisite conversation context;
the findings, constraints and return requirements are included here.

## 1. Objective

Independently validate and reduce avoidable allocation/copying at the **tensor
arithmetic to matrix boundary**, starting with equal-shape `tensor.AddInto`.
For two `[1,10]` float32 inputs and an existing disjoint `[1,10]` destination,
the backend allocates **336 B / 7 objects** to perform ten additions:
two copied input matrices, a new output matrix, and a copied `Values()` result,
followed by a final copy into tensor storage that already exists.

Inspect related host `tensor.MatMulInto([1,256], [256,10], [1,10])` planning and
output staging: **144 B / 3 objects**, despite an existing destination. Its
inputs already reach the host kernel without matrix copies. Establish whether
any of these costs can be removed while retaining the complete public contract,
especially destination atomicity on returned errors. Keeping necessary MatMul
staging is an acceptable evidence-based outcome; do not weaken its contract to
meet an allocation target.

This is a small allocation-focused backend problem, not a request to rewrite
kernels, add ONNX-specific special cases, reopen Conv/ReLU, change tolerances,
or pursue later optimization candidates. ONNX has stopped production optimization
pending this handoff. No backend production change or workaround was implemented
by the ONNX investigating agent.

## 2. Measured evidence and its limits

### Provenance and measurement policy

Measured 2026-11-12 on Apple M3, Mac15,13, 8 cores (4 performance / 4 efficiency),
24 GB; macOS 26.5.2 (25F84), Darwin 25.5.0, darwin/arm64, Go 1.27.2.
ONNX HEAD: `0803d3ea12ae02597592d014e4792c95ec5ca364`.
Released backend: **v0.10.9**, merge commit
`740a244bf56214bb3443a69287a2d800674620e4`, module sum
`h1:kmfxkbiZZ8fbw0cxCxvar2nfYJ2Bx6kci/QUNH0w6oQ=`.
Other module versions: simd v1.3.0, x/sys v0.45.0, protobuf v1.36.12.
Use the appropriate current backend base and first determine whether subsequent
changes already address the mechanism; these historical numbers are not a reason
to revert newer work.

Latency builds used `GOMAXPROCS=1 CGO_ENABLED=0 GOWORK=off GOARM64=v8.0`, default
optimization, no tags/gcflags. Each benchmark ran 200 ms, seven repetitions,
sequentially. Tables show medians and full ranges in microseconds. Allocation
profiles used separate 10,000-call runs with `-memprofilerate=1`. Escape builds
used separate `-m=2` compilations. Neither profiling nor race instrumentation
was present in latency measurements.

Backend microbenchmarks used deterministic synthetic finite float32 values:
`float32((index+offset)%29-14)*0.125`, offset 3 for left, 7 for right. Inputs and
disjoint destinations were built outside timing. The private adapter diagnostics
use the same checked dimensions as the public calls. These results are separate
from the ONNX adapter results on actual MNIST intermediates below.

### Add measurements: `[1,10] + [1,10]`

| Scope | Median µs | Range µs | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: |
| Complete `tensor.AddInto` | 0.176100 | 0.172100–0.207200 | 336 | 7 |
| `matrixBinaryInto`, excluding outer tensor validation | 0.101500 | 0.098290–0.102800 | 336 | 7 |
| One `matrix.FromSlice(1,10,values)` | 0.026200 | 0.024720–0.026440 | 96 | 2 |
| One `matrix.New(1,10)` | 0.024540 | 0.023230–0.024720 | 96 | 2 |
| `Matrix.AddInto` on existing owned matrices | 0.020220 | 0.020060–0.020550 | 0 | 0 |
| `Matrix.Values()` | 0.016590 | 0.015960–0.016740 | 48 | 1 |
| Existing `Matrix.ValuesInto(existingSlice)` | 0.005811 | 0.005798–0.005826 | 0 | 0 |
| Four copies into existing ten-element buffers | 0.005786 | 0.005777–0.005868 | 0 | 0 |
| Direct ten-element addition, arithmetic ceiling | 0.005684 | 0.005664–0.005699 | 0 | 0 |
| Tensor validation + broadcast shape + destination validation + direct addition | 0.079250 | 0.078450–0.093400 | 0 | 0 |

The last two rows are **diagnostic ceilings**, not a proposed production path.
The validated Add ceiling preserves input/type/storage, broadcast shape and
destination validation for this fixture, but changes arithmetic dispatch and
does not establish large-vector, SIMD, device or general numerical equivalence.
Complete/validated CVs were 6.37% / 6.24%; retain all samples. The difference of
medians is only **0.096850 µs**. Do not promise that exact saving.

Allocation ledger for one complete equal-shape Add:

| Allocation site in v0.10.9 | Allocator bytes | Objects |
| --- | ---: | ---: |
| `matrix/matrix.go:38,52`, two `FromSlice` structs and copied input buffers | 192 | 4 |
| `matrix/matrix.go:21,30`, output `New` struct and buffer | 96 | 2 |
| `matrix/matrix.go:275`, `Values` copied buffer | 48 | 1 |
| **Total** | **336** | **7** |

Each Matrix is 48 bytes; each ten-float buffer requests 40 bytes and occupies
the 48-byte allocator class. The filtered 10,000-call profile attributes exactly
40,000 objects / 1,920,000 B to `FromSlice`, 20,000 / 960,000 B to `New`, and
10,000 / 480,000 B to `Values`: **70,000 objects / 3,360,000 B**.
Outer Add shape validation has zero allocations at this rank in v0.10.9.

Source-derived logical copying is **160 B per Add**: two 40-B input copies,
40 B from output matrix to `Values`, and 40 B from `Values` to the tensor
destination. That is 320 B of copy read/write traffic, plus at least 120 B of
useful addition traffic. Zero initialization and metadata accesses are additional.
These are cached logical bytes, not a measured DRAM bandwidth claim.
The four-copy microbenchmark excludes allocation, validation and arithmetic;
its buffers are synthetic and its latency cannot be subtracted as an independent
whole-call speedup. Most of the adapter cost is not raw payload copying.

### MatMul measurements: `[1,256] × [256,10] → [1,10]`

| Scope | Median µs | Range µs | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: |
| Complete `tensor.MatMulInto` | 1.681000 | 1.643000–1.789000 | 144 | 3 |
| `newMatMulPlan` | 0.063140 | 0.062650–0.063500 | 96 | 2 |
| `matMulBatchesInto` with existing plan, including staging | 1.564000 | 1.556000–1.578000 | 48 | 1 |
| Host `matrixProductInto` into existing storage | 1.497000 | 1.492000–1.526000 | 0 | 0 |
| One ten-float copy into existing storage | 0.001843 | 0.001833–0.001852 | 0 | 0 |
| Input/destination validation + direct product with pre-resolved geometry | 1.577000 | 1.571000–1.592000 | 0 | 0 |

The final row **omits shape-plan construction, signature guards, batching and
staged error handling**, retaining only tensor validation and destination
validation with the already resolved shape. It is a fixed-signature diagnostic,
not a complete-contract candidate. Complete/validated CVs are 3.19% / 0.49%.
Difference of medians: **0.104000 µs**. Most of the ~1.68-µs complete call is
still arithmetic. Do not equate an older coarse ORT gap with removable staging.

Allocation sites:

| Site | Allocator bytes | Objects |
| --- | ---: | ---: |
| `tensor/matmul_plan.go:7`, returned 72-byte plan, rounded | 80 | 1 |
| `tensor/matmul_plan.go:41`, two-dimension result-shape backing array | 16 | 1 |
| `tensor/matmul.go:79`, ten-float call-owned staging output | 48 | 1 |
| **Total** | **144** | **3** |

Filtered 10,000-call profile: `newMatMulPlan` **20,000 objects / 960,000 B**;
`matMulBatchesInto` **10,000 / 480,000 B**. There are no Matrix objects or copied
input payloads on the measured host path. The final output copy transfers **40 B**
(80 B logical read/write traffic). Batch prefixes are empty in this fixture;
zero-length stride helpers add no heap allocations. General batched/high-rank
cases require separate measurement and validation.

### ONNX measurements and history

Fresh ONNX numerical adapters on real ramp intermediates, excluding session
dispatch, destination allocation, retirement and publication:

| Node and shapes | Median µs | Range µs | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: |
| 01 Add `[1,8,28,28] + [8,1,1]` | 1.929000 | 1.876000–2.167000 | 0 | 0 |
| 05 Add `[1,16,14,14] + [16,1,1]` | 1.177000 | 1.169000–1.194000 | 0 | 0 |
| 11 Add `[1,10] + [1,10]` | 0.195800 | 0.194600–0.198200 | 336 | 7 |
| 10 MatMul `[1,256] × [256,10]` | 1.690000 | 1.667000–1.695000 | 144 | 3 |

Complete ordinary-ramp inference: **55.841 µs**, range 55.514–56.453, CV 0.48%,
**92,874 B/op / 73 allocs/op**. This is a baseline only, not a before/after or
a fresh Go/ORT comparison. The Add adapters sum to 3.3018 µs; most is broadcast
arithmetic outside this task. Argument gathering allocates a separate 16-B
two-pointer slice per binary node in full inference; those allocations are
excluded from the table and are not backend adapters.

Relevant established history: published v0.10.9 removed broadcast metadata
allocations in Pass 15: three Adds **784 B / 22 → 336 B / 7**; complete inference
**93,322 B / 88 → 92,874 B / 73**. The first two channel Adds now allocate zero.
The remaining equal-shape allocations are value-dependent temporary adapters,
not immutable broadcast metadata. Do not attribute the old 15-object saving to
this new work. Earlier published fixed-policy Go/ORT results were 54.750 /
42.375 µs, with coarse profiled MatMul ~1.004 µs; different scopes and sampling
prevent treating their difference as a staging budget.

Even eliminating all ten observed Add/MatMul allocations would remove only
**480 B / 10 objects per inference**, conditionally 92,394 B / 63 objects.
That is an allocation ceiling, not an achieved or required outcome. The two
validated diagnostic differences sum to only **0.200850 µs**, with overlapping
validation/dispatch effects and missing general contracts. No complete-runtime
transfer, fixed-policy speedup, retained-memory reduction or initialization
improvement has been demonstrated for a new implementation.

### Escape analysis and ownership findings

`-m=2` and ARM64 disassembly establish:

- `matrix.New` and `FromSlice` return local `m` by address; `m` and their data
  allocations escape. Constructors are not inlined in this build. Matrix
  `Values` allocates returned storage. The adapter really calls those constructors,
  the indirect operation function and `Values`, then copies to the tensor.
- `matrixBinaryInto`'s tensor arguments and operation function do not escape.
  There is no successful-path closure allocation or interface-boxing explanation
  for the seven objects. Error-formatting allocations occur on error branches.
- `newMatMulPlan`'s returned local plan and result-shape array escape. The
  dimensions are copied, not payload data. No input tensor is deliberately
  retained in a reusable plan here; this plan is reconstructed per call.
- MatMul's output `make([]float32, len(destination.floats))` is reported as
  **non-escaping**, yet the measured call allocates 48 B and generated code
  contains `runtime.makeslice`. Non-escaping is not proof of no heap allocation
  for a runtime-sized slice. Its two literal matrix-block shape slices do not
  escape and are not extra objects in this fixture.
- All ten observed objects are invocation-lived. A new shared cache or retained
  value buffer would introduce ownership/concurrency/memory obligations; it is
  not an established solution.

Rejected or unproved explanations/options:

- ONNX is not converting these tensors to matrices: the copies originate inside
  backend `tensor/arithmetic.go`. Duplicating loops or copying through ONNX's
  public value accessors would not fix backend ownership.
- Host MatMul does not copy the 10,240-B weight through matrices. It already uses
  `internal/float32product`; matrix views cannot remove nonexistent host input
  adapters. The Metal-tagged product path is different.
- The remaining Add objects are not shape snapshots, escaping coordinates,
  boxed contexts, or immutable preparation metadata.
- `Matrix.ValuesInto` already exists and avoids one copied `Values` allocation.
  It is a legitimate smaller backend alternative to investigate, not an ONNX
  fix and not evidence that a new public view API is required.
- A borrowed Matrix view or shared slice kernel might avoid more adapters, but
  neither API design nor full numerical/device equivalence has been demonstrated.
- MatMul staging supports all-errors atomicity; simply deleting its final copy
  without establishing all failure points before writes is insufficient.
- Pooling, unsafe field access and per-model value caches are not justified by
  this evidence. Prior static Add descriptors concern different planning costs
  and are not necessary to recover these value-dependent copies.

Original evidence is preserved in ONNX at `tools/compare_runtime/pass16/`:
`diagnostic_test.go.txt`, replay scripts, `data/summary.json`,
`data/validation.json`, and lossless `data/evidence.json.gz` (benchmark logs,
profiles by site, compiler escape output and disassembly). You do not need those
files to understand or independently reproduce the mechanism described here.
Do not edit ONNX to use them. Performance diagnostics used a byte-identical copy
of released backend source and a test-only overlay, with no production patch,
module replacement or edits to the actual backend repository.

## 3. Exact backend path and ONNX entry point

All backend paths below are relative to the NeuralNetwork repository, at v0.10.9:

- `tensor/arithmetic.go`: `AddInto` → `binaryFloat32Into` → equal-shape
  `matrixBinaryInto` → `matrix.FromSlice` twice → `matrix.New` → operation
  `func(*matrix.Matrix,*matrix.Matrix,*matrix.Matrix) error`, specifically
  `(*matrix.Matrix).AddInto` → `output.Values` → `copy(destination.floats,values)`.
  Sub/Mul share the adapter; Div does not use it. Preserve shared callers.
- `tensor/destination.go`: `validateTensorType`, `validateDestination`.
  `tensor/broadcast.go` / `arithmetic.go`: shape and stride helpers, common-rank
  stack scratch with arbitrary-rank fallback.
- `matrix/matrix.go`: `Matrix{rows,cols,data,residency}`, `New`, `FromSlice`,
  `AddInto`, `Values`, `ValuesInto`, `WithRowView`. Fields are private.
  `WithRowView` borrows from an already owned Matrix; it cannot wrap Tensor
  storage. The public copy contracts of existing constructors/accessors matter.
- `matrix/elementwise_arm64.go`, `elementwise_amd64.go`, `elementwise_default.go`,
  `elementwise_pure.go`: dispatch and arithmetic. ARM64 Add below 16 values
  uses the portable loop; larger arrays use simd32. Inspect current architecture
  selection before replacing a matrix call with a tensor-local scalar loop.
- `matrix/residency.go`, `matrix/execution.go`, `matrix/forward_device_default.go`
  and `backward_device_metal.go`: execution inheritance, host/device synchronization,
  mutation and device dispatch. Measured adapters have nil residency; public
  matrix device behavior is broader than that fixture.
- `tensor/matmul.go`: `MatMulInto` → `newMatMulPlan` → destination validation →
  `matMulBatchesInto` → batch stride/block planning → complete output staging →
  `matrixProductInto` for each batch → final destination copy.
- `tensor/matmul_plan.go`: `matMulPlan` and vector/batch/result-shape validation.
- `tensor/matrix_product_host.go` (host/purego/CGO-disabled build) →
  `internal/float32product/product.go`: `Standard`, transpose variants.
  Host `matrixProductInto` returns nil after synchronous validated arithmetic.
- `tensor/matrix_product.go` (`metal && !purego && darwin && cgo`): copied
  Matrix adapters, matrix products and host values, with possible backend errors.
  Shared Gemm callers need review if product helpers change.

ONNX currently calls these directly:
`Session.Run` → `evaluation.execute` → `preparedCall.run` (or dynamic
`newNumericalCall`) → `numericalCall.run` → `tensor.AddInto` / `MatMulInto`.
`preparedCall` copies immutable geometry onto each invocation's stack, supplies
current arguments and obtains a private destination; ONNX does not own a Matrix.
`numericalCall` stores step, arguments, output shape, payload bytes and workspace
charge. Tensor shape/value fields are private to the backend; ONNX's immutable
Tensor facade exposes copies, not writable slices.

## 4. Semantic contract that must survive

**Ownership and mutation.** Backend tensors own contiguous row-major storage and
shape metadata. Existing constructors, `Shape`, value accessors and `Clone`
return/capture copies. Inputs are read-only during an operation, destination is
fully overwritten on success. Do not mutate, retain, steal or publish aliases
to input/initializer storage. ONNX outputs are independent owned copies and
survive later Runs and model/session closure. Backend callers may update weights
between calls; do not cache their numerical contents by shape or pointer.

**Aliases and destinations.** Tensor destination must be disjoint from both
inputs. `identity` detects pointer and shallow-copy aliases, including empties.
The same tensor may be both left and right if destination is disjoint. Matrix
elementwise APIs separately allow an exact destination/input alias. Do not
import that looser rule into Tensor. Any proposed borrowed-slice API must define
overlap and enforce/reject unsupported partial aliases. Caller-supplied destination
storage and shape remain owned by the caller; do not replace or resize them.

**Validation and errors.** Preserve nil/zero/invalid Tensor rejection, exact
float32 type, shape/storage length checks, dimension/product/byte/host-int
overflow checks, broadcasting compatibility and exact destination shape.
Add validates left, right, broadcast shape, destination and identities before
the empty-result return or any write. MatMul validates left, right, plan,
destination and identities before empty/zero-contraction handling. Preserve
error precedence and meaningful wrapping. Add promises argument errors leave
destination unchanged; **MatMul promises every returned error leaves destination
unchanged**, across all batches and supported backends. Do not silently relax
either guarantee to remove a staging buffer.

**Shapes and arithmetic.** Add includes equal shapes, scalars, arbitrary ranks,
bidirectional broadcasts, zero/empty outputs and general stride fallbacks.
MatMul promotes vectors, broadcasts batch prefixes and returns a scalar for
vector-vector products; scalar operands fail. Empty contractions write positive
zeros. Preserve float32 accumulation order and existing default/purego numerical
behavior, operand order, NaNs/infinities/signed zero, and architecture dispatch
semantics. Do not change tolerances to accept different arithmetic. Shared Sub,
Mul, Div and Gemm behavior must remain correct if common code changes.

**Dynamic shapes and overrides.** Backend entry points accept each call's actual
concrete shapes. ONNX prepares only fixed validated signatures; symbolic/unknown
dimensions use the dynamic path. Binding and produced-value checks enforce
original graph declarations before prepared consumers run. Default inputs can
be overridden, with current values even when shapes match. Any optional prepared
API must own its geometry, validate signatures on use, reject or fall back for
changed shapes, and never learn/cache payloads as model constants.

**Cancellation and deadlines.** Add/MatMul backend APIs have no context argument
and currently run synchronously without polling. ONNX checks context around
numerical execution and copies, per argument/node, and before publication;
`context.Canceled` / `DeadlineExceeded` and failed-Run recovery must survive.
Do not add asynchronous completion or context ownership. A new API requiring
ONNX adoption must specify where it fits without removing those checks.

**Concurrency and lifetime.** Independent ONNX sessions share immutable plan,
weights and input readers, with invocation-private destinations/state. One
session rejects overlapping Run/Close with ErrBusy; independent sessions run
concurrently. Backend arithmetic with shared read-only inputs and disjoint
outputs must remain race safe. No global mutable geometry/buffer cache or
non-thread-safe shared descriptor. A borrow, if chosen, must be synchronous,
nonretaining, valid only while its owner lives, and safe on errors/panics. Existing
Matrix row views expire at callback return/panic and must not be retained,
mutated or shared across goroutines; this does not establish a new view contract.

**Budgets and retained state.** ONNX currently conservatively charges four
output-sized Add scratch buffers for equal shapes (160 B for node 11), and one
output-sized host MatMul staging buffer (40 B here; none for empty contractions).
Other build paths charge differently. These logical workspace charges are not
allocator-rounded B/op. Keep ONNX untouched; its conservative existing charges
can stay until a separate released integration. Any additional persistent state,
numerical payload, scratch requirement or initialization work must be measured
and reported so that later ONNX load/live/workspace accounting remains sound.

**Builds and devices.** Preserve purego, CGO-disabled, supported 32/64-bit targets
and race safety. Host MatMul and Metal MatMul are distinct build paths. Proving
host kernels infallible after validation does not prove that device submissions,
synchronization or later batch operations cannot fail. Retain conservative
fallback where failure atomicity/device behavior is not proved.

## 5. Desired investigation before implementation

1. Read current `AGENTS.md`, `STYLEGUIDE.md`, go.mod, CI and PR template. Inspect
   the actual current source and validate this diagnosis before changing production.
2. Establish a backend baseline on the current base, including complete public
   Into calls, the isolated adapter, Matrix allocation/copy methods and host
   MatMul plan/staging/kernel scopes. Use the shapes/value formula above. Keep
   destination creation outside timing; do not substitute a stripped kernel for
   complete-contract performance. Record raw samples, range/CV, µs/op, B/op and
   allocs/op. Confirm allocation sites and generated code where needed.
3. Review existing `ValuesInto` as a smaller possibility. A shared internal slice
   kernel or a general borrow/view boundary are hypotheses, not prescribed APIs.
   Compare contract/maintenance cost with the small absolute opportunity.
   Do not add a public view just because it sounds zero-copy.
4. Map every possible failure before considering direct MatMul destination writes,
   including batching, integer/stride validation and Metal. A host-only internal
   improvement with unchanged fallbacks may be appropriate; independently prove it.
5. Implement the smallest justified backend change and explicit tests. If current
   source already resolves the issue, document/measure that outcome instead of
   inventing a new optimization. Keep hypotheses and achieved improvements distinct.

## 6. Acceptance criteria and validation

- Provide same-host, same-toolchain/backend-policy **before/after µs/op, B/op,
  allocs/op** for complete equal-shape Add and the related MatMul paths; measure
  isolated adapters/copies separately. Retain all valid samples, use repeated
  comparisons with order balanced where useful, and explain variance. Demonstrate
  a reproducible benefit on the selected path with no material supported-path
  regression. No required latency percentage is inferred from these ceilings.
- Cover small/large equal shapes, scalar/empty/zero dimensions, broadcast shapes
  above, higher ranks and SIMD boundaries (at least lengths 1, 10, 15, 16, 17 and
  larger arrays). Preserve zero allocations for the two existing channel Adds.
  Review shared Sub/Mul and any changed device paths.
- For MatMul, cover vectors/dot products, batch broadcasting, changing dimensions,
  zero inner dimension/empty output, overflow and mismatches, prefilled
  destinations, repeated calls and mutable weights. Verify no partial writes
  on every returned-error path, including a later-batch/device failure when
  applicable. If this cannot be established for a path, keep its staging.
- Verify read-only inputs bitwise unchanged, repeated aliased inputs allowed,
  destination identity/shallow-copy aliases rejected even for empties, independent
  owned outputs, lifetime after source references are dropped, concurrent shared
  readers/disjoint writers, no unexpected retention. If a new borrow/API is
  needed, add explicit ownership/overlap/panic/lifetime/race tests for it.
- Use existing correctness suites plus focused regressions. Relevant v0.10.9 tests:
  `tensor/arithmetic_test.go` (`Reference`, `ArgumentErrors`, `Aliases`,
  `EmptyUnusedStrides`, `BroadcastCarries`, `ConcurrentAndOutputLifetime`),
  `broadcast_add_test.go` (`GenericBroadcast`, `SpanArgumentErrors`,
  `ConcurrentSpans`, `MetadataRankBoundaries`), and `products_test.go`
  (`Reference`, `ProductShapes`, `ArgumentErrors`, `AliasesAndEmptyContractions`,
  `EmptyUnusedBatchSizes`, `ExistingMatrixKernels`, `DynamicWeightsAndConcurrency`).
  Run relevant Matrix/device tests when affected. Include IEEE special values.
- Run `gofmt`, repository formatting checks, `go vet ./...`, `go mod tidy -diff`,
  `go build ./...`, `go test ./...`, `CGO_ENABLED=0 go test ./...`,
  `CGO_ENABLED=0 go test -tags=purego ./...`, `go test -race ./...`, and
  `go test -race -tags=purego ./...`; apply additional current repository CI rules.
  Run supported Metal validation on an appropriate host if affected, or report
  the exact limitation rather than claiming unexecuted coverage.
- Cross-build default and purego with CGO disabled for darwin/arm64,
  darwin/amd64, linux/arm64, linux/amd64, windows/amd64 and linux/386, subject
  to the repository's actual supported targets. Record compilation separately
  from native execution. Example: `CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build ./...`.
- Measure added/removed retained bytes and objects, payload, constructor time,
  initialization allocations and any warmup/amortization cost. Report zero when
  established by design/measurement; do not equate lower B/op with lower retained
  memory or invent a measured RSS result.

The ONNX investigating agent ran fresh complete ONNX suites with CGO disabled,
purego, race and purego+race, plus vet/module verification. All passed. Test-only
ceilings matched complete finite-fixture output bits and left inputs unchanged
in those four modes. These results validate the investigation, not your future
backend implementation. You must run your own complete backend validation.

## 7. API requirements

Prefer keeping `tensor.AddInto` / `MatMulInto` unchanged if internal ownership
and dispatch changes can solve the problem. No new public API is yet proved
necessary. Existing public copy constructors/accessors must retain copy semantics.

If a public API is required, ONNX needs synchronous computation using its
**existing owned backend Tensor inputs and existing disjoint destination**,
without forcing temporary owned matrices or copied values. Optional caller-owned
immutable geometry must be reusable with current values, have precise signature
validation and lifetime rules, and remain safe for concurrent sessions. Specify
scratch/error atomicity and any persistent or device resources. A general backend
API is preferred; avoid model-, shape-, MNIST- or ONNX-specific hooks. Do not expose
unchecked strides or unsafe writable storage merely to bypass validation.

## 8. Repository workflow

1. Work only in `github.com/itsmontoya/neuralnetwork`. Inspect status; preserve
   unrelated work. Choose the appropriate current base after inspecting the
   repository/default branch, then create a new branch such as
   `optimization-pass-16-matrix-adapters`. Record the base commit.
2. Keep generated evidence under `.benchmarks/<topic>/<distinct-run>/` per the
   backend's instructions. Put reusable benchmarks/tests alongside their packages.
   Preserve raw data and provide a durable reviewable summary, since ignored
   artifacts alone are insufficient for a reviewer.
3. Make small, concise commits, scoped to the measured mechanism; follow the
   backend style guide, including file-per-type and constructor rules, named
   returns with explicit return expressions, grouped declarations and no shadowing.
4. Complete the validation above and required current CI checks.
5. Push the branch and create a PR using `gh`. Follow the existing
   `.github/PULL_REQUEST_TEMPLATE.md` exactly, keeping section order, required
   fields and checklist items. The backend's instruction for generated PR
   descriptions requires the entire description in a fenced Markdown block,
   with no explanatory text outside it. Follow the actual current instructions;
   for multiline gh input, write the exact body to a file and use `--body-file`.
6. Report the resulting PR and commits. Do not merge, publish/release, or tag
   unless separately instructed. Do not modify ONNX or use it as a temporary
   integration sandbox. ONNX adoption waits for review, merge and a published
   backend version in a separate step.

## 9. Return contract

Your final report must contain:

- Base and branch, commit SHAs, PR URL, files changed and the selected mechanism.
- Public API changes, if any, with signatures, ownership/mutation/alias rules,
  signature guards, destination/error guarantees and build/backend applicability.
- Backend before/after tables with **µs/op, B/op and allocs/op**, full measurement
  policy, ranges/variance, raw evidence location and regression results. Label
  kernel ceilings separately from complete public operations. Explicitly state
  when MatMul staging is retained and why.
- Correctness, ownership/alias/dynamic/lifetime/concurrency test results; race,
  purego, CGO-disabled and supported cross-build results; unexecuted checks and
  known limitations. Do not claim ONNX whole-Run improvement from backend probes.
- Retained-memory changes (bytes/objects/payload), any new initialization cost,
  scratch needs, warmup cost and device lifetime obligations.
- **Exact later ONNX integration instructions**: whether a dependency-only update
  suffices, or exact API use needed in `numerical_call.go` / prepared descriptors;
  which shape guards/fallbacks/context boundaries to preserve; and whether
  `numerical_product_host.go` / Metal workspace accounting could be revised.
  Existing conservative charges can remain if sound; do not require premature
  quota relaxation. State that the next ONNX agent must consume a real published
  version, without local replacements.
- A transfer-validation checklist for that later ONNX agent: fresh component
  measurements, complete Run B/op and allocs/op, unchanged fixed-policy numerical
  checks and Go/ORT timings, output/initializer bits and ownership/lifecycle,
  dynamic/override/alias tests, cancellation/deadlines, quota bounds, races,
  purego/CGO-disabled suites and supported cross-builds.

Do not stop after a proposal when a safe scoped implementation is demonstrated.
Complete the backend work through a reviewable PR, and give ONNX enough precise
information to resume after a published version exists.
