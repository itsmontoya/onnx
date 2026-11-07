# neuralnetwork dependency handoff

This records the implementation requirements for
[`github.com/itsmontoya/neuralnetwork`](https://github.com/itsmontoya/neuralnetwork)
to support a Go-native ONNX inference runtime. Tasks in this file belong in that
library. Runtime tasks live in [ROADMAP.md][onnx-roadmap] in the separate
[`github.com/itsmontoya/onnx`](https://github.com/itsmontoya/onnx) repository.
`NN-*` identifiers refer to tasks here; `RT-*` identifiers refer to that roadmap.
The original task definitions are retained below; current status and published
handoffs take precedence over their original proposed API names.

## v0.7.0 status

Reviewed on 2026-11-06 against published [`v0.7.0`][backend-release], commit
[`6048452c57a0a293db4591cd2e2febd7d6b743a3`][backend-commit]. The GitHub tag
was verified, and all 87 published tensor, handoff, tensor-design, and module
files matched the adjacent source checkout. NN-01–NN-12 are implemented.
**NN-13 optimization is not implemented**, as recorded in [PR #114][nn13-pr] and
[its tracked handoff][nn-13]. RT-01–RT-09 now implement the runtime contract
through serial M1 sessions, including broadcast arithmetic, products, activations
and exact float/int Constant forms.
RT-10 adds offline typed conformance fixtures and retains the backend reference
differences as documented extensions or explicit future-adapter rejections;
see [the fixture guide](testdata/conformance/README.md). M3 and reduction runtime support remain deferred. RT-12 implements the M2
shape adapters with the reference exceptions below.
RT-11 demonstrates this working M1 runtime with original MLP and two-input/
two-output branching models at batches 1 and 3. The [integration report](docs/integration-rt11.md)
records reference comparisons, provenance/hashes, lifecycle/concurrent-session
checks and exact supported restrictions. RT-12 adds a shape-heavy graph at both batch sizes and supplies the
[accessible runtime handoff](docs/handoff-nn13.md) needed to resume NN-13.

The library's `NEEDS.md` is ignored and is not included in the release tree.
Use the published [tensor contract][tensor-contract], source, tests, and tracked
handoffs below rather than a link to `neuralnetwork/blob/main/NEEDS.md`.

| Task | Status in v0.7.0 and tracked handoff | Runtime consumer |
| --- | --- | --- |
| NN-01 | [Complete: additive tensor contract][nn-01] | RT-01 |
| NN-02 | [Complete: owned float32/int64 tensors][nn-02] | RT-04–RT-05 |
| NN-03 | [Complete: checked shapes, axes, broadcasting][nn-03] | RT-04, RT-07 |
| NN-04 | [Complete: broadcast arithmetic][nn-04] | RT-09 |
| NN-05 | [Complete: MatMul/Gemm][nn-05] | RT-09 |
| NN-06 | [Complete: activations and axis Softmax][nn-06] | RT-09 |
| NN-07 | [Complete: copying tensor transforms][nn-07] | RT-12 |
| NN-08 | [Complete: Concat/Gather][nn-08] | RT-12 |
| NN-09 | [Complete: NCHW convolution][nn-09] | RT-13 |
| NN-10 | [Complete: pooling][nn-10] | RT-13 |
| NN-11 | [Complete: stateless batch normalization][nn-11] | RT-13 |
| NN-12 | [Complete: reductions and math][nn-12] | RT-19, when selected |
| NN-13 | [Blocked: runtime measurements and optimization][nn-13] | After RT-12; before RT-16 |

The module declares Go 1.26.1 and depends on `github.com/tphakala/simd v1.3.0`
and indirect `golang.org/x/sys v0.45.0`. Upstream handoffs report checks on
Go 1.27.2, Darwin/arm64. NN-12 records passing vet, normal, cgo-disabled
`purego`, and race tests; PR #114 changes only documentation. Those checks were
not rerun for this planning update, and they do not validate an ONNX runtime.
RT-02 must verify the selected toolchain against the full dependency graph.

### Integration constraints

| Boundary | Released behavior and required runtime work |
| --- | --- |
| Ownership | Constructors, shapes, typed accessors, and `Clone` copy. `Into` kernels require separate destination storage, including shallow copies and empty tensors. All transforms copy; no public borrowed view or workspace exists. RT-04/RT-08 must preserve immutable constants and owned output lifetimes. |
| Limits and cancellation | Checked sizes establish representability, not a memory budget. Matrix/activation adapters and MatMul allocate call-local copies. Kernels are synchronous and have no context cancellation API. RT-01/RT-08/RT-15 must account for temporary storage and state the actual cancellation granularity. |
| Activations | [NN-06][nn-06] documents ReLU mapping NaN to positive zero and empty-axis Softmax succeeding where the pinned evaluator rejects it. RT-01/RT-09 must resolve and test these compatibility cases before advertising coverage. |
| Shape and CNN edge cases | [NN-07][nn-07], [NN-08][nn-08], and [NN-09][nn-09] record empty-shape/reference differences. Schema defaults, reshape `0`/`-1`, omitted axes, automatic padding, and accepted ONNX geometry remain runtime responsibilities. |
| Pooling | [NN-10][nn-10] records ceil-window, NaN, empty/all-padding, and reference disagreements. `CountIncludePad` uses the full kernel area, including partial ceil windows. Max indices/storage-order variants are absent. RT-13 must validate its supported subset and retain explicit reference differences. |
| Normalization | [NN-11][nn-11] preserves supplied inference statistics. The ONNX 1.16.2 schema-9 evaluator applies momentum and differs on 23 fixtures; ORT 1.19.2 agrees on 30 of 31, with a zero-variance/zero-epsilon difference. RT-10/RT-13 need independent stored-statistics expectations. |
| Expansion | [NN-12][nn-12] uses resolved mathematical axes; empty axes mean identity. Its reduction reference models use opset 18, while unary math/Pow models use opset 13. Kernel availability does not expand this runtime's proposed opset-13 coverage. |

Fixture versions, tolerances, hashes, and actual reference exceptions are recorded
in the [published fixture provenance][backend-fixtures]. A reference rejection
or disagreement must not count as a passing compatibility case.

### RT-12 inputs for NN-13

RT-11 and RT-12 use the released copying backend and provide the
[runtime source/fixture handoff](docs/handoff-nn13.md), including reproducible
MLP and shape-heavy graphs, named inputs/expected outputs and batches 1 and 3. NN-13 can then measure the unchanged
baseline, select a justified bridge/workspace change, and report before/after
latency, allocations, copied bytes, and peak live memory with ownership/race
checks. Its [handoff][nn-13] defines the full measurement plan.

NN-13 does not require RT-16 to finish first. RT-16 consumes NN-13's results;
neither optimization task blocks M1–M4. No optimization or benchmark result is
claimed by v0.7.0's NN-13 blocker report.

## Original review baseline (before v0.7.0)

Reviewed on 2026-11-06 against the local library checkout at commit
[`e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1`](https://github.com/itsmontoya/neuralnetwork/tree/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1),
dated 2026-10-02. Its tracked files were clean. The untracked
`CurrentRoadmap.md` was excluded from the implementation baseline. GitHub's
repository page was also inspected, but remote HEAD could not be verified from
this environment. Recheck the library revision before implementing these tasks.
This was a source and contract review, not a claim that its test suite was run.
The gaps described in this historical review motivated the tasks below; use
the v0.7.0 handoffs above for current implementation status.

The library declares Go 1.26.1 and a SIMD dependency. Its existing `purego` build
tag selects a CPU path without the external SIMD wrappers or Metal. The runtime
should work with `CGO_ENABLED=0`; optional assembly acceleration can remain an
implementation detail. Native Go does not require a standard-library-only module.

### What already exists

Links below identify source at the reviewed revision. Inspect implementation and
matching tests before replacing anything.

| Area | Existing implementation | Consequence for this project |
| --- | --- | --- |
| Numeric storage | [`matrix/matrix.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/matrix/matrix.go): owned, row-major `float32` matrices; copying constructors/accessors; destination operations; scoped row views | Reuse arithmetic and ownership patterns. This is a positive-size rank-two container, not a general tensor. |
| Linear algebra | The same file provides `MatMulInto`, left/right transpose variants, transpose, scalar operations, and row/column sums | A small rank-two MLP adapter is possible today. Batched tensor multiplication and general broadcasting are missing. |
| Activations | [`activation/`](https://github.com/itsmontoya/neuralnetwork/tree/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/activation): ReLU, sigmoid, tanh, exact GELU, other activations, destination APIs, row and segmented softmax | Reuse formulas. Row softmax does not cover arbitrary tensor axes. |
| CNNs | [`layer/conv2d.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/layer/conv2d.go), convolution/pooling configs, `MaxPool2D`, and `Flatten` | Existing flattened NCHW processing is useful, but convolution has symmetric padding and no groups/dilation; pooling has valid windows only. |
| Normalization | [`layer/batch_normalization.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/layer/batch_normalization.go) and [`layer/batch_normalization2d.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/layer/batch_normalization2d.go) | Both feature normalization and spatial channel normalization already exist. Spatial normalization must not be listed as missing. |
| Execution | [`model/sequential.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/model/sequential.go) and [`model/inference_session.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/model/inference_session.go) | Separate inference sessions already isolate mutable state and share parameters. A session is single-use-at-a-time; outputs are reused. This does not implement named ONNX dataflow graphs. |
| Forward state | [`layer/dense.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/layer/dense.go), CNN layers, and [`layer/layer.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/layer/layer.go) | Layers expose backward behavior and retain caches. ONNX kernels need a forward-only entry point with caller-owned destinations. |
| RNNs | [`layer/simple_rnn.go`](https://github.com/itsmontoya/neuralnetwork/blob/e0a5cbe2d7592cae921d17d83ca5e4ce6972f7f1/layer/simple_rnn.go): fixed-tanh, stateless recurrence, flattened sequence rows | Useful mathematical reference, not a general ONNX RNN/GRU/LSTM implementation. |
| Persistence | `model/serialization.go` and `model/serialization_binary.go` implement library-specific formats | Preserve both. Neither format is ONNX protobuf. |

### Recommendation and ownership

Add an inference-oriented `tensor` package and reusable forward kernels without
changing the stable `matrix`, `layer.Layer`, training, or serialization contracts.
This is an additive proposal: existing library design documents deliberately
excluded general tensors from earlier milestones. NN-01 must reconcile that
boundary explicitly before code depends on a new public API.

The ONNX runtime owns protobuf decoding, domains/opsets, schema defaults, graph
validation, symbolic metadata, scheduling, external files, sessions, diagnostics,
and compatibility reporting. The library owns concrete tensor storage, checked
shape arithmetic, and numerical kernels. There must be no import from
`neuralnetwork` back into `onnx`.

A narrow MLP proof of concept can use current matrix APIs with copies and no
upstream changes. The tasks below are requirements for the proposed reusable
tensor backend and broader milestones, not prerequisites for every experiment.
Do not expand `Sequential` into a general ONNX executor.

## Original implementation instructions

NN-01–NN-12 are complete; the following instructions and task definitions are
retained from the original backlog. Use the published handoffs above for the
implemented APIs and follow [ROADMAP.md](ROADMAP.md) for new runtime work.

Read this library's `AGENTS.md`, `STYLEGUIDE.md`, current source, this file,
and the listed dependency handoffs. Implement `NN-*` tasks in `neuralnetwork`;
implement `RT-*` tasks in the separate ONNX runtime repository. Public names and
package placement below are proposals until NN-01 records the contract.

Each task includes implementation, GoDoc, focused explicit tests, and a handoff.
Use one file per primary type, keep its methods together, put constructors above
their types, use explicit returns and early exits, group stateful variables, and
avoid shadowing. Preserve existing APIs and model encodings. New inference
primitives do not require backward implementations or optimizer parameters.

For code changes run `gofmt`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test -tags=purego ./...`, and `go test -race ./...` where supported.
Record unavailable checks. Match existing behavior with regression tests whenever
reusing internals; use reference fixtures for new numerical semantics. Tests must
cover scalar and empty tensors where an operation permits them. An unchecked
shape multiplication or conversion to Go `int` is not acceptable.

On completion append a **Handoff** to the task: revision, changed files, exact
exported API, ownership/alias rules, supported cases, commands/results, and
remaining limitations. Check a task only after its acceptance criteria pass.

## Original task order

| Task | Needed for | Depends on |
| --- | --- | --- |
| NN-01 | Shared design contract | None |
| NN-02 | M1: concrete tensors | NN-01 |
| NN-03 | M1: shape and indexing rules | NN-02 |
| NN-04 | M1: elementwise kernels | NN-03 |
| NN-05 | M1: matrix products | NN-03 |
| NN-06 | M1: activation kernels | NN-03 |
| NN-07 | M2: tensor transforms | NN-03 |
| NN-08 | M2: indexing and concatenation | NN-03 |
| NN-09 | M3: convolution | NN-03, NN-05 |
| NN-10 | M3: pooling | NN-03 |
| NN-11 | M3: inference normalization | NN-03 |
| NN-12 | Later operator expansion | NN-03, NN-04 |
| NN-13 | Later performance work | NN-04, NN-05, NN-06, NN-07, NN-08, [RT-12][onnx-roadmap] |

NN-04 through NN-11 can be assigned independently once their listed contracts
are complete. M1 (MLP and branching inference), M2 (shape processing), and M3
(CNN inference) are defined in the [ONNX runtime roadmap][onnx-roadmap].

## Foundation and first inference milestone

### [x] NN-01 — Define the additive tensor and kernel contract

**Context:** Existing matrix ownership is deliberate; ONNX requires values whose
shape and type cannot be represented by a positive-size `float32` matrix.
**Depends on:** None. **Deliverable:** a design document in the library and a
precise API handoff for [RT-01][onnx-roadmap].

**Work:** Specify an additive `tensor` package with concrete row-major shapes,
`float32` and `int64` storage, immutable published inputs, checked constructors,
copying accessors, and caller-owned output destinations. Decide public signatures
for kernels and optional workspace objects. Keep symbolic dimensions in `onnx`.
Start with contiguous values; transpose may materialize instead of adding public
strides. Specify scalar shape `[]` (one element), empty dimensions (zero elements),
alias rejection, failure behavior, and concurrency. Recommend no overlapping
inputs/destinations initially; metadata-only aliases need an explicit read-only
contract. Define how existing matrix kernels are reused, even if the first bridge
copies. All workspace state must belong to a call or session.

**Acceptance:** The document lists exact proposed public APIs and examples for a
scalar, `[0, 3]`, an `int64` shape vector, and rank-two matrix multiplication.
It explains compatibility with existing library design exclusions and v1 APIs,
and separates mandatory additions from later zero-copy optimizations. No ONNX
protobuf or operator-version types enter the library's API.

### [x] NN-02 — Add concrete typed tensor storage

**Context:** `Matrix` stores only positive-size two-dimensional `float32` values.
**Depends on:** NN-01. **Likely files:** `tensor/tensor.go`, `tensor/data_type.go`,
and matching type-specific tests; use the final names from NN-01.

**Work:** Implement validated shape/type/storage construction, zero-filled
allocation, typed copying construction, cloning, and copying value accessors.
Use concrete `int64` dimensions at the public boundary; check dimension products,
byte sizes, and host index conversion before allocation. Distinguish unknown
metadata from a concrete zero dimension: concrete tensors never contain symbolic
or negative dimensions. Preserve large integer values exactly, including values
above `2^24`. Allow scalar and zero-element tensors. Keep backing slices private;
do not expose mutable initializer storage through a generic `any` escape hatch.

**Acceptance:** Table tests cover ownership isolation, wrong storage length/type,
negative dimensions, dimension/byte overflow, scalars, `[0]`, `[2, 0, 3]`, and
integer precision. Invalid input returns an error before allocation. Tests verify
that shape accessors also copy and that the zero-value object's behavior is
documented. Existing matrix construction behavior remains unchanged.

### [x] NN-03 — Implement checked shape and indexing utilities

**Context:** Every numerical adapter needs consistent concrete dimension rules.
**Depends on:** NN-02. **Deliverable:** shared checked shape helpers, private
unless another package genuinely needs them.

**Work:** Implement rank/axis validation, negative-axis normalization, contiguous
strides, safe offsets, shape equality, and trailing-axis multidirectional
broadcasting. Also provide a distinct unidirectional broadcast check for Gemm's
bias. Plan output shapes without allocating numeric payloads. Include checked
products and sums for reshape, concatenation, convolution output geometry, and
workspace sizing. Empty dimensions must follow broadcasting rules rather than a
blanket `max(left, right)` shortcut: dimension `0` with `1` produces `0`.

**Acceptance:** Explicit tests cover scalar broadcasting, mismatched ranks,
`[2, 3, 4] + [4]`, `[2, 1, 4] + [1, 3, 1]`, zero dimensions, invalid axes,
and index/size overflow. Shapes incompatible under ONNX/NumPy broadcasting fail
before any destination writes. The handoff identifies helpers later kernels
must reuse instead of duplicating arithmetic.

### [x] NN-04 — Add broadcast elementwise arithmetic

**Context:** Existing add/subtract/multiply operations require matching matrices;
row-vector addition covers only one special case. Existing matrix division
returns an error for a zero denominator, so it cannot be mapped blindly to
floating ONNX Div.
**Depends on:** NN-03. **Deliverable:** forward `float32` Add/Sub/Mul/Div kernels
with destination APIs.

**Work:** Support arbitrary compatible ranks, scalar operands, empty results,
and broadcast operands without expanding a complete copy of either input.
Reuse existing arithmetic for safe matching-shape cases. Define and implement
floating division behavior against the pinned reference, including zero,
signed zero, infinity, and NaN; keep existing matrix division's contract intact.
Validate destination shape/type and alias rules before mutation.

**Acceptance:** Tests include channel bias `[1, C, 1, 1]`, residual addition,
scalar operations, incompatible shapes, destination errors, and special floating
values. Results match independent reference fixtures. Inputs remain unchanged,
and calls on disjoint tensors are safe concurrently. Integer arithmetic is
outside this task; reject unsupported types explicitly.

### [x] NN-05 — Add tensor MatMul and Gemm kernels

**Context:** Rank-two multiplication and separate transpose variants already
exist; ONNX MatMul also needs vector promotion and batch broadcasting.
**Depends on:** NN-03. **Deliverable:** reusable forward matrix-product kernels.

**Work:** Implement MatMul with vector-vector scalar output, matrix-vector,
vector-matrix, rank-two products, and broadcast batch prefixes. Reject scalar
inputs. Use existing matrix arithmetic for compatible nonempty blocks; handle
zero-length reductions without constructing invalid zero-size matrices.
Implement rank-two Gemm with both transpose flags, alpha/beta scaling, optional
bias, and unidirectional bias broadcasting. A missing bias means no bias term.
Keep these parameters mathematical; schema defaults are supplied by `onnx`.

**Acceptance:** Tests cover rectangular matrices, all transpose combinations,
both runtime-provided and constant weights, omitted/scalar/vector/matrix bias,
batch broadcasting, incompatible inner dimensions, empty contractions, and
destination aliases. Fixtures verify the actual numeric result, not just shapes.
The handoff documents accumulation precision and permitted numeric tolerances.

### [x] NN-06 — Add tensor activations and axis softmax

**Context:** Elementwise activation formulas and destination interfaces exist;
softmax currently normalizes rows or row segments.
**Depends on:** NN-03. **Deliverable:** forward tensor ReLU, Sigmoid, Tanh, and
Softmax kernels.

**Work:** Reuse the existing elementwise implementations across concrete tensor
shapes. Add numerically stable softmax over a normalized arbitrary axis, including
an axis with a noncontiguous logical reduction in row-major storage. The runtime
passes the resolved axis; the library does not select defaults by opset.
Keep intermediate values and workspace local to the invocation. Define empty
axis and nonfinite behavior using pinned reference cases, recording any
implementation-dependent cases rather than inventing compatibility claims.

**Acceptance:** Compare rank-one, rank-two, and rank-three outputs for first,
middle, and final axes; include large logits, invalid axes, empty values, and
documented special-value cases. Existing activation results and backward behavior
remain unchanged. Repeated calls cannot overwrite another invocation's output.

## Shape-processing milestone

### [x] NN-07 — Add reshape, flatten, transpose, squeeze, and unsqueeze

**Context:** `layer.Flatten` copies already-flattened spatial rows. It is not a
general tensor shape transformation.
**Depends on:** NN-03. **Deliverable:** type-preserving tensor transforms for
both supported storage types.

**Work:** Implement reshape by a resolved concrete shape, flatten by a normalized
split axis, explicit permutation, insertion of unit axes, and removal of valid
unit axes. The runtime translates ONNX-specific reshape `0`/`-1` rules into the
concrete target; keep that policy out of this library. Materializing a transpose
is acceptable initially. A metadata-only reshape must obey NN-01's alias and
lifetime contract; otherwise copy. Support flatten axes at both endpoints.

**Acceptance:** Test exact row-major order using distinguishable integer values,
all dtype-preserving paths, duplicate/out-of-range permutations and axes,
nonunit squeeze rejection, negative axes as contracted, scalar transitions,
empty shapes, and mismatched element counts. No transform silently reinterprets
integer bytes as floating values. Record copy versus alias behavior per API.

### [x] NN-08 — Add concatenation and general gather

**Context:** Matrix row selection exists, but tensor indexing must work along
arbitrary axes, including integer-valued shape calculations.
**Depends on:** NN-03. **Deliverable:** type-preserving Concat and Gather kernels.

**Work:** Concatenate a nonempty list of compatible tensors along a normalized
axis with checked dimension sums. Gather from an arbitrary data axis using
`int64` index tensors, including scalar indices and negative indices. Output
shape is the data prefix, index shape, and data suffix. Preserve float32 or int64
data type; reject unsupported index types explicitly. Work from contiguous blocks
where useful without publishing borrowed mutable backing slices.

**Acceptance:** Tests cover concatenating feature maps and integer shape vectors,
scalar/multidimensional/empty indices, negative indices, repeated indices,
out-of-range indices, incompatible non-axis dimensions, and overflow. Validate
every index before writing a result. Source tensors remain unchanged.

## CNN milestone

### [x] NN-09 — Add forward-only general 2D convolution

**Context:** Existing Conv2D performs cross-correlation over flattened NCHW
inputs, with weights stored as `[C*kH*kW, M]`. ONNX stores convolution weights
as `[M, C/group, kH, kW]`. Existing configs lack asymmetric padding, groups,
and dilation.
**Depends on:** NN-03, NN-05. **Deliverable:** a float32 NCHW inference kernel and
validated mathematical geometry, preserving existing layer APIs.

**Work:** Accept input/weights and optional bias as tensors, independent height
and width strides/dilations, explicit begin/end pads, and groups including
depthwise convolution. The runtime resolves `auto_pad` to concrete pads; keep
output geometry checked and shared with shape planning. Reuse existing loops or
matrix products where correct, documenting any weight packing. Avoid constructing
optimizer parameters, gradients, or backward input caches. Dynamic weights must
work; optional constant-weight packing must not become a correctness requirement.

**Acceptance:** Fixtures cover standard/grouped/depthwise convolution,
asymmetric pads, unequal strides, dilation, omitted bias, multiple batches, and
kernel-boundary cases. Invalid channels/groups fail clearly. Results match a
reference and the existing layer for its overlapping subset. Training and saved
model regression tests continue to pass.

### [x] NN-10 — Add forward-only max, average, and global average pooling

**Context:** Existing MaxPool2D handles valid 2D windows and stores argmax for
backpropagation. Average/global pooling and general padded geometry are absent.
**Depends on:** NN-03. **Deliverable:** float32 NCHW pooling kernels.

**Work:** Add max and average pooling with explicit pads, independent strides,
floor/ceil geometry, and average-pool inclusion/exclusion of padded elements.
Add dilation for max pooling and global spatial averaging with singleton spatial
output dimensions. The runtime resolves automatic padding and schema-specific
attributes. Max pooling may omit indices in the initial runtime; if an indices
destination is added, define exact int64 flattened index and storage-order rules.
Do not allocate argmax history when no indices output is requested.

**Acceptance:** Test negative-only padded max windows, average divisors at
borders, ceil-mode removal of windows beginning beyond the allowed input region,
dilated max windows, and odd spatial extents. Compare against reference fixtures.
Explicitly document unsupported indices/storage-order cases so the runtime can
reject them. Existing valid MaxPool2D remains compatible.

### [x] NN-11 — Add stateless inference batch normalization

**Context:** Both feature and spatial batch normalization already exist, but
their layer objects include training state and caches.
**Depends on:** NN-03. **Deliverable:** a float32 forward inference kernel with
explicit scale, bias, mean, variance, and epsilon.

**Work:** Normalize per channel across tensors shaped `[N, C, ...]`, using stored
statistics and `scale * (x - mean) / sqrt(variance + epsilon) + bias`. Cover
rank-two feature inputs and NCHW spatial inputs. Accept supplied parameter
tensors without training-mode mutation. The ONNX adapter enforces its supported
schema's rank and output restrictions. There is no running-statistics update or
momentum application during inference.

**Acceptance:** Reference tests cover nondefault epsilon, nontrivial learned
statistics, multiple channels/batches, parameter shape errors, and empty values
where supported. Tests compare the overlapping subset against both existing
normalization layers in evaluation mode and prove that inputs/statistics remain
unchanged. No optimizer state or backward caches are constructed.

## Expansion and optimization

### [x] NN-12 — Add reductions and math needed by model-driven expansion

**Context:** Future exported graphs often express normalization and activations
as primitive arithmetic. Existing reductions only cover matrix rows/columns.
**Depends on:** NN-03, NN-04. **Deliverable:** forward float32 ReduceSum,
ReduceMean, ReduceMax, Exp, Log, Sqrt, and broadcast Pow primitives.

**Work:** Implement reduction axes and keep-dimensions behavior over arbitrary
ranks. Accept resolved mathematical axes; ONNX attribute-versus-input changes
and omitted-axis defaults remain runtime responsibilities. Reuse shape planning
and establish numeric identities or explicit errors for empty reductions from
the supported reference behavior. Provide destination APIs and optional local
workspace. Keep exact versus approximate activation formulas distinct.

**Acceptance:** Tests cover multiple/negative/duplicate axes, keepdims on/off,
all-axis reduction to scalar, scalar inputs, zero dimensions, and invalid math
domains/nonfinite values according to the documented contract. Fixtures compare
values as well as shapes. Only kernels selected by the runtime's next coverage
task need integration; this task does not imply transformer support.

### [ ] NN-13 — Optimize tensor/matrix interoperability and workspaces

**Context:** Copying adapters are acceptable for correctness, but copying each
matrix product's inputs, weights, and outputs may dominate inference costs.
**Depends on:** NN-04, NN-05, NN-06, NN-07, NN-08, [RT-12][onnx-roadmap].
**Deliverable:** measured, compatible improvements, coordinated with
[RT-16][onnx-roadmap].

**Work:** Benchmark the working runtime before changing ownership. Add a narrowly
scoped contiguous float32 bridge or shared internal storage abstraction only
where measurements justify it. Avoid exposing `internal/device` or mutable
initializer slices. Explicitly track read-only versus writable access, host/device
coherence if existing matrix internals are touched, alias lifetime, and call-local
scratch. Add reusable workspace and optional constant-weight packing without
hidden global state. Keep a copying fallback.

**Acceptance:** Record before/after allocations, copied bytes, latency, and peak
live memory on MLP and shape-heavy graphs. Prove output lifetime, immutable
weights, cross-session independence, and race safety. Normal, `purego`, and
cgo-disabled paths agree numerically; exercise Metal coherence only if this
change touches that path. Publish improvements without setting an invented
universal zero-allocation or speedup requirement.

## Explicitly outside this library backlog

The [runtime roadmap][onnx-roadmap] owns ONNX import and execution. ONNX export,
training/autograd, GPU enablement, general sparse/sequence/map/optional values, quantization,
RNN/GRU/LSTM compatibility, and control-flow subgraphs need separate model-driven
design and implementation tasks. They are not hidden requirements of NN-01.
[RT-19][onnx-roadmap] creates bounded extension tasks after the first runtime
is usable.

For numerical semantics consult the relevant versioned operator definitions,
especially [MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html#matmul-13),
[Gemm](https://onnx.ai/onnx/operators/onnx__Gemm.html#gemm-13),
[Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html#softmax-13),
[Conv](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11),
[MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12),
[AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html#averagepool-11),
and [BatchNormalization](https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9).
Resolve schemas from the pinned release selected in [RT-01][onnx-roadmap]; the
current site may also display newer definitions that do not apply to the initial
opset.

[onnx-roadmap]: ROADMAP.md
[backend-release]: https://github.com/itsmontoya/neuralnetwork/tree/v0.7.0
[backend-commit]: https://github.com/itsmontoya/neuralnetwork/tree/6048452c57a0a293db4591cd2e2febd7d6b743a3
[tensor-contract]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/tensor-design.md
[backend-fixtures]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/tensor/testdata/README.md
[nn13-pr]: https://github.com/itsmontoya/neuralnetwork/pull/114
[nn-01]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-01.md
[nn-02]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-02.md
[nn-03]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-03.md
[nn-04]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-04.md
[nn-05]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-05.md
[nn-06]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-06.md
[nn-07]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-07.md
[nn-08]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-08.md
[nn-09]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-09.md
[nn-10]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-10.md
[nn-11]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-11.md
[nn-12]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-12.md
[nn-13]: https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-13.md

### Normative RT-12 shape adapter decisions

The published [NN-07][nn-07] and [NN-08][nn-08] exceptions are binding runtime
inputs. Flatten computes both concrete split products, accepting valid empty
prefixes where the pinned evaluator rejects inferred -1 reshaping. Gather uses
the exact data-prefix + full-indices-shape + data-suffix formula; the evaluator's
empty-indices `[0]` shortcut is retained as a real disagreement. Neither is
counted as unqualified reference agreement. The [M2 fixture guide](testdata/m2/README.md)
records 60 agreements, two Flatten rejections and four Gather differences.
Reshape-13 zero means copy the same-position input dimension; one -1 may be
inferred only when the known product is nonzero. Omitted Squeeze axes remove all
unit dimensions; explicitly empty axes preserve shape. Unsqueeze axes normalize
against output rank. All transforms preserve exact int64 and float32 bits.
