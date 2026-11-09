# Go-native ONNX runtime roadmap

Build an inference runtime in `github.com/itsmontoya/onnx`, using
[`neuralnetwork`](https://github.com/itsmontoya/neuralnetwork) for reusable tensor
storage and numerical kernels. Runtime users should load an `.onnx` model and
evaluate named inputs entirely in Go, with `CGO_ENABLED=0` and no installed
Python, C/C++ runtime, external service, or native ONNX Runtime shared library.
Python reference tools may generate development fixtures; they are not runtime
dependencies. A Go protobuf dependency is compatible with this goal.

This document tracks the implementation plan. RT-01 has specified the contract;
RT-02 provides the Go module and verification foundation. RT-03 adds generated
wire bindings and bounded internal decoding. RT-04 supplies immutable public
tensors and metadata records. There is no inference runtime yet.
Checked tasks have met their stated acceptance criteria; planned operator records
do not represent implemented runtime support.

## Backend baseline and handoff

Use `github.com/itsmontoya/neuralnetwork v0.7.0`, published at commit
[`6048452c57a0a293db4591cd2e2febd7d6b743a3`](https://github.com/itsmontoya/neuralnetwork/tree/6048452c57a0a293db4591cd2e2febd7d6b743a3).
NN-01–NN-12 are complete. [NEEDS.md](NEEDS.md) records the verified release,
links the tracked API handoffs, and summarizes integration constraints. Its
original pre-release review is retained as history. The library's `NEEDS.md`
is ignored and absent from the release; use the published handoffs instead.

[PR #114](https://github.com/itsmontoya/neuralnetwork/pull/114) leaves NN-13
optimization blocked on a working RT-12 runtime and graph benchmarks. The
released copying backend is sufficient for M1–M4. Start at RT-01, complete
RT-11/RT-12, and publish an accessible runtime revision with reproducible MLP
and shape-heavy fixtures for at least two batch sizes. That unblocks NN-13;
RT-16 consumes its results later and is not an NN-13 prerequisite. The existing
NN dependencies below identify released contracts to consume, not missing work
to repeat. Backend kernel tests do not establish runtime ONNX compatibility.

## Architecture and initial scope

The proposed execution path is:

```text
.onnx bytes -> bounded protobuf decoder -> validated named graph
            -> versioned operator resolution -> immutable execution plan
            -> session-local evaluation -> neuralnetwork tensor kernels
            -> named, caller-owned outputs
```

Keep protobuf details and ONNX semantics in this repository. Keep concrete tensor
math in `neuralnetwork`. [NEEDS.md](NEEDS.md) records that library's completed
tasks, remaining optimization work, and published dependency revision. The
runtime must not flatten every graph into `model.Sequential`: fan-out, joins,
multiple inputs/outputs, and shared values
must remain explicit. Existing inference sessions are a useful ownership model,
but they are not an ONNX graph executor.

ONNX graphs are acyclic, topologically ordered dataflow graphs. Validate their
declared order and named values before execution. Model IR versions and imported
operator-set versions are distinct compatibility checks.
See the [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html) and
[versioning specification](https://onnx.ai/onnx/repo-docs/Versioning.html).

The following is a deliberately bounded first target, to be recorded precisely
in RT-01. It is not a claim of general ONNX compatibility:

| Dimension | Proposed first target |
| --- | --- |
| Execution | CPU inference; cgo disabled; deterministic serial node scheduling |
| Model envelope | IR version 8, standard empty-string domain, imported opset exactly 13 |
| Values | Dense `float32` computation and exact `int64` constants/shape values; reject other types |
| Shapes | Concrete runtime shapes, including scalars/empty tensors; fixed dimensions plus symbolic or unknown leading batch metadata initially |
| Weights | Inline TensorProto values initially; explicit external-data support in RT-14 |
| Graph | A top-level DAG, multiple inputs/outputs, fan-out/joins, optional input/output placeholders where the selected schema allows them |
| Operator coverage | Only the implemented schema/type/attribute/rank combinations listed below |
| Unsupported features | Clear load-time errors where statically knowable, contextual run-time errors for data-dependent violations |

Accepting IR 8 does not mean accepting every IR 8 value kind. Reject sparse
initializers, local functions, subgraphs/control flow, custom domains, training
requests, and unsupported types explicitly. Optional empty *name slots* do not
imply support for ONNX Optional values. Exporters using another IR/opset need
an explicitly compatible export or a later version-expansion task; do not rewrite
their version numbers and assume equivalence.

### Milestones and coverage

All schemas below are resolved under an imported standard-domain opset of 13.
An operator's own schema version may be older than that import.

| Milestone | Deliverable and boundary | Required tasks |
| --- | --- | --- |
| M0 | Written contracts, module, and reproducible schema binding | NN-01; RT-01–RT-03 |
| M1 | Working MLP and branching inference with named values | NN-02–NN-06; RT-04–RT-11 |
| M2 | Dynamic shape values and common tensor transforms | NN-07–NN-08; RT-12 |
| M3 | Float32 2D CNN inference | NN-09–NN-11; RT-13 |
| M4 | External weights, hardened lifecycle/limits, documented usable release | RT-14–RT-15, RT-18; M1–M3 remain required |
| Later | Measured optimization, additional versions, and model-selected operators | NN-12–NN-13; RT-16–RT-17, RT-19 |

| Operator family | Schema selected at opset 13 | Coverage milestone |
| --- | --- | --- |
| Identity, Constant | 13, 13 | M1; float32/int64 values; supported scalar/vector/tensor Constant forms only |
| Add, Sub, Mul, Div | 13 each | M1; float32, multidirectional broadcasting |
| MatMul, Gemm | 13 each | M1; float32; MatMul vector/batch rules; rank-two Gemm including optional C |
| Relu, Sigmoid, Tanh, Softmax | 13 each | M1; float32; Softmax axis semantics from schema 13 |
| Flatten, Reshape, Transpose | 13 each | M2; preserve float32/int64 storage |
| Squeeze, Unsqueeze, Shape, Gather, Concat | 13 each | M2; int64 axes/indices/shape outputs; Gather/Concat data may be float32/int64 |
| Conv | 11 | M3; float32 NCHW 2D, groups/dilation, resolved automatic or explicit padding |
| MaxPool, AveragePool, GlobalAveragePool | 12, 11, 1 | M3; float32 NCHW 2D; initially MaxPool values output only, default storage order |
| BatchNormalization | 9 | M3; inference only, one output, supported rank-two/NCHW inputs |

Each operator entry needs a machine-readable capability record, schema-correct
validation, and positive/negative fixtures. Partial type/rank coverage must be
visible. Unsupported nonempty optional outputs are errors, not discarded values.
This plan does not promise that arbitrary exported CNNs, transformers, recurrent
models, or quantized models will run at M3.

## Instructions for a fresh implementing agent

Read [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md), this document,
[NEEDS.md](NEEDS.md), the published library handoffs, current source, and
preceding runtime handoffs. Implement one listed task at a time.
Architectural and public API names below are proposals
until their contract tasks settle them. Do not infer completed dependencies from
the presence of a document or tests alone.

Every task includes focused tests and accurate documentation. Follow the style
guide's file-per-type and constructor rules, named results where appropriate,
explicit returns, early exits, grouped stateful variables, and no shadowing.
Define small interfaces at their consumer; avoid a plugin/provider framework
before there is a second backend. Do not import the other module's `internal/`
packages or break its public API to make an adapter convenient.

For code changes run `gofmt`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`, and
`go test -race ./...` where supported. Use checked-in fixtures so normal tests
need neither network access nor Python. After the module exists, check diffs and
local Markdown links for documentation changes. Record unavailable checks
honestly; keep compatibility records updated with the implementation.

Append a **Handoff** beneath the completed task with changed files, commit,
exported signatures, contract decisions, commands/results, supported/rejected
cases, and next-task requirements. Cross-repository dependencies must name an
available `neuralnetwork` revision/version. Do not commit a local `replace`
directive or mark a task complete against unpublished local changes. Optional
`go.work` use is local development configuration only.

## M0: contracts and project foundation

### [x] RT-01 — Specify the runtime contract and compatibility profile

**Depends on:** NN-01. **Deliverables:** `docs/runtime-contract.md` and the initial
operator capability manifest. **Scope:** design and explicit interfaces.

**Work:** Resolve the proposed IR 8/opset 13 subset against a pinned ONNX release
(suggested starting source: ONNX v1.16.2, with fixtures deliberately using IR 8
and opset 13). Record exact source revision, schema versions, license, and
reference-tool versions. Define module/package boundaries and the public load,
metadata, session, run, and close APIs. Recommend immutable compiled models,
single-call-at-a-time sessions, independent concurrent sessions, and owned output
copies that remain valid after another run or Close. Set finite configurable
decode/model/tensor/workspace limits and context cancellation behavior.

Use the released v0.7.0 tensor signatures and ownership rules. Account for the
copying matrix/activation bridge and call-local MatMul buffers when specifying
budgets; checked shapes alone do not bound memory use. Kernels have no in-loop
cancellation API, so state the initial cancellation granularity explicitly.
Resolve the numerical/reference differences summarized in `NEEDS.md`, including
ReLU NaN behavior and empty tensors, in the compatibility profile and fixture
plan. Do not infer ONNX support from the presence of a mathematical kernel.

Specify initializer/input overlap: an initializer that is also a graph input is
a default value and may be overridden, with validated type/shape. Define required
versus defaulted inputs, requested output ordering, error categories, same-session
busy behavior, and failure recovery. Unknown intermediate shape metadata must
not require full ONNX shape inference. Document exactly which symbolic input
metadata is accepted initially.

**Acceptance:** A fresh agent can implement each boundary from written signatures
and examples. Every restriction above is represented in the capability profile.
The selected release's checker validates a tiny target model. The document
identifies unsupported feature errors and distinguishes this runtime from the
Microsoft ONNX Runtime implementation. Proposed behavior is clearly distinguished
from implemented behavior.

**Handoff (RT-01):** The task revision is the commit introducing
`docs/runtime-contract.md`; obtain its exact hash with
`git log -1 --format=%H -- docs/runtime-contract.md`. Changed files are that
contract, `capabilities.json`, `tools/check_profile.py`, the tiny original
`testdata/profile/` model/provenance, and README/planning status. Previously
ignored ROADMAP/NEEDS/TODO documents are now tracked so fresh clones retain the
contract dependencies and completion history; their existing context is retained.

No Go symbols are implemented/exported yet. The contract supplies intended
load/metadata/session/run/close signatures, an immutable backend-backed Tensor
facade, named owned outputs, finite logical budgets including bridge copies,
leading-batch metadata, default overrides, deterministic errors and lifecycle.
It pins ONNX v1.16.2 at `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557` and
`neuralnetwork v0.7.0` at `6048452c57a0a293db4591cd2e2febd7d6b743a3`.
All 25 capability records remain explicitly planned. ReLU must propagate NaN
via a runtime adapter; empty Softmax and backend shape/CNN reference exceptions
are explicit rather than claimed reference agreement.

Validation: the pinned Python 3.9.6 / ONNX 1.16.2 / NumPy 1.26.4 / protobuf
4.25.3 environment ran `tools/check_profile.py --write` and a second unchanged
`tools/check_profile.py`; all 25 schema versions matched and ONNX's full checker
accepted the IR-8/opset-13 Identity fixture. Its SHA-256 is
`e7fed3b2d29d884d48c43427b69c08c6514ed6dcf0c7a2bdd002d5f46b0ad469`.
JSON validation, Go contract-block syntax formatting, local Markdown paths, and
`git diff --check` passed. Go package tests/vet/race are not applicable before
RT-02 creates the module; no runtime conformance is claimed. RT-02 must pin used
Go dependencies/toolchain and verification CI. RT-03 owns schema binding
regeneration; RT-04 owns the proposed immutable facade.

### [x] RT-02 — Initialize the Go module and verification workflow

**Depends on:** RT-01. **Likely files:** `go.mod`, `go.sum`, package documentation,
and a focused CI workflow.

**Work:** Create module `github.com/itsmontoya/onnx`. Select a Go toolchain
compatible with `neuralnetwork v0.7.0` and its full dependency graph. Its module
declares Go 1.26.1; upstream handoffs report checks on Go 1.27.2. Pin v0.7.0
and the selected pure-Go protobuf dependency.
Establish the package layout from RT-01: public facade, internal wire schema,
graph planning, operator adapters, and test fixtures. Add only packages needed
by subsequent tasks. Include cgo-disabled and `purego` test jobs and race checks
on a supported platform. Document regeneration versus ordinary build commands.

**Acceptance:** A clean checkout builds and runs baseline checks without local
path replacements. Go module metadata names this project and includes only
dependencies used by this runtime. Runtime builds do not
require protoc, Python, Metal, cgo, or native shared libraries. The README remains
accurate about which features have actually landed.

**Handoff (RT-02):** The task revision is the commit introducing `go.mod`;
obtain its exact hash with `git log -1 --format=%H -- go.mod`. Changed files are
`go.mod`, `go.sum`, `doc.go`, `dependency_test.go`, `.github/workflows/go.yml`,
`docs/development.md`, the runtime contract's dependency record, README, and this
roadmap. Module `github.com/itsmontoya/onnx` declares Go 1.26.1 and selects
toolchain Go 1.27.2.
It pins published `neuralnetwork v0.7.0` at the recorded
`6048452c57a0a293db4591cd2e2febd7d6b743a3` revision and pure-Go
`google.golang.org/protobuf v1.36.12`; required SIMD/x/sys versions remain
indirect dependencies. No `replace` directive or workspace is required.

No production symbols are exported yet. The dependency smoke test checks an
exact int64 scalar above 2^24 against literal protobuf wire bytes; importing the
backend tensor package compiles its numerical dependency paths. This is a
toolchain/dependency baseline, not ONNX conformance. Internal packages will be
introduced with actual behavior in RT-03 and later, using the RT-01 boundaries.
CI checks formatting, module tidiness, build, vet, normal tests, cgo-disabled
tests, cgo-disabled `purego` tests, and races on Linux. Development documentation
separates ordinary Go commands from optional Python profile-fixture regeneration
and RT-03's forthcoming protobuf regeneration.

Validation on Go 1.27.2, Darwin/arm64: `go mod verify`, `go fmt ./...`,
`go mod tidy -diff`, `go build ./...`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`,
and `go test -race ./...` passed. Build/test commands used a writable
`GOCACHE=/private/tmp/onnx-runtime-gocache` and network-disabled
`GOPROXY=off GOSUMDB=off` after dependency retrieval. Local Markdown paths and
`git diff --check` passed. An isolated source copy also built and passed
cgo-disabled `purego` tests with `GOWORK=off` and networking disabled.
The inherited pinned profile checker still passes. Normal builds need neither
Python/protoc nor native runtime libraries. RT-03 must add pinned, reproducible
ONNX bindings and bounded decoding; all capability records remain planned.

### [x] RT-03 — Add reproducible ONNX protobuf bindings and bounded decoding

**Depends on:** RT-02. **Likely files:** `internal/onnxpb/`, a generation tool,
source/license notices, and decoder tests.

**Work:** Generate Go bindings from the exact ONNX schema pinned in RT-01,
including required imported schema files. Isolate generated wire types from the
public API. Check in generated artifacts and pin generation tools. Use a
deterministic generated-code split if necessary to preserve one primary type
and its methods per file; do not hand-maintain a second protobuf wire parser.
Bound input byte size before unmarshalling and configure recursion limits.
Preserve protobuf unknown fields as supported by the decoder, while separately
rejecting unsupported semantic features during validation.

**Acceptance:** Regeneration is reproducible and leaves checked-in output
unchanged. Decode tests cover a valid ModelProto/TensorProto, typed repeated
fields, packed fields, unknown fields, truncation, invalid wire data, recursion,
and oversized input. Reader errors include context. The loader neither closes
caller-owned readers nor reads unbounded bytes before applying a size limit.

**Handoff (RT-03):** The task revision is the commit introducing
`internal/decode/protobuf.go`; retrieve it with
`git log -1 --format=%H -- internal/decode/protobuf.go`. Changed files include
the unmodified pinned schema/license and attribution in `third_party/onnx/`,
the scoped `.gitattributes` preserving its original trailing blank line,
`tools/onnxgen/main.go`, 36 generated `internal/onnxpb/*.pb.go` files and package
documentation, `internal/decode/` with its tests, and README/development/contract
documentation. ONNX source is v1.16.2, commit
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`; this schema has no imports.
The backend remains published `neuralnetwork v0.7.0`, commit
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, without replacements.

Internal APIs are `ParseModel(ctx context.Context, data []byte, limits Limits)
(*onnxpb.ModelProto, error)`, `ReadModel(ctx context.Context, source io.Reader,
limits Limits) (*onnxpb.ModelProto, error)`, and corresponding ParseTensor/
ReadTensor returning `*onnxpb.TensorProto`. `decode.Limits` has MaxBytes and
MaxRecursion fields; zero selects 64 MiB and 64 known-message levels. Negative
limits and byte-limit lookahead overflow are invalid arguments. Errors retain
I/O/context/protobuf causes and expose internal ErrInvalidArgument, ErrLimit,
and ErrInvalidWire categories. No public load/execution API is implemented.

Byte inputs are bounded before unmarshalling; readers consume at most limit+1
bytes in chunks no larger than 32 KiB, never close caller resources, reject
invalid read counts/no progress, and check cancellation between reads and
before/after synchronous protobuf decoding. Unknown fields/groups are retained.
Official protobuf v1.36.12 applies configured recursion to known messages but
uses a separate fixed 10,000-descent allowance for unknown groups (10,001 nested
levels accepted, 10,002 rejected). Both behaviors are tested and documented;
there is no second wire parser. Decoded wire objects still permit unsupported
ONNX semantic features, for RT-05/RT-06 validation, and byte limits do not claim
to bound protobuf object overhead or total process heap.

Regeneration uses protoc 33.2 (generated version 6.33.2), the module-pinned
protoc-gen-go v1.36.12, schema SHA-256 checking, and a deterministic AST split
that keeps each generated type/method set together while retaining upstream
generated implementation style. Ordinary builds/tests need no protoc or Python.
Two generation passes reproduced all 36 files byte-for-byte.

Validation on Go 1.27.2, Darwin/arm64: formatting, `go mod tidy -diff`,
`go mod verify`, `go build ./...`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`, and
`go test -race ./...` passed with downloaded dependencies and networking
disabled. Decoder statement coverage is 97.1%. Literal tests cover packed and
unpacked float32/int64, exact large/negative integers, NaN payload/signed-zero
bits, owned raw bytes, unknown preservation, truncation/invalid wire, both
recursion limits, byte bounds, I/O causes, reader ownership, and cancellation.
The inherited 25-schema/checker fixture, local Markdown paths, and
`git diff --check` pass. RT-04 next implements public tensor values/metadata;
RT-05 reuses TensorProto decoding for validated owned numerical payloads.

## M1: first working inference runtime

### [x] RT-04 — Expose runtime tensor values and metadata

**Depends on:** RT-02, NN-02, NN-03. **Likely files:** public tensor/value metadata
types and conversion helpers in their own files.

**Work:** Build the public input/output surface around the library tensor
contract, using a direct public type or thin facade as decided in RT-01. Avoid
maintaining a second numerical storage engine. Represent model metadata
separately: known dimension, symbolic dimension, and unknown dimension must
remain distinct. Accept concrete typed values with validated shapes; retain
int64 precision. Define tensor ownership at load/run boundaries, and enforce
the chosen input immutability contract during a run.

**Acceptance:** Tests cover scalar, empty, rank-two/rank-four values, integer
shape tensors, copied accessors, invalid dtype/storage, and metadata distinctions.
There is no cast of int64 shape data through float32. GoDoc explains which
objects may be shared across sessions and how long outputs remain valid.

**Handoff (RT-04):** The task revision is the commit introducing `tensor.go`;
obtain its exact hash with `git log -1 --format=%H -- tensor.go`. Changed files
are `tensor.go`, `data_type.go`, `errors.go`, `dimension_kind.go`, `dimension.go`,
`value_info.go`, `metadata.go`, matching tensor tests/metadata example, package
documentation, README, development/contract status, and this handoff.

The implemented constructors are `NewTensor(DataType, []int64) (*Tensor, error)`,
`NewFloat32([]int64, []float32) (*Tensor, error)`, and
`NewInt64([]int64, []int64) (*Tensor, error)`. Tensor methods are
`Validate() error`, `DataType() DataType`, `Shape() []int64`, `Rank() int`,
`Len() int64`, `ByteSize() int64`, `Float32Values() ([]float32, error)`,
`Int64Values() ([]int64, error)`, and `Clone() (*Tensor, error)`, all with named
results and explicit returns. `DataType` aliases the released backend type;
`Invalid`, `Float32`, and `Int64` are its public constants. ErrInvalidArgument
classifies invalid constructors, receivers, and typed access through errors.Is;
contextual backend causes remain wrapped.

Storage delegates to `neuralnetwork v0.7.0`, published commit
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, without replacements or duplicated
numeric storage. The facade retains a private backend pointer and exposes no
destination/mutation API. Constructors/accessors/Clone own copies; immutable
values may be shared by concurrent readers and future sessions. Scalars, empty
values, rank-two/rank-four data, exact int64 extrema, and special float bit
patterns are covered. Invalid types/shapes/counts and nil/zero receivers fail;
typed access rejects the other dtype even for empty tensors. Constructor size
checks establish representability; load/run quotas remain the executor's work.

DimensionKind/Dimension/ValueInfo/Metadata implement the contract's plain
metadata records. Known zero, symbolic, and unknown dimensions remain distinct;
ShapeKnown distinguishes unknown rank from a known scalar. Metadata parsing,
validation, and copied model accessors arrive with graph/model tasks.

Validation on Go 1.27.2, Darwin/arm64 passed: `go fmt ./...`, `go mod tidy -diff`,
`go mod verify`, `go build ./...`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`, and
`go test -race ./...`, using a writable build cache and offline dependency mode.
The pinned profile checker, local Markdown paths, Go contract-block formatting,
and diff whitespace checks also passed. No separate linter is configured.
RT-05 should decode into owned backend values through the shared internal
TensorProto decoder; facade wrapping must remain private to package onnx.
Future execution must allocate disjoint destinations and clone published outputs
so constants and previously returned tensors remain immutable.

### [ ] RT-05 — Decode inline initializers and tensor attributes

**Depends on:** RT-03, RT-04. **Deliverable:** one shared TensorProto decoder for
initializers, Constant tensor attributes, and backend test value files.

**Work:** Implement supported `float_data`, `int64_data`, and little-endian
`raw_data` paths with exact element/byte counts. Check dimensions, data type,
segment restrictions, conflicting encodings, byte overflow, and allocation
budgets before tensor allocation. Distinguish a scalar with no dimensions from
a tensor with a zero dimension. Preserve special float32 bit patterns. Reject
unsupported types, sparse tensors, and external storage with descriptive errors
until RT-14 provides an explicit resolver.

**Acceptance:** Literal wire/value tests verify typed and raw encodings agree,
including negative integers, values above `2^24`, NaN/infinity, scalars, and empty
tensors. Malformed/truncated data, mismatched counts, negative dimensions, and
large allocations fail without panics. Resource accounting includes protobuf's
decoded payload and the owned tensor copy rather than claiming zero-copy loading.

### [ ] RT-06 — Validate the model and build a named graph

**Depends on:** RT-01, RT-03, RT-05. **Deliverable:** internal graph/value/node
records, version checks, and structural validation.

**Work:** Validate IR/opset/domain imports, required graph fields, unique value
producers, declared inputs/outputs, initializer identity, attribute uniqueness,
and topological order. Allow graph outputs sourced directly from an input or
initializer; node labels need not be unique and are not value identifiers.
Handle initializer/input overlap per RT-01. Preserve empty optional slots by
position, rather than dropping them before schema validation. Check the entire
graph, including nodes outside requested output paths; no silent pruning of
unsupported operators. Detect forward references, cycles, missing values, and
invalid repeated definitions. Reject functions/subgraphs and unsupported value
kinds explicitly while preserving harmless metadata.

**Acceptance:** Table tests cover branches/joins, multiple graph inputs/outputs,
shared constants, defaulted inputs, missing imports, duplicate outputs, invalid
order, cycles, and optional placeholders. Every structural failure identifies
the node index or named value. Constants and symbolic metadata cannot change
after the model is compiled.

### [ ] RT-07 — Resolve versioned operators and plan types/shapes

**Depends on:** RT-06, RT-04. **Deliverable:** an internal operator registry,
capability manifest integration, and execution-plan compilation.

**Work:** Resolve by domain, operator type, and imported opset. Select the newest
applicable schema only within an explicitly tested support interval; do not use
one implementation for arbitrary future imports. Validate arity, optional slots,
attribute types/defaults, type constraints, and static rank restrictions.
Represent known versus deferred output shapes, infer the supported subset from
inputs/attributes, and allow missing intermediate `value_info`. Keep runtime
checks for data-dependent shapes. Do not allocate numeric output tensors while
performing static inference. Emit deterministic diagnostics with node index/name,
domain, operator, imported opset, and relevant input shapes.

**Acceptance:** Tests exercise real selection differences and schema versions
older than the model import, unsupported attributes/types/outputs, and absent
metadata. A capability query reports exactly the implemented restrictions. The
planner can compile an Identity/Constant test graph through minimal handlers;
numeric families arrive in RT-09. Errors retain causes for `errors.Is`/`As`.

### [ ] RT-08 — Implement sessions and serial graph execution

**Depends on:** RT-07. **Deliverable:** compiled-model/session/run lifecycle from
RT-01, initially exercised with the minimal handlers.

**Work:** Bind provided/defaulted named inputs, validate dtype, rank, fixed
dimensions, and repeated symbolic batch bindings, then run nodes in validated
topological order. Support named requested outputs and values with multiple
consumers. Keep per-run value tables, allocations, operator workspace, and shape
bindings private to each session. Do not reuse a value while another consumer
or requested output needs it. Honor context cancellation between nodes and
thread a documented cancellation mechanism through long kernels where available.
Apply finite per-run tensor/workspace budgets from the first executor version.

**Acceptance:** Execute branching, multi-input, multi-output, zero-node passthrough,
and default-override fixtures. Two sessions can run concurrently without sharing
scratch; overlapping calls on one session follow the documented busy contract.
Returned outputs survive later runs and Close. Missing/unexpected inputs,
cancelled calls, limits, and operator errors release temporary state and permit
subsequent valid runs unless the contract explicitly defines a terminal failure.

### [ ] RT-09 — Integrate M1 arithmetic, activations, and constants

**Depends on:** RT-08, NN-04, NN-05, NN-06. **Deliverable:** all M1 operators in
the coverage table, each with schema validation and shape planning.

**Work:** Map Identity and supported Constant forms to typed values. Map
Add/Sub/Mul/Div, MatMul/Gemm, Relu/Sigmoid/Tanh, and Softmax to library kernels.
Resolve ONNX defaults here, including Gemm scalars/transpose flags and Softmax-13
axis `-1`. Support Constant's permitted float/int scalar/vector/tensor forms for
the two supported types; reject sparse/string/other forms explicitly. Treat
omitted Gemm C correctly. Keep all runtime inputs, including weights, eligible
to be graph values rather than requiring every weight to be an initializer.

Carry the NN-06 reference differences into the adapter fixtures: the backend
ReLU maps NaN to positive zero, and empty-axis Softmax succeeds where ONNX
1.16.2's evaluator rejects it. Implement and document the treatment selected
in RT-01; do not report these cases as unqualified reference agreement.

**Acceptance:** Tiny graphs verify each operator plus an MLP and a residual
branch. Test nonfinal softmax axes, MatMul vector/batch rules, all Gemm flags and
bias shapes, scalar broadcasting, empty values, zero-denominator floats, and
invalid combinations. Load-time known errors fail before execution; dynamic
errors name the failing node. No operator invokes training or constructs an
optimizer solely to perform inference.

### [ ] RT-10 — Build the fixture and differential conformance harness

**Depends on:** RT-03, RT-04. **Deliverables:** fixture provenance manifest,
`testdata/` models/values, deterministic generation tooling, and Go test harness.
This task can begin before the numeric adapters and grows with each milestone.

**Work:** Select and pin ONNX backend cases compatible with the chosen profile;
add small independently generated graphs for gaps. Store model bytes, named
inputs, expected outputs, exact IR/imports, origin/license, reference versions,
and hashes. Generate expected values with the pinned ONNX reference evaluator
or independently pinned ONNX Runtime during development. Include hand-calculated
cases so decoding and comparison are not verified only through round trips.
Separate integer/shape exact comparisons from float tolerances; define explicit
NaN/infinity/signed-zero handling. Record unsupported cases with a reason.

Use the v0.7.0 handoffs and fixture provenance to retain actual reference
differences, especially empty Gather shapes, pooling geometry/special values,
and BatchNormalization-9. The pinned normalization evaluator applies momentum
even to one-output inference, so cross-check stored-statistics expectations
with an independent formula and the recorded ORT cases. NN-12's reduction
fixtures use opset 18 and do not establish opset-13 adapter coverage.

**Acceptance:** Ordinary Go tests run entirely offline without Python or a
shared library. A deliberate numeric error makes a fixture fail. Skips cannot
count as supported cases. The harness supports both isolated operators and
end-to-end graphs, and it reports input/output names and mismatch positions.
See the [ONNX backend testing documentation](https://onnx.ai/onnx/repo-docs/OnnxBackendTest.html)
for the upstream suite's role; use cases from the pinned release.

### [ ] RT-11 — Demonstrate and document the first working runtime

**Depends on:** RT-09, RT-10. **Deliverables:** a small inference example,
an integration report, and accurate getting-started documentation.

**Work:** Check in a small deterministic IR 8/opset 13 MLP fixture and a
branching fixture with at least two named inputs/outputs. The example loads the
model, inspects metadata, creates a session, supplies values, and reads named
results using the actual public API. Exercise more than one batch size through
the allowed leading batch metadata. Explain how to export for this profile and
how unsupported imports/operators are reported; avoid implying all exporter
defaults are compatible.

**Acceptance:** Examples and golden integration tests work with cgo disabled.
M1 fixtures match reference outputs; lifecycle/race checks pass. A clean clone
can run the documented commands. The capability table lists every M1
restriction. This is the first usable milestone; M2/M3/optimization are not
presented as prerequisites for showing inference working.

## M2 and M3: useful model coverage

### [ ] RT-12 — Integrate shape tensors and transformations

**Depends on:** RT-11, NN-07, NN-08. **Deliverable:** all M2 operators in the
coverage table and runtime-dependent shape evaluation.

**Work:** Implement Shape from the concrete input, type-preserving Flatten,
Transpose, Reshape, Squeeze/Unsqueeze, Gather, and Concat. Use schema-13 axes
inputs for squeeze/unsqueeze, not older attribute forms. Translate Reshape-13's
zero-copy-dimension and single inferred `-1` rules, including ambiguous zero-size
cases, into a concrete library shape. Do not accept a later `allowzero` attribute
under schema 13. Normalize axes in the correct input/output rank. Evaluate shape
subgraphs at run time when they depend on inputs; keep results in int64.

**Acceptance:** A graph using `Shape -> Gather -> Unsqueeze/Concat -> Reshape`
works for two batch sizes, with scalar and empty shape cases as allowed by the
schemas. Tests cover omitted squeeze axes, invalid duplicate axes, permutation
defaults, negative Gather indices, and invalid reshape counts. Missing inferred
intermediate shapes do not prevent execution. No integer shape passes through a
float conversion or leaks between sessions.

**Handoff requirement for NN-13:** Provide an accessible runtime source revision
and exact backend version, reproducible MLP and shape-heavy fixtures with hashes,
named inputs/expected outputs, and runnable examples for at least two batch
sizes. Link [NN-13's measurement plan](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-13.md).
Do not wait for RT-16 to establish the unchanged copying baseline.

### [ ] RT-13 — Integrate 2D CNN operators

**Depends on:** RT-12, NN-09, NN-10, NN-11. **Deliverable:** all M3 adapters in
the coverage table and a small end-to-end CNN fixture.

**Work:** Implement Conv-11, MaxPool-12, AveragePool-11,
GlobalAveragePool-1, and inference BatchNormalization-9. Resolve strides,
dilations, groups, pads, `auto_pad`, kernel shape, ceil mode, and supported output
counts from their actual schemas. Compute SAME_UPPER/SAME_LOWER padding from
concrete spatial dimensions and validate conflicting attributes. Map NCHW and
ONNX weight order correctly; do not feed ONNX weight bytes directly into the
existing layer's differently shaped matrix. BatchNormalization-9 must use the
inference output form; do not invent newer training-mode attributes for it.

Use the NN-10/NN-11 edge cases recorded in `NEEDS.md` when validating the
supported subset. Test full-kernel average divisors for partial ceil windows,
ceil trimming, empty/all-padding windows, and normalization's zero-variance/
zero-epsilon case. Preserve actual reference disagreements in the compatibility
record instead of advertising backend extensions as universal ONNX behavior.

**Acceptance:** Reference cases cover grouped/depthwise/dilated convolution,
asymmetric and automatic padding, padded negative max pooling, ceil geometry,
average padding divisors, and nondefault normalization epsilon. Reject a
requested MaxPool indices output or unsupported storage order explicitly in
this initial profile. A composed Conv/normalization/activation/pooling/flatten/
Gemm model matches reference outputs for multiple batches.

## M4: loading, robustness, and a documented release

### [ ] RT-14 — Add explicit external tensor-data resolution

**Depends on:** RT-11, RT-05. **Deliverable:** bounded external-data support via
an explicit loader resolver and an optional confined filesystem helper.

**Work:** Decode TensorProto external location/offset/length fields, validate
their integer ranges and payload sizes, and reuse the typed tensor decoder.
Require a resolver for reader/byte-based loading; never guess a working-directory
base. The filesystem helper must confine reads to a supplied model directory,
reject traversal/absolute paths and symlink escapes, and avoid an unchecked
validate-then-open race. If confinement cannot be guaranteed on a platform,
require the caller's resolver rather than claiming the helper is confined.
Do not automatically fetch URLs. Account for aggregate model weight bytes and
handle reused locations/ranges without accidentally sharing writable values.

**Acceptance:** Test shared weight files, nonzero offsets, omitted length,
short reads, invalid ranges, traversal, symlink escapes, and budget exhaustion.
Owned bytes are valid after resolver resources close. Missing resolvers and
storage failures report tensor/location context. Document the chosen checksum
verification policy against the pinned
[external-data specification](https://onnx.ai/onnx/repo-docs/ExternalData.html).

### [ ] RT-15 — Harden resource limits, errors, cancellation, and lifecycle

**Depends on:** RT-11; rerun its cases against RT-12–RT-14 as they land.
**Deliverable:** fuzz targets, resource/lifecycle integration tests, and an audit
of the limits already introduced in RT-03, RT-05, and RT-08.

**Work:** Bound graph counts, names/attributes, ranks/dimensions, initializer
bytes, per-value outputs, total live values/workspace, and concurrent session
resources according to RT-01. Extend context checks into long loops through a
small library contract addition if needed; report the actual cancellation
granularity. Fuzz protobuf entry points, tensor decoding, graph references, and
shape/attribute planning with finite inputs. Exercise run failures, failed load
cleanup, overlapping Run/Close, independent sessions, and idempotent closure.
Do not recover arbitrary programmer panics and return success.

**Acceptance:** Malformed or oversized fixtures return bounded, contextual errors
without process panics or lingering session state. Race tests pass across
independent sessions and lifecycle contention. A failing/cancelled run preserves
already returned owned outputs and immutable constants. Allocation accounting
tests include intermediates and scratch, not just final tensor sizes. All
documented default limits and cancellation promises have direct tests.

### [ ] RT-18 — Publish complete usage and compatibility documentation

**Depends on:** RT-12, RT-13, RT-14, RT-15 and passing M1–M4 conformance cases.
RT-16/RT-17 are optional later improvements, not release blockers.
**Deliverable:** a release-ready repository and checklist; publishing a tag is a
separate maintainer action.

**Work:** Document the actual load/run/session APIs, named I/O, dtype/shape
restrictions, defaulted initializers, owned outputs, concurrency, limits,
cancellation, external-data resolver, and error handling. Generate an operator
support table from the same manifest used for validation. Record supported
IR/opset imports, tested export instructions, model fixture provenance, license
notices, dependency versions, pure-Go commands, and realistic benchmark results
if available. Update README/TODO and the task handoffs to match shipped code.

**Acceptance:** A clean environment can run MLP and CNN examples with
`CGO_ENABLED=0`, no Python, and no native runtime libraries. Documented failures
are reproducible. The release checklist captures passing tests, reference
comparisons, and known gaps; no skipped tests are advertised as support. Do not
call the package fully ONNX-compliant or imply compatibility with every model.

## Later: performance and compatibility expansion

### [ ] RT-16 — Measure inference and optimize live tensor storage

**Depends on:** RT-12, RT-13, RT-15, NN-13. **Deliverable:** reproducible
benchmarks and measured compatible optimization.

**Work:** Measure load time, first/warm inference, allocations, copied bytes,
peak live tensor/workspace bytes, and latency on fixed MLP/CNN/shape-heavy
fixtures. Separate parsing from numerical execution. Add value consumer counts
and conservative lifetime-based reuse; graph outputs, aliased views, shared
constants, and defaulted/overridden inputs require correct ownership. Cache
packed weights only when immutable. Consider constant folding only for supported
pure operators with checked budgets and identical numerical semantics. Avoid
parallel node scheduling until measurements justify its added state management.

**Acceptance:** Before/after reports include Go version, build tags, hardware,
thread settings, model hash, and batch sizes. Branches/joins, output subsets,
aliases, changing batch sizes, cancellation, and concurrent sessions still pass
golden/race tests. Memory reuse never changes previously returned outputs. Keep
optimizations only with demonstrated benefit or a documented memory objective.

### [ ] RT-17 — Add explicit IR/opset profiles beyond the first target

**Depends on:** RT-11, RT-12, RT-13, RT-10. **Deliverable:** one additional tested
profile per change, with version-specific schema adapters and fixtures.

**Work:** Select a demanded target from real exports and compare every supported
operator's applicable schema with the current profile. A reasonable first
candidate is IR 8/opset 17; verify it rather than assuming. Account for changes
such as Reshape's later `allowzero` and BatchNormalization's later training-mode
contract. If supporting older imports, handle the older Softmax coercion/default
and squeeze/unsqueeze attribute forms separately. Add new IR versions only after
auditing their semantic features; a newer generated protobuf does not prove
runtime support. Preserve explicit rejection of unsupported training modes.

**Acceptance:** Tests include equivalent graphs under each accepted profile and
graphs whose semantics differ by version. Version rejection remains deterministic,
and the machine-readable manifest lists tested intervals/restrictions. The
runtime never silently converts opsets or invokes an external converter.

### [ ] RT-19 — Extend coverage from an audited model corpus

**Depends on:** RT-18; NN-12 when integrating the reduction/math family.
**Deliverable:** a model compatibility report and bounded implementation tasks
for the next selected model family, followed by the first selected operator set.

**Work:** Collect small redistributable exports for intended workloads and record
model hashes, licenses, IR/opsets, operators, dtypes, shapes, attributes, and
external-data needs. Produce deterministic unsupported-feature reports with
counts and failing nodes. Prioritize the smallest additions that enable useful
models. For primitive normalization/activation graphs, integrate NN-12's
reductions/math only with schema-version-specific adapters and reference cases.
Do not declare transformer support based on MatMul and Softmax alone.

For larger extensions create separate task files with contracts, dependencies,
and acceptance fixtures: Slice/Expand/Pad/Resize; bool/int32 and comparison/Where;
float16/bfloat16; RNN then GRU and LSTM with explicit layouts/gates/state;
quantization with rounding/saturation/per-axis scales; and functions/control-flow
with nested scopes and bounded execution. Sequence/map/optional/sparse values
also need explicit value-model design. Split these initiatives before production
work; they are not one unbounded implementation ticket.

**Acceptance:** Each selected extension has a named enabling model and failing
fixture, exact schema/type scope, the required library task, and an independent
handoff-ready implementation task. The first implemented set passes isolated
and model-level reference checks. Unselected features remain explicit gaps,
and adding coverage does not weaken existing validation.

## Suggested first assignments

Start with RT-01 against the released v0.7.0 backend, then RT-02/RT-03.
RT-04/RT-05 join existing tensor storage and runtime decoding; RT-06–RT-09
establish execution while RT-10 builds reference fixtures. Finish RT-11 before
expanding coverage. RT-12 then supplies the working graphs needed by NN-13;
RT-13 and M4 can proceed without that optimization. Keep each handoff tied to
an accessible source revision and the actual backend version.
