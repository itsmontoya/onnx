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
tensors and metadata records. RT-05 adds shared, budgeted inline tensor
conversion. RT-06 adds immutable named graphs and whole-model structural checks.
RT-07 adds versioned operator resolution and immutable Identity/Constant planning.
RT-08 adds public loaders, model/session lifecycle and serial Identity/Constant
execution. RT-09 adds all M1 numerical handlers and scalar/list Constants.
RT-10 supplies the reusable offline typed fixture and differential harness.
RT-11 demonstrates MLP/branching inference with multi-batch reference goldens,
a runnable public example and the first integration report. RT-12 adds all M2
shape operators, exact runtime shape subgraphs, and the NN-13 baseline handoff.
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
released copying backend is sufficient for M1–M4. RT-11/RT-12 now provide an [accessible runtime revision and handoff](docs/handoff-nn13.md)
with reproducible MLP/shape-heavy fixtures and baseline commands at batches 1
and 3. NN-13 can measure that unchanged copying baseline; RT-16 consumes its
results later and is not an NN-13 prerequisite. The existing
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

### [x] RT-05 — Decode inline initializers and tensor attributes

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

**Handoff (RT-05):** The task revision is the commit introducing
`internal/decode/tensor_decoder.go`; obtain its full hash with
`git log -1 --format=%H -- internal/decode/tensor_decoder.go`. Changed files are
that decoder, `tensor_limits.go`, `payload.go`, their tests plus the deterministic
cancellation fixture, internal errors/package documentation, public package
status documentation, README, TODO, development/contract documentation, and this
roadmap. Backend remains published `neuralnetwork v0.7.0`, commit
`6048452c57a0a293db4591cd2e2febd7d6b743a3`; protobuf remains v1.36.12, and the
vendored ONNX schema remains v1.16.2 at
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`. No replacement or unpublished API
is used.

Internal API is `NewTensorDecoder(context.Context, proto.Message, int64,
TensorLimits) (*TensorDecoder, error)`, with `Decode(context.Context,
*onnxpb.TensorProto) (*tensor.Tensor, error)` and
`Usage() (loadBytes, initializerBytes int64)`. Pass the complete decoded ONNX
model/attribute/test tensor and the actual original encoded byte count. The
constructor registers nested TensorProto pointers and counts all decoded fields
before numeric conversion: repeated/scalar numeric widths, strings/bytes, and
retained unknown bytes, conservatively including shape/metadata. TensorLimits
implements MaxLoadBytes (1 GiB), MaxInitializerBytes (256 MiB), MaxTensorBytes
(64 MiB), MaxRank (32), MaxDimension (2^31-1), and MaxRecursion (64); zero fields
select defaults and negatives fail. Match raised wire/traversal recursion limits.
The existing ParseModel/ReadModel/ParseTensor/ReadTensor APIs still return wire
objects, not numerical values.

One decoder per load aggregates all owned initializer/Constant copies. Each
typed conversion charges one owned copy; raw conversion charges both staging
and an owned copy. Successful staging charges remain conservatively reserved.
Usage is unchanged on errors/cancellation. Checked remaining-budget arithmetic,
backend ValidateShape, rank and every-dimension checks precede numeric allocation.
Object headers, slice excess capacity and GC retention are outside these logical
payload limits. The decoder is local to a load, retains registered wire tensor
pointers, and requires wire objects to remain unmodified during use. Returned
backend values own their shapes/payloads. Root package onnx must wrap backend
values privately; internal packages must not import its public facade.

Supported encodings are exact float_data/int64_data or little-endian raw_data.
Scalar and empty shapes remain distinct; empty payloads are valid for zero
counts. Float NaN payloads/infinities/signed zero and exact int64 extrema are
preserved. Wrong/conflicting payloads, segments, unsupported dtypes, sparse
messages and all external location/metadata are rejected. Empty repeated fields
have no wire-object presence and are treated as absent. New internal
ErrInvalidTensor/ErrUnsupported categories and the broadened ErrLimit retain
errors.Is classification and wrapped causes; conversion errors name the tensor.
Cancellation is polled during decoded traversal, every 1,024 raw elements, and
before/after synchronous backend copies; those backend copies are not
interruptible.

Validation on Go 1.27.2, Darwin/arm64 passed: `go fmt ./...`, `go mod tidy -diff`,
`go mod verify`, `go build ./...`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`, and
`go test -race ./...`, using the writable build cache and offline dependency
mode. Literal tests verify typed/raw and packed/unpacked values, exact bit/value
preservation, independent owned data, malformed/truncated wire, count/type/
encoding errors, huge/overflowed shapes, scalar/empty values, nested registration,
sparse/external rejection, aggregate/boundary budgets, constructed-object cycles,
and cancellation/failed reservation rollback. The pinned 25-schema/checker
fixture, local Markdown targets, Go contract-block formatting, and diff whitespace
checks pass. Ordinary tests need no Python or protoc.

RT-06 must validate the entire model graph and use one decoder with its complete
wire object and actual encoded size, tracking reader consumption rather than
substituting proto.Size. It must avoid uncharged extra numerical copies, retain
immutable constants, and release wire/decoder state when compilation finishes.
RT-07/RT-09 still own Constant schema/attribute validation and operator handling;
RT-14 must add explicit external resolution. No public loader, graph validator,
operator, or session is enabled by this task.

### [x] RT-06 — Validate the model and build a named graph

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

**Handoff (RT-06):** The task revision is the commit introducing
`internal/graph/graph.go`; obtain its full hash with
`git log -1 --format=%H -- internal/graph/graph.go`. Changed files are the new
`internal/graph/` records, builder, validation/metadata helpers and matching tests;
`internal/decode/tensor_decoder.go` and its reservation tests; README, TODO,
package documentation, development/contract documentation, and this roadmap.
Backend remains published `neuralnetwork v0.7.0`, commit
`6048452c57a0a293db4591cd2e2febd7d6b743a3`; protobuf remains v1.36.12,
and ONNX source remains v1.16.2 at
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`. No replacement/backend change
or unpublished dependency is used.

Internal API is `graph.Build(context.Context, *onnxpb.ModelProto, int64, Limits)
(*Graph, error)`. Limits contains Tensor (decode.TensorLimits), MaxNodes,
MaxValues, MaxAttributes and MaxNameBytes; zeros choose the contract defaults,
negatives fail. Graph methods Metadata, Inputs, Outputs, Nodes, Values and
Value(name) return independent records; Usage returns conservative load/owned
constant bytes. The records are Metadata, ValueInfo, Value, Node, Attribute,
Constant, Dimension/DimensionKind, AttributeKind, Property and Annotation.
Value stores its producer index (-1 for sources), input/default state and all
declarations. Constant methods Name, DocString, Metadata, DataType, Shape,
ByteSize, Float32Values, Int64Values and Clone expose copied metadata/numerical
values or an independent backend clone; no writable shared tensor is exposed.
The new `TensorDecoder.ReservePayload(context.Context, int64) error` charges
additional attribute scalar/list and dimension snapshots before allocation,
without treating those as initializer bytes. No public loader/planner/executor
or root error API is introduced.

Build requires a complete decoded model and its actual original encoded length;
future reader loaders must track consumption rather than use proto.Size. It uses
one decoder with original registered tensor pointers, converts every initializer
and dense tensor attribute, and retains neither wire nor decoder state. Logical
usage includes encoded/decoded fields, owned tensors, raw staging and extra
snapshot payloads. Compilation makes no extra numerical clones. Strings are
immutable references; slice/header/object overhead remains outside the payload
quota. Copying accessors allocate outside Build. The future compiler/executor
must budget Constant.Clone before allocation and avoid retaining redundant
compiled copies indefinitely. Internal records do not import the public facade.

The whole declared graph is checked, including unused nodes. IR must be 8 and
the sole operator import must have the empty domain and opset 13. Producer,
input/output/initializer/declaration/attribute names are checked; topological
preceding-producer checks reject missing/forward references and cycles without
reordering. Node labels may repeat. Empty optional slots remain positional and
create no values. Passthrough input/initializer outputs, zero-node graphs, empty
graphs and graphs without outputs are structurally valid. Initializer defaults
must match input dtype/rank/fixed axes without freezing symbolic/unknown leading
batch sizes. Output/intermediate unknown ranks/dimensions remain valid; missing
intermediate metadata needs no shape inference. Contradictory declarations fail.

Functions/overloads/references, training, subgraphs, sparse storage, unsupported
dtypes and non-dense value kinds fail with node/value context. Harmless model,
graph, node, value and tensor documentation/properties/denotations, descriptive
model domain, and quantization annotations are preserved; annotations enable no
quantized execution. Unknown protobuf fields remain in source wire objects and
are ignored semantically. Generic attribute kinds/payloads are checked, including
proto3 scalar defaults and empty lists. Schema names, arity, optional positions,
operator-specific attributes and numerical support remain RT-07/RT-09's work.
Unresolved standard-domain operators are deliberately retained in this structural
graph; no capability record is marked implemented. Errors wrap internal
ErrInvalidModel/ErrInvalidArgument/ErrUnsupported/ErrLimit and retain context/
decoder causes for errors.Is. Public structured errors remain planned.

Validation on Go 1.27.2, Darwin/arm64 passed: gofmt, `go mod tidy -diff`,
`go mod verify`, `go build ./...`, `go vet ./...`, `go test ./...`,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`, and
`go test -race ./...`, with the writable cache and offline dependency mode.
Table tests cover branches/joins/shared constants, multiple IO, defaults,
passthroughs, empty/scalar/unknown metadata, imports, all producer/reference
failures, unsupported features, generic attributes, immutable wire/accessor
snapshots, concurrent readers, aggregate boundary budgets and cancellation
through construction/final publication. The noncanonical-wire test distinguishes
actual encoded length from proto.Size. The pinned 25-schema/checker fixture,
local Markdown paths, contract Go-block formatting and diff whitespace checks
pass. Ordinary tests need no Python/protoc/native runtime.

RT-07 must consume all nodes, resolve versioned schemas/capability status, validate
operator-specific attribute/type/rank/optional-slot combinations, reconcile each
Value's declarations with inferred/deferred metadata, and plan without numeric
output allocations. It must use already-owned attribute constants rather than
trying to decode copied protobuf pointers. Deep record accessors may copy large
numeric attribute lists: reserve snapshot payloads before calling them (Usage
provides a conservative whole-graph bound) and release redundant snapshots.
RT-08 owns sessions, input symbol
bindings/default overrides, requested-output copies and per-run clone budgets.

### [x] RT-07 — Resolve versioned operators and plan types/shapes

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

**Handoff (RT-07):** Implementation revision
`02a1182b27a08cbc93532803c0d83bcf54249fe5` introduces
`internal/operators/plan.go`; retrieve it with
`git log -1 --format=%H -- internal/operators/plan.go`. The focused node-metadata
preservation revision is `a692f9c5f1d88b01968062818651fd350a8d4bd3`;
retrieve it with `git log -1 --format=%H -- internal/operators/step.go`.
Changed files are the new `internal/operators/` compiler, registry, plan/value/
step/limits/validation and matching tests; `internal/capability/` manifest
records; `internal/runtimeerror/` diagnostic/sentinel and tests; root
`capabilities.go`, capability aliases and tests, `error.go` and `errors.go`;
`capabilities.json`; graph Constant.Rank and its test; `tools/check_profile.py`;
README, TODO, package GoDoc, development/runtime-contract documentation and this
roadmap. Backend remains published `neuralnetwork v0.7.0`, commit
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, protobuf v1.36.12, and
ONNX v1.16.2 at `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`.
Module baseline/toolchain remain Go 1.26.1/1.27.2; no dependency, backend or
local replacement changed.

Public API adds `Capabilities() []Capability`, aliases Capability,
CapabilityArity (Min/Max/OptionalSlots), CapabilityAttribute
(Type/Required/Default JSON), and Error with Error/Unwrap/Is methods, plus
ErrInvalidModel/ErrUnsupported/ErrLimit. Existing ErrInvalidArgument retains its
identity contract and is now shared internally. Query results deep-copy all
operator records, including defaults/maps/slots, and include planned statuses.
The root embeds the canonical JSON and passes those same records to the internal
registry; no generated support table duplicates it. Only Identity-13 and
Constant-13 are marked implemented, explicitly for static planning. Constant
accepts exactly one dense float32/int64 `value` TENSOR; scalar/list attributes
remain RT-09 work. There is still no public loader, model, session or Run API.

Internal APIs are `NewRegistry([]capability.Record) (*Registry, error)`,
`Registry.Resolve(domain, operator string, imported int64) (capability.Record,
error)`, and `Compile(context.Context, *graph.Graph, *Registry, Limits)
(*Plan, error)` in `internal/operators`. Registry snapshots records and chooses
the newest applicable schema before enforcing its tested interval; no older
fallback or future-opset widening occurs. Production imports remain exactly 13.
Plan methods Metadata/Inputs/Outputs/Values/Value/Steps return independent
snapshots; Usage reports aggregate logical load/initializer bytes. Steps carry
schema/import/node context, immutable Constant references, value-table indices,
and preserved node docs/properties. Values retain original declarations as well
as merged type/shape information and a Deferred shape flag. Missing intermediate
metadata is inferred. Scalars, fixed zero and unknown/symbolic dimensions remain
distinct. Identity propagates shape/type, and Constant derives concrete metadata
from the already-owned tensor. Defaulted inputs retain their override contract.

The plan retains no protobuf, decoder or source Graph. It shares immutable
constants and performs no numeric output allocation or Constant.Clone.
Compilation continues Graph.Usage accounting, reserving one conservative
whole-graph bound before each of five deep accessors and reserving later shape/
dimension payloads before allocation. Constant.Rank avoids an unreserved shape
query. Limits contains MaxLoadBytes/MaxTensorBytes/MaxRank/MaxDimension with
1 GiB/64 MiB/32/2^31-1 defaults; use the same effective limits as graph.Build.
Static sizes use backend ValidateShape without allocating numerical tensors.
Initializer usage is unchanged. Transient reservations stay conservatively
charged; headers/capacities/GC retention remain outside logical byte accounting.
Every declared node is validated, including unused paths. Errors deterministically
retain cause/classification and node/import/input-shape context. Unknown or
unavailable diagnostic shapes are nil; scalar shapes are nonnil and empty.

Validation passed on Go 1.27.2, Darwin/arm64: gofmt, `go mod tidy -diff`,
`go mod verify`, `go build ./...`, `go vet ./...`, normal tests,
`CGO_ENABLED=0 go test ./...`, `CGO_ENABLED=0 go test -tags=purego ./...`, and
`go test -race ./...`, using the writable cache and offline dependency mode.
Tests cover actual Softmax-11/13 defaults, Squeeze-11/13 axes attribute/input
differences under isolated resolver profiles, older Conv-11 selection at import
13, rejection of untested future imports, optional slots, wrong arity/attributes/
types/outputs, static rank/dimension/byte/product failures, dead unsupported nodes,
absent intermediate metadata, unknown/symbolic/scalar/empty shapes, tensor-only
Constants with exact int64 values, passthrough/default contracts, deep-copy and
concurrent reads, numerical-allocation regression, exact budget boundaries,
every observed cancellation boundary and errors.Is/As. The pinned 25-schema/
default/status/checker audit and unchanged fixture hash, local Markdown targets,
contract Go-block formatting and diff checks also pass. Ordinary Go tests remain
offline and need no Python/protoc/native runtime.

RT-08 must bind input symbols/default overrides, verify every produced value's
inferred metadata **and original declarations**, and budget all backend constant
clones and output copies before allocation. Deferred describes shape precision,
not whether all runtime declaration checks are complete. Extend the diagnostic
formatter's current `onnx compile` stage for loading/running. RT-09 owns numerical
families and scalar/list Constant materialization; charge those new constant
payloads against both initializer and load budgets. RT-17 owns production
profile expansion; synthetic resolver tests do not broaden the accepted imports.

### [x] RT-08 — Implement sessions and serial graph execution

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

**Handoff (RT-08):** Implementation revision
`f0ffed51f7f05db95a211b99ba5788008aa43940` introduces `model.go` and
`session.go`; retrieve it with `git log -1 --format=%H -- model.go`.
Changed files are root Model/Session/Options/Limits/RunOptions/NamedTensor/
ExternalDataResolver types, private execution plan/evaluation/run budget, matching
tests and context/reader fixtures; public metadata/error/capability GoDoc and
manifest; internal ReadModelWithSize, Plan.SnapshotBytes, staged diagnostics and
tests; three original `testdata/sessions/` models with provenance/README; the
profile checker/provenance description; README, TODO, NEEDS, development/runtime
contract documentation and this roadmap. Dependencies remain published
`neuralnetwork v0.7.0` at `6048452c57a0a293db4591cd2e2febd7d6b743a3`, protobuf
v1.36.12 and ONNX v1.16.2 at `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`.
Go baseline/toolchain remain 1.26.1/1.27.2. No dependency, backend, native runtime
or local replacement changed.

Public APIs are `Load(context.Context, io.Reader, Options) (*Model, error)`,
`LoadBytes(context.Context, []byte, Options) (*Model, error)`, and
`LoadFile(context.Context, string, Options) (*Model, error)`; Model methods
`Metadata() Metadata`, `NewSession() (*Session, error)`, and `Close() error`;
Session methods `Run(context.Context, map[string]*Tensor, RunOptions)
([]NamedTensor, error)` and `Close() error`. Options contains Limits and
ExternalDataResolver; RunOptions has Outputs (`[]string`), NamedTensor has Name
and Tensor (`*Tensor`). Limits implements all fourteen fields/defaults in the
runtime contract; zero selects defaults and negative/host-invalid values fail
before I/O. ExternalDataResolver reserves `Resolve(context.Context, string,
int64, int64) (io.ReadCloser, error)`, but a nonnil option and all external tensors
fail as unsupported until RT-14; no resolver is invoked. ErrBusy/ErrClosed are
direct state sentinels. Error adds Stage (`load`, `run`, or default `compile`),
maps wire/tensor/graph/planner classifications, and preserves causes/node/value/
shape context through errors.Is/As, including joined operational failures.

The loader checks options/context before I/O and publishes only a complete plan.
Load never closes its reader; LoadFile owns/closes its file. The new internal
`decode.ReadModelWithSize(context.Context, io.Reader, decode.Limits)
(*onnxpb.ModelProto, int64, error)` reuses bounded reads and returns the actual
encoded size, including noncanonical varints. Existing ReadModel still works.
`Plan.SnapshotBytes(context.Context) (int64, error)` estimates copied record/index/
dimension payload without allocations or numerical clones, with cancellation and
checked sums. Executable preparation reserves this before deep plan accessors
and the public metadata copy; it shares immutable constants/string data and
retains no protobuf, decoder, source Graph or compiler Plan. RT-07's conservative
compilation reservations remain; executable preparation adds metadata payload,
without multiplying prior reservations by whole-plan snapshots.

Every run creates private table/bindings/budget state and executes every node in
declared order, including unused output paths. Identity aliases immutable values;
Constant and initializer/default binding clone budgeted backend storage. All
graph inputs, including weights, are ordinary eligible values. Supplied/defaulted
inputs validate dtype/rank/fixed axes and shared symbols; each value also checks
both inferred Info and every original Declaration. Symbols bind afresh each run,
including zero batches; refinement must never erase an original symbol. Requested
names must be unique declared outputs, with a constant-time membership lookup;
empty requests use declaration order, otherwise request order. Scalar/empty,
zero-node input/initializer passthroughs and graphs without outputs work. Every
published output is independently cloned and survives later runs and any Close.

Run/Close use a nonblocking session TryLock. Failed/cancelled calls discard local
state and release the guard; successful Close is idempotent and later Run returns
ErrClosed. Model.Close/NewSession linearize under a model lock; existing sessions
retain their plan. A plan lock enforces MaxSessions and releases slots on successful
Session.Close, including after Model.Close. Metadata returns independent snapshots
and remains inspectable after model closure.

Per-run live bytes include resident model tensors once, every bound/default source,
every intermediate/alias retained through Run, and every output copy. All clones
reserve destinations before allocation. Audited released tensor.go Clone copies
directly into owned destination storage without numeric staging; current handlers
therefore reserve zero numerical workspace. The budget checks tensor/workspace/
live bounds and overlapping reservations with subtraction-based checked sums.
RT-09 must reserve each node's complete audited scratch at once before calling a
kernel and release workspace afterward; output destinations must remain disjoint
from immutable sources/constants. Runtime shape metadata, Go headers/capacities,
GC retention and earlier caller-retained outputs remain outside logical tensor
limits. Context checks cover loading/planning/runtime loops, before/after each
node/copy and final publication. Backend/protobuf calls remain synchronous with
no in-kernel interruption or partial published results.

Validation passed on Go 1.27.2, Darwin/arm64: gofmt, `go mod tidy -diff`,
`go mod verify`, build, vet, normal tests, `CGO_ENABLED=0 go test ./...`,
`CGO_ENABLED=0 go test -tags=purego ./...`, and `go test -race ./...`, with the
writable cache and offline dependency mode. Explicit/table tests cover branching,
multi-input/output, defaults/overrides/runtime weights, output order/subsets,
passthrough/scalar/empty/unknown metadata, exact int64 and float bit preservation,
original input declarations and refined output/intermediate constraints, fresh shared symbols,
independent copies/constants, reader ownership/errors/limits/noncanonical lengths,
session caps, busy/concurrent sessions, Close/NewSession races, operator failure,
every observed load/run cancellation boundary, exact load/initializer/live/tensor/
structural limits, workspace/overflow checks and recovery. The pinned 25-schema/
default/checker audit, unchanged profile model hash, three pinned-checker session
fixtures and provenance hashes, local Markdown targets, contract Go-block format,
changed Go style checks and diff whitespace checks pass. Ordinary tests require
only Go and checked-in artifacts.

RT-09 is next: integrate M1 numerical handlers and scalar/list Constant forms,
preserving all named graph inputs and lifecycle guarantees. Charge any new constant
materialization against both load and initializer budgets; supply audited per-node
destination/scratch reservations for copying backend adapters. No arithmetic,
activation, shape or CNN execution was added by RT-08.

### [x] RT-09 — Integrate M1 arithmetic, activations, and constants

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


**Handoff (RT-09):** Implementation revision
`4a618955d26dc588fe384b91355d13ca52a2be63`, based on RT-08
`aca84a7bded56138243e59f8d30936b068861a54`, on `rt-09-m1-operators`.
The final task revision is the commit introducing this handoff; retrieve it with
`git log -1 --format=%H -- ROADMAP.md`. Backend remains published
`neuralnetwork v0.7.0` (`6048452c57a0a293db4591cd2e2febd7d6b743a3`);
protobuf v1.36.12, Go 1.26.1 baseline/Go 1.27.2 toolchain, no new dependencies.

Changed implementation: [numerical_call.go](numerical_call.go),
[evaluation.go](evaluation.go), operator compiler/registry/Step/Plan/limits and
[shape_inference.go](internal/operators/shape_inference.go), TensorDecoder,
graph builder/Constant. Changed verification: matching operator/registry tests,
capability/model assertions, [numerical_call_test.go](numerical_call_test.go),
[m1_integration_test.go](m1_integration_test.go),
[check_m1.py](tools/check_m1.py), profile checker, and
[M1 models/provenance](testdata/m1/provenance.json). Updated canonical manifest,
README, TODO, NEEDS, package GoDoc, development and runtime contract.

No public API signatures changed. Internal Step adds `Gemm tensor.GemmConfig`
and `Axis int64`, storing resolved defaults. Its copying accessors remain
immutable; Plan.SnapshotBytes includes these fixed fields without numerical
clones. New internal decoder method:
`Materialize(ctx context.Context, shape []int64, floats []float32, integers []int64, dataType tensor.DataType) (out *tensor.Tensor, err error)`.
It validates scalar/list shapes, types and counts and reserves both aggregate
load/initializer bytes before copying borrowed values into owned storage. Escape
analysis confirms the graph adapter's single-element literals remain on stack.
All methods remain with their primary types; numerical planning/execution use
focused helpers and a call-local numericalCall.

Implemented Add/Sub/Mul/Div with multidirectional broadcasting, scalar/empty
values and IEEE float32 division; MatMul vector promotion/batch broadcasting and
empty contractions; rank-two Gemm with all transpose flags, alpha/beta defaults
1, omitted/empty C and scalar/vector/row/column/full unidirectional bias.
Supplied C is validated even with beta zero; zero beta skips nonfinite bias
arithmetic in agreement with the pinned reference. Relu/Sigmoid/Tanh preserve
shape at every rank. Softmax-13 defaults axis -1 and normalizes only the chosen
axis, including nonfinal axes. Constant accepts exactly one dense float32/int64
value tensor, float/int scalar or float/int list; int64 never uses float storage.
Known type/rank/dimension/attribute/arity errors fail at load; remaining concrete
broadcast/contraction/bias errors identify the node at Run and allow recovery.
Static inference only creates metadata, including partial symbolic dimensions;
large-shape allocation regressions cover arithmetic, products and activations.

Each numerical node reserves its full destination and complete audited scratch
once before allocation/backend calls. Matching nonempty Add/Sub/Mul reserve
4 output sizes, Sigmoid/Tanh 3, Relu 6 (bridge plus accounted NaN correction),
Gemm copied A/B plus two product buffers and an extra B for both transposes;
empty-contraction Gemm uses one output-sized temporary. MatMul reserves its full
temporary output plus the sum of every batch's two copied inputs and two product
buffers. Broadcast arithmetic, Div, Softmax and empty output paths use no numeric
scratch. Checked sums/products reject overflow. Immutable inputs/constants remain
disjoint from destinations; requested outputs are independent owned clones.
Workspace releases on every exit; cancellation checks surround kernels and poll
Relu correction every 1,024 values. No in-kernel interruption is promised.

The 81 original IR 8/opset 13 checker-validated models cover every M1 operator,
all 24 Gemm flag/bias combinations, dynamic graph weights, initializer-backed
MLP, residual branches, scalars/empties/nonfinite values and exact int64 extremes.
Eighty match ONNX 1.16.2 reference outputs; empty-axis Softmax retains its actual
reference rejection and independently expected empty output as an explicit
extension. Relu propagates NaN and maps either signed zero to positive zero as
chosen in RT-01. Comparisons use finite atol/rtol 1e-5, NaN classification,
infinity signs and specified zero signs. Provenance SHA-256:
`19227f9920883b065788cc8d1a2acdfa803c632a8bc0b4bc24ff720157d315ce`.
The original profile model remains
`e7fed3b2d29d884d48c43427b69c08c6514ed6dcf0c7a2bdd002d5f46b0ad469`.

Validation passed on Go 1.27.2 Darwin/arm64 using the writable temporary cache
and offline dependencies: gofmt/go fmt, `go mod tidy -diff`, `go mod verify`,
build, vet, normal tests, `CGO_ENABLED=0 go test ./...`,
`CGO_ENABLED=0 go test -tags=purego ./...`, and `go test -race ./...`.
Also passed pinned 25-schema/default/status audit, deterministic M1 regeneration,
profile/fixture hashes, local Markdown targets, contract Go-block formatting,
handwritten if/var spacing/type-method colocation, and diff whitespace checks.
Tests include exact workspace/live boundaries for each kernel path, aggregate
initializer/Constant/load boundaries, every observed numerical cancellation
boundary, contextual dynamic failure/recovery and input/output ownership.
Ordinary Go tests require no Python, network or native runtime.

RT-10 is next: expand these focused M1 artifacts into the complete fixture and
differential conformance harness. RT-11 adds runnable public examples and multiple
batch sizes. Shape/CNN/external data, other versions/dtypes and optimization
remain explicit gaps; no backend, training or optimizer changes were made.

### [x] RT-10 — Build the fixture and differential conformance harness

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

**Handoff (RT-10):** Implementation revision
`70d5ef59161fbe0a2c290e6ee9b7448328618b68`, based on RT-09
`0f3c29f39316bed55fb795dcd1d3b407c19b0c26`, on `rt-10-conformance`.
The final task revision is the commit introducing this handoff; obtain it with
`git log -1 --format=%H -- ROADMAP.md`. That revision also aligns the deferred
opset-18 identity fixture with NN-12's omitted-axes encoding. Backend remains
published `neuralnetwork v0.7.0` at
`6048452c57a0a293db4591cd2e2febd7d6b743a3`; protobuf v1.36.12, Go
1.26.1 baseline/Go 1.27.2 toolchain and production capabilities are unchanged.
No public/exported signatures, production logic or numerical adapters change.

Changed files: root `conformance_artifact_test.go`, `conformance_value_test.go`,
`conformance_comparison_test.go`, `conformance_case_test.go` and
`conformance_suite_test.go`; `tools/check_conformance.py`; models, named typed/raw
TensorProto values, provenance, full upstream audit and guide under
`testdata/conformance/`; README, TODO, NEEDS, development/runtime contract,
third-party attribution and this roadmap. Existing RT-09 models/provenance stay
unchanged. Each test-only type and all its methods share a file; the suite's
constructor uses a local concrete value and explicit named-result returns.

The reusable runner hashes and decodes model/value artifacts offline through
the shared bounded decoder, validates recorded envelope/schema identities,
loads/runs public sessions and compares retained outputs after closing both
owners. Names/output order, type, shape/count and int64 payloads are exact.
Finite floats use `abs(got-want) <= 1e-5+1e-5*abs(want)`, NaN classification,
exact infinity signs and explicit per-case signed-zero policy. Unexpected
nonfinite values fail finite expectations. Nonempty MatMul/Gemm zero signs are
relaxed; empty MatMul contractions require positive zero. The numeric regression
injects `[4,7]` into the hand-calculated Add output `[4,6]` and requires the
same fixture comparison path to report input/output names, flat offset 1 and
row-major position `[1]`. Separate cases assert exact integers above 2^24 and
both extrema, scalars/empties, shape/type/name errors and nonfinite handling.

There are **86 reference agreements**, **one empty-axis Softmax extension** and
**nine explicit ErrUnsupported rejections**; no skips count as support. The
agreements reuse RT-09's 80 reference results and add six original independently
hand-calculated cases, including a two-output branch. Six deferred cases retain
actual Gather-empty, pooling-ceil/NaN, BatchNormalization-9 stored-statistics/
zero-domain and opset-18 reduction observations. BN literals independently
cross-check ORT; evaluator momentum and ORT's zero-domain difference remain
visible. All these deferred models are rejected at load and establish no
M2/M3/reduction execution support.

Pinned source audit: official ONNX v1.16.2 commit
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557` contains **1,422** backend models,
including 352 IR-7/opset-13 models and **zero** exact IR-8/opset-13 models.
The complete path/hash/envelope audit is checked in. Three selected upstream
M1 operator models and their values retain original bytes as version-rejection
tests. This is the pinned suite's availability constraint: original checked
IR-8/opset-13 graphs provide positive coverage; no upstream version numbers
were rewritten and no profile expansion was made. Copied artifacts retain
Apache-2.0 attribution and the included upstream license; no root NOTICE exists.
Original fixture provenance accurately records that this repository has no
LICENSE rather than inventing a formal license from fixture origin.

Validation passed on Go 1.27.2 Darwin/arm64: gofmt, `go mod tidy -diff`,
`go mod verify`, build, vet, normal tests, `CGO_ENABLED=0 go test ./...`,
`CGO_ENABLED=0 go test -tags=purego ./...`, and `go test -race ./...`, with
`GOCACHE=/private/tmp/onnx-runtime-gocache` and offline Go dependency mode.
Pinned Python 3.9.6 / ONNX 1.16.2 / NumPy 1.26.4 / protobuf 4.25.3 /
ORT 1.19.2 regeneration and an unchanged second verification pass reproduce all
fixtures/audit using the archive command in the fixture guide. The original
25-schema/default/capability/profile checker and 81-model M1 reference checker,
handwritten Go spacing/type-method colocation, contract Go-block formatting,
local Markdown targets and diff checks pass. Normal Go tests invoke no Python,
network or shared runtime. Hosted Linux verification is required before handing
the stacked PR to the next task.

RT-11 is next: public runnable MLP/branch examples, at least two batch sizes,
and an integration report. Reuse this harness/corpus, preserve explicit
extension/rejection counts and the unchanged version envelope. RT-12/RT-13 own
deferred shape/CNN adapters; NN-12 reductions remain later opset-selected work.

### [x] RT-11 — Demonstrate and document the first working runtime

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

**Handoff (RT-11):** Implementation revision
`e527d778ace196ce6ddb8362cc349501ec6859bf`, based on RT-10
`dc9898b48e23c6fab82429f8f0039718342b51bc`, on `rt-11-inference-examples`.
The final task revision is the commit introducing this handoff; obtain it with
`git log -1 --format=%H -- ROADMAP.md`. Backend remains published
`neuralnetwork v0.7.0` at `6048452c57a0a293db4591cd2e2febd7d6b743a3`,
protobuf v1.36.12, Go 1.26.1 baseline/Go 1.27.2 toolchain. No production runtime
logic, dependency, public API signature, capability status or contract changes.

Changed files: [examples/inference/main.go](examples/inference/main.go),
[inference_integration_test.go](inference_integration_test.go),
[tools/check_inference.py](tools/check_inference.py), the two models, 12 named
TensorProto values, provenance and guide under [testdata/inference](testdata/inference/README.md),
[getting started](docs/getting-started.md), [integration report](docs/integration-rt11.md),
README, TODO, NEEDS, development/runtime-contract documentation and this roadmap.
The example uses actual `LoadFile(ctx, path, Options{})`, `Model.Metadata`,
`Model.NewSession`, `NewFloat32(shape, values)`,
`Session.Run(ctx, map[string]*Tensor, RunOptions{})`, named result/value accessors
and both `Close() (err error)` methods. Named-error deferred cleanup joins close
failures. No reference tool is imported or invoked by normal Go code/tests.

Both original IR-8/opset-13 models declare float32 `["batch", 2]` I/O and keep
all four weight/bias initializers inline. MLP is Gemm/Relu/Gemm/Softmax. Branch
adds the named `residual` input, joins it with shared logits and fans out into
named `probabilities` and `delta` outputs. Four actual ONNX 1.16.2 evaluator
returns cover both graphs at batches 1 and 3, independently cross-checked by
NumPy formulas. They are additional to RT-10's unchanged 86 agreements, one
extension and nine explicit rejections. No complete-profile/ORT/performance
claim is made; original provenance records the existing missing repository
LICENSE rather than inventing a license.

The generator pins Python 3.9.6, ONNX 1.16.2 source
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, NumPy 1.26.4 and protobuf 4.25.3,
checks exact tool versions/full ONNX checker acceptance, and reproduces all
model/value hashes. MLP SHA-256:
`04df066d0a12b0e6ddf2ba89e9e452768939a745ddd574cfb39780d0e14c06f9`;
branch: `3c8710155dd18231cdfcc730d0a98b2d6c158dfeb8c7a47c1f3d0765982f4446`;
provenance: `9fa5307cf7e12b632d1057f6d6cac8c8f799ae8dcc1fd21ef5267ea95058feb7`.
Its generated getting-started table contains every implemented M1 record's
types, schema, arity/optional slots, attributes/defaults and restrictions from
`capabilities.json`. Export guidance preserves the exact IR/import envelope and
explains unsupported-feature diagnostics rather than trusting exporter defaults.

Offline integration tests reuse RT-10's artifact/typed-value/comparison helpers:
hash/envelope/type/shape/name/count checks are exact; finite floats use
atol=rtol=1e-5 with explicit nonfinite/zero-sign rules. They validate symbolic
metadata, run both batches through one session per model, request reversed named
output order and compare retained outputs after later runs and both Close calls.
Two sessions per shared numerical model also start concurrently at batches 1
and 3; outputs compare after joining, with race checks passing. Owned outputs,
immutable weights, finite budgets and cancellation granularity are unchanged.

Validation passed on Go 1.27.2 Darwin/arm64 with the temporary GOCACHE and offline
Go dependency mode: gofmt/go fmt, `go mod tidy -diff`, `go mod verify`, build,
vet, normal tests, cgo-disabled tests, cgo-disabled purego tests and race tests.
All four cgo-disabled CLI graph/batch combinations and a purego branch run pass.
A fresh clone of the implementation revision ran cached dependency download,
the full cgo-disabled suite and all four examples offline with a clean tree.
Pinned profile/schema, 81-model M1, 96-case conformance/1,422-model upstream and
RT-11 regeneration/table checks pass, as do handwritten if/var spacing and
type/method colocation, contract Go-block formatting, local Markdown targets
and diff checks. [Hosted Linux verification](https://github.com/itsmontoya/onnx/actions/runs/42425356391)
also passed formatting, tidy, build, vet, normal, cgo-disabled, purego and race
checks at the implementation revision. The final stacked PR must retain green
checks before the next task starts.

RT-12 is next: consume these reproducible MLP fixtures and add shape operators
and a shape-heavy graph with named I/O/reference outputs for at least two batch
sizes. Publish its exact accessible runtime revision/backend version/hashes to
unblock [NN-13](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-13.md).
RT-11 alone does not unblock it. Shape/CNN/external data, broader profiles,
optimization and the root license decision remain separate tasks; none blocks
the working M1 examples demonstrated here.

## M2 and M3: useful model coverage

### [x] RT-12 — Integrate shape tensors and transformations

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

**Handoff (RT-12):** Accessible implementation revision
[`3b2895ab233b28c0c7fc731e5927e82c99416fcf`](https://github.com/itsmontoya/onnx/tree/3b2895ab233b28c0c7fc731e5927e82c99416fcf),
based on RT-11 `93babe6159cd80b0fc6f940657de23fb857bb404`, on
`rt-12-shape-operators`; [stacked PR #12](https://github.com/itsmontoya/onnx/pull/12)
targets `rt-11-inference-examples`. The final task revision is the commit
introducing this completion record; retrieve it with
`git log -1 --format=%H -- ROADMAP.md`. Backend remains published
`neuralnetwork v0.7.0` at `6048452c57a0a293db4591cd2e2febd7d6b743a3`,
protobuf v1.36.12, Go 1.26.1 baseline/Go 1.27.2 toolchain. No local replace,
backend edit or exported runtime signature change.

Changed production files: [shape_call.go](shape_call.go), evaluation.go,
internal/operators/compiler.go, shape_planner.go, shape_rules.go, step.go,
plan.go and registry.go; [capabilities.json](capabilities.json) now records
exactly 20 implemented operators. Tests cover those boundaries in
shape_call_test.go, m2_integration_test.go, inference_integration_test.go and
the existing capability/model/planner suites. [tools/check_m2.py](tools/check_m2.py)
and [testdata/m2](testdata/m2/README.md) add exact typed fixtures; existing
inference/conformance/profile tools and documentation track current statuses.
The public example gains `-graph shape`; [inference_benchmark_test.go](inference_benchmark_test.go)
provides Load/First/Warm benchmarks. README, TODO, NEEDS, runtime/development/
getting-started guides and the [NN-13 handoff](docs/handoff-nn13.md) record the
implemented contracts and unchanged copying baseline.

All eight M2 schema-13 operators execute with float32/int64 data as applicable.
Shape produces exact int64 dimensions from each concrete runtime input. Flatten
uses the split-axis products (default 1); Transpose defaults to full reversal
and validates complete unique nonnegative permutations. Reshape-13 accepts a
rank-one int64 shape, copies same-position input dimensions for 0, infers at
most one -1, allows an unambiguous inferred zero, and rejects ambiguous zero
products, dimensions below -1, repeated -1, invalid products and allowzero.
Squeeze/Unsqueeze consume schema-13 rank-one int64 axes inputs. Omitted Squeeze
axes remove all concrete unit dimensions; explicit-empty axes are identity.
Unsqueeze axes are mandatory and normalize in output rank. Gather preserves
prefix + full indices shape + suffix, handles arbitrary-rank/scalar/empty int64
indices and negative values, and validates every index even for empty results.
Concat requires a nonempty same-type/rank list, checks non-axis dimensions and
checked axis sums, including otherwise-empty tensors.

The compiler's focused shapePlanner propagates metadata without allocating
numerical node outputs. It reserves copied immutable initializer/Constant/
Identity int64 access before validating known values; overridable graph-input
defaults remain dynamic. Shape/axes metadata copies are also bounded before
allocation. Genuinely runtime values/ranks defer to Run; missing intermediate
value_info is allowed. Runtime shapeCall reserves int64 accessor/Shape staging
before copying, then adds the complete destination while staging remains live.
Audited v0.7.0 transforms/Concat/Gather Into paths require no additional
numerical scratch. Existing M1 reservations, errors, cancellation boundaries,
model/session ownership/defaults, independent output copies and concurrency
remain intact. Separate calls create fresh value/symbol tables. No storage
reuse, constant folding, backend bridge, NN-13 or RT-16 optimization is added.

The [66 M2 fixtures](testdata/m2/provenance.json) include **60 evaluator
agreements**, **two empty-Flatten evaluator rejections**, and **four empty-Gather
shape differences**, each independently checked with NumPy. Int64 extrema and
values above 2^53 remain exact; transform floats preserve NaN payload bits,
infinity signs and signed zeros. Static invalid constant values fail Load;
dynamic invalid axes/reshape/Gather values fail contextually and recover.
Tests cover omitted versus explicit-empty axes, output-rank axes, default and
invalid permutations, scalars/empties, input-default override, sum/product
bounds and exact staging/destination live budgets before numerical allocation.
The original conformance suite keeps 86 agreements and one Softmax extension,
now executes its previously deferred Gather case as one independent expected
shape difference, and retains eight genuine unsupported-model rejections.
Actual evaluator `[0]` results remain in provenance; no difference counts as
unqualified agreement. NEEDS makes the NN-07/08 exceptions normative.

The shape-heavy graph composes Shape → Gather → Unsqueeze/Concat → Reshape →
Unsqueeze → Transpose → Squeeze → Flatten at batches 1 and 3, without inferred
intermediate value_info. Named output `target` remains int64 `[batch,-1]`;
`reshaped` and `flat` compare exactly with independent row-major formulas.
One shared session runs both batches; separate sessions start concurrently;
retained outputs compare after later runs/Close. Omitted Squeeze also changes
rank correctly through batches 1 → 3 → 1 without stale state.

The [NN-13 handoff](docs/handoff-nn13.md) links the published measurement plan,
accessible runtime/backend revisions, model and every named input/output hash,
regeneration, examples and runnable baseline commands. Existing MLP SHA-256 is
unchanged: `04df066d0a12b0e6ddf2ba89e9e452768939a745ddd574cfb39780d0e14c06f9`;
shape model: `64800db535d7fd83d30f98d7c30ccf402c00e444a082a9d0865c812ed79b2cb6`;
inference provenance: `547933bf4e3be8e5e57e537820a1529822fcf3ddda17d953a83e98eca887ff2c`;
M2 provenance: `4bd43bb170feb5427a4f596ba4a171ff1da03cd251558e80e22adb1121fba59f`.
Pinned Python 3.9.6 / ONNX 1.16.2 source
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557` / NumPy 1.26.4 /
protobuf 4.25.3 reproduces these files; the conformance audit additionally uses
ORT 1.19.2. No repository LICENSE is invented for original fixtures.

Validation passed on Go 1.27.2 Darwin/arm64 with temporary GOCACHE and offline
Go dependencies: gofmt/go fmt, tidy-diff, mod verify, build, vet, normal,
CGO_ENABLED=0, CGO_ENABLED=0 purego and race suites. All six public CLI
model/batch combinations and a purego shape run pass. A fresh implementation
clone passes cached dependency download, the cgo-disabled suite and MLP/shape
examples for both batches offline with a clean tree. Pinned profile/defaults
(25 schemas, 20 implemented), unchanged 81-model M1, 66-case M2, 96-case
conformance/1,422-model upstream audit and three-model/six-case inference/table
reproduction pass. Handwritten spacing/type-method colocation, contract Go-block
formatting, local Markdown targets and diff checks pass. The baseline benchmark
smoke executes Load/First/Warm for both graphs/batches; no optimized result,
copied-byte measurement or peak-memory report is claimed.
[Hosted Linux verification](https://github.com/itsmontoya/onnx/actions/runs/42427087896)
passed formatting, tidy, build, vet, normal, cgo-disabled, purego and race checks
at the implementation revision. Final PR-head verification must remain green.

RT-13 is next: add the bounded CNN adapters and composed CNN fixture against
NN-09/10/11, retaining their reference differences. NN-13 may independently
consume this published handoff; RT-16 remains later. External weights, broader
profiles, reductions, optimization and the root license decision remain separate
maintainer-controlled work. The final stacked PR must be open, unmerged, based
on RT-11 and green before starting the next task.

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
