# Development and verification

RT-02 establishes module `github.com/itsmontoya/onnx`, package `onnx`, with a
Go 1.26.1 language baseline and the selected Go 1.27.2 toolchain. The pinned
`neuralnetwork v0.7.0` module requires Go 1.26.1; its SIMD and `x/sys`
dependencies require Go 1.25.0. The pure-Go protobuf dependency is
`google.golang.org/protobuf v1.36.12` (Go 1.23). No local path replacements
or workspace are required.

The root package provides RT-04's immutable typed Tensor facade and metadata
records, along with a dependency smoke test. The smoke test exercises exact
int64 tensor storage and known protobuf wire bytes; tensor tests verify copied
ownership, exact values, invalid arguments, and concurrent readers.
RT-03 adds checked-in ONNX wire bindings and bounded internal protobuf decoding.
RT-05 adds shared inline tensor conversion with aggregate payload accounting.
RT-06 adds whole-graph structural validation and immutable named graph records.
RT-07 adds versioned schema resolution, immutable execution plans, capability
queries and structured compilation errors for a minimal Identity/Constant subset.
RT-08 adds public Load/LoadBytes/LoadFile, shared Models and private serial
Sessions with cancellation, finite budgets, defaults and owned requested outputs.
[The capability manifest](../capabilities.json) marks only Identity and dense
tensor-valued Constant implemented; numerical families remain RT-09 work.

Public `NewTensor`, `NewFloat32`, and `NewInt64` delegate storage validation and
allocation to the pinned backend. They copy caller shapes/payloads; shape and
typed value accessors also copy. No public method exposes a writable backend
pointer. Scalar shapes have one element, while a concrete zero dimension has
zero elements. Constructors check representability rather than a load/run
memory quota. See [the runtime contract](runtime-contract.md) for the implemented
loader/session budgets and ownership boundaries.

## Ordinary builds and tests

Install Go 1.27.2, then run these commands from the repository root:

```sh
go mod download
go build ./...
go fmt ./...
go vet ./...
go test ./...
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go test -race ./...
git diff --check
```

Dependency download initially requires network access. Once dependencies are
cached, ordinary builds/tests use only Go and checked-in artifacts. They need
neither Python nor protoc, Metal, a native inference runtime, or shared libraries.
The race detector is a development check and requires a supported platform and
cgo toolchain; cgo-disabled runtime builds do not acquire that requirement.
The `purego` tag selects the backend's portable CPU path.

[The CI workflow](../.github/workflows/go.yml) runs formatting, module tidiness,
build, vet, normal tests, both cgo-disabled configurations, and race tests on
Linux. Local baseline checks also run on Darwin/arm64. A separate linter is not
configured. Use `go mod tidy -diff` to check module metadata without rewriting it.

## Package boundaries

The [runtime contract](runtime-contract.md#public-surface) defines
the following boundaries. Create each internal package when its implementation
task introduces real behavior, rather than adding empty packages in advance.

| Location | Responsibility | First task |
| --- | --- | --- |
| Root `onnx` package | Public values, metadata, load/model/session facade | RT-04/RT-08 |
| `internal/onnxpb` | Generated ONNX wire types, isolated from the public API | RT-03 |
| `internal/decode` | Bounded protobuf and tensor decoding | RT-03/RT-05 |
| `internal/graph` | Named graph validation and immutable planning | RT-06/RT-07 |
| `internal/operators` | Versioned schema planning and numerical adapters | RT-07/RT-09 |
| `testdata` | Checked-in model/value fixtures and provenance | RT-01/RT-10 |

Internal packages use backend tensors and internal metadata records without
importing the public facade. Concrete numerical storage and kernels remain in
`neuralnetwork/tensor`.

## Regeneration

Regeneration is a development operation, separate from ordinary builds. Install
the official [protoc 33.2 release](https://github.com/protocolbuffers/protobuf/releases/tag/v33.2)
for your platform, then run this command from the repository root with Go 1.27.2:

```sh
go run ./tools/onnxgen
git diff --exit-code -- internal/onnxpb
```

The tool verifies the [vendored schema](../third_party/onnx/README.md) SHA-256,
requires protoc 33.2 and the module's exact protobuf v1.36.12, builds that
version's official protoc-gen-go in a temporary directory, and invokes protoc.
It splits the generated Go AST into one file per type, retaining that type's
methods and enum values/maps together; descriptor initialization is separate.
It preserves the generated protobuf logic instead of maintaining a wire parser.
No schemas need importing. Repeated generation leaves all 36 generated files
unchanged. Generated code retains upstream implementation style; the handwritten
generation and decoding code follows this repository's style guide.

Only optional regeneration requires protoc. Go builds and tests use checked-in
bindings and require neither protoc nor Python. The generator itself can run
offline after module dependencies have been downloaded.

## Internal wire decoding

`internal/decode` provides ParseModel/ReadModel and ParseTensor/ReadTensor.
Each accepts a context and Limits: zero values select 64 MiB serialized bytes
and 64 levels of known-message recursion. Byte input is checked before
unmarshalling. Readers are consumed in chunks of at most 32 KiB, with at most
limit+1 bytes consumed to detect excess, and are never closed. I/O errors retain
their causes, repeated empty reads fail, and cancellation is checked between
reads and before/after the synchronous protobuf call.

Unknown fields are retained. In protobuf v1.36.12, MaxRecursion bounds known
messages; unknown groups instead use protowire's fixed recursive allowance of
10,000, permitting 10,001 nested group levels and rejecting the next level.
Tests exercise both limits. Serialized bytes remain bounded in either case.
This is the official decoder's behavior, not a uniform configurable depth bound.

Decoding verifies wire syntax, not ONNX compatibility. Empty protobuf messages,
unsupported versions/types, graph attributes, and other semantic features remain
wire-decodable for later validation; none establish inference support. The byte
limit bounds serialized input, not protobuf object-tree overhead or total heap.
RT-05 adds the payload ownership/accounting described below. Graph and operator
checks are separate from wire decoding: RT-06 implements the graph checks below,
and RT-07 owns operator resolution.

## Inline tensor conversion

`NewTensorDecoder(ctx, wire proto.Message, encodedBytes int64, limits TensorLimits)`
constructs one internal decoder for an entire decoded model, attribute, or test
TensorProto. Pass the actual original serialized byte count, including unknown
fields and noncanonical wire encodings; the public reader loader tracks that
count through ReadModelWithSize. Proto.Size is not a substitute for the original
byte count.
Use ParseModel/ReadModel or ParseTensor/ReadTensor first to enforce wire bounds.
Pass the complete decoded model so all coexisting payloads are charged before
any numerical conversion. Foreign schemas and sparse tensors are rejected.

`Decode(ctx, value *onnxpb.TensorProto) (*tensor.Tensor, error)` converts any
TensorProto registered within that message, including initializers and tensor
attributes. It returns the pinned backend's owned tensor, keeping the public
facade private to package onnx. Do not mutate the wire message during decoder
use; after conversion, source wire bytes/values can be released independently of
the owned tensor. The decoder is local to a load and is not concurrent. Repeated
Decode calls allocate and charge separate owned copies.

Float32 float_data and exact int64 int64_data must match the shape's element
count. Raw data must match its byte count and is decoded in little-endian order.
Float bits, including NaN payloads and signed zero, and all integer bits survive
conversion. No dimensions means one scalar element; any zero dimension means
zero elements. Empty tensors may omit payloads or use empty raw data. Empty
repeated fields have no protobuf presence, so an empty typed field cannot be
distinguished from an absent one after unmarshalling. A present raw field with
nonempty typed data is rejected. Wrong typed fields, invalid shapes/counts,
segments, unsupported dtypes, sparse values, and all external storage/location
metadata are rejected; RT-14 must add an explicit resolver.

TensorLimits selects defaults of 1 GiB MaxLoadBytes, 256 MiB
MaxInitializerBytes, 64 MiB MaxTensorBytes, rank 32, dimension 2^31-1, and
64 decoded-message levels. Zero fields choose defaults and negative fields
fail. Apply the same recursion limit to parsing and payload traversal when
raising it. Every dimension is checked even in empty tensors. Backend
ValidateShape checks products/byte sizes/host indices without numeric allocation.
Payload traversal also bounds cycles in manually constructed protobuf objects.

`Usage() (loadBytes, initializerBytes int64)` reports logical reservations. Load
bytes include original encoded bytes plus all decoded fields (numeric widths,
string/byte lengths and retained unknown bytes), owned tensor copies, and raw
conversion staging. Counting metadata and retaining successful staging charges
is deliberately conservative. Typed conversion needs one new numeric copy;
raw conversion needs two. Sum checks precede staging/backend allocations and
failed or cancelled conversions leave usage unchanged. Cancellation is checked
through payload traversal, every 1,024 raw elements, and around the synchronous
backend copy; the backend copy itself is not interruptible. Go allocation
headers, excess slice capacities, GC retention and total process heap are outside
these logical limits. ErrInvalidTensor and ErrUnsupported classify semantic
failures; ErrLimit covers wire and payload limits. Causes remain discoverable
with errors.Is. Ordinary tests use checked-in literal bytes and Go values and
need no Python.

## Named graph validation

`internal/graph.Build(ctx, model *onnxpb.ModelProto, encodedBytes int64, limits Limits)`
returns an immutable `*graph.Graph`. Use bounded wire parsing first and pass the
complete decoded model plus its actual original encoded length. The builder reads
wire without mutation and creates one shared TensorDecoder. It converts every
initializer and dense tensor attribute while the original pointers are registered,
then releases decoder/wire references. No proto.Clone or second numerical copy is
needed at load. ReadModelWithSize returns the actual encoded length for public
reader loading; proto.Size cannot reconstruct it.

Build validates IR 8, exactly one empty-domain opset-13 import, a named top-level
graph, unique producers/declarations/attributes, and all named references in
declared topological order. Checking preceding producers rejects missing/forward
references and cycles without reordering nodes. Repeated node labels are valid.
Input/initializer overlap creates a default with a checked type/rank/fixed-axis
contract; the input's leading symbolic/unknown batch is retained for overrides.
Output/intermediate unknown ranks and dimensions remain distinct from scalars,
fixed zero and shared symbols. Missing intermediate metadata is allowed.
Statically contradictory declarations fail without full operator shape inference.

Every node, including unused output paths, remains in Nodes. Empty input/output
names keep their positions without creating values. Graph outputs may directly
reference inputs or initializers; empty graphs and graphs without declared outputs
are structurally accepted. Functions, function overloads/references, training,
subgraphs, sparse storage and non-dense/unsupported types fail with context.
Generic attribute kinds/defaults are checked, but allowed operator names, attribute
names, arity and optional-slot positions must be resolved by RT-07 before execution.
Structural acceptance alone does not mark a capability implemented.

Graph Metadata/Inputs/Outputs/Nodes/Values/Value accessors return copied records.
Values retain producer indices, defaults and every metadata declaration. Harmless
model/graph/node/value/tensor docs, properties, denotations, descriptive model
domains and quantization annotations are preserved. Unknown fields remain in the
source decoded wire object and are ignored semantically; the graph retains no
protobuf objects. Constants privately own backend storage, expose copied typed
values/shape/metadata, and provide Clone for backend consumers. The
planner/executor must reserve these clones before allocation and avoid retaining
both original and compiled numerical copies unnecessarily. Internal packages do
not import the public facade.

Later compilation is still part of the load budget: before calling deep snapshot
accessors such as Nodes/Values, its consumer must reserve any copied attribute
list/dimension payloads as well as numerical Constant.Clone bytes. It may use
Graph.Usage conservatively when estimating an entire snapshot. Avoid retaining
duplicate snapshots after producing a plan.

Graph Limits contains Tensor (decode.TensorLimits), MaxNodes (100,000), MaxValues
(200,000 distinct producers), MaxAttributes (400,000 aggregate), and MaxNameBytes
(4,096 for each name/symbol/domain/doc/metadata/attribute string). Zero fields
choose defaults, negatives fail. Tensor rank/dimension limits also apply to all
metadata. String traversal follows descriptor order for deterministic diagnostics
and skips numeric repeated payloads. Attribute scalar/list and dimension snapshots
are reserved with TensorDecoder.ReservePayload before their allocation; owned
tensor copies and raw staging remain charged by Decode. Usage includes the
complete encoded/decoded payload and these reservations. Go object/header/slice
capacity overhead and copying accessors after Build are outside the load budget.
Cancellation is checked through traversal, graph loops, conversions and before
publication; synchronous backend/protobuf operations remain bounded units.

## Versioned operator planning

The root facade embeds the canonical `capabilities.json`. `Capabilities()` decodes
independent operator records, including nested optional slots, attribute maps and
JSON defaults. It reports planned entries explicitly. The private `compileGraph`
bridge constructs `operators.NewRegistry([]capability.Record)` from those same
records, then calls `operators.Compile(ctx, graph, registry, limits)`. There is no
second operator support table; the compiler remains private to the public loader.

Registry snapshots its records and resolves by exact domain/operator and imported
opset. It selects the newest schema no newer than the import, then checks that
schema's explicit tested interval. It never falls back to older semantics when
the selected schema lacks coverage. Production intervals remain exactly 13.
Resolver tests use isolated records for actual Softmax-11/13 defaults and
Squeeze-11/13 attribute/input differences, and select Conv-11 at import 13; those
tests do not broaden the accepted model envelope or implement those operators.

Compile visits every node, including unused paths, and rejects planned/unknown
operators, invalid arity, required empty slots, extra outputs, unknown attributes
and incorrect attribute kinds. Identity-13 preserves float32/int64 type and shape
at any rank, including scalar and empty tensors. Constant-13 requires exactly one
`value` TENSOR attribute, already converted and owned by graph.Build. Float/int
scalar/list attributes remain unsupported until RT-09. Neither handler allocates
or clones numerical output storage. Defaulted input metadata keeps its symbolic
or unknown batch dimension rather than the initializer's current size.

Plan stores input/output contracts, source/producer-ordered Values and resolved
Steps with value-table indices. It retains no protobuf or source Graph. Constants
are shared immutable references. Metadata/Inputs/Outputs/Values/Value/Steps return
independent snapshots; Usage reports aggregate logical load/initializer bytes.
Value.Info merges declarations and inference, distinguishing absent rank, scalar,
fixed zero, symbolic and unknown dimensions. Value.Deferred means the merged
shape is not concrete. All original Value.Declarations are retained: even a
concrete inferred summary can contain a declared constraint that still needs
runtime verification. RT-08 checks every produced value against both inferred
metadata and declarations, and binds symbols afresh each run.

Limits contains MaxLoadBytes (1 GiB), MaxTensorBytes (64 MiB), MaxRank (32) and
MaxDimension (2^31-1); zero fields choose defaults and negatives fail. Pass the
same effective load/tensor/rank/dimension limits used to build the Graph. The
compiler starts with Graph.Usage and reserves a conservative whole-graph payload
bound for each of five deep accessors before requesting any snapshot. Further
dimension/shape snapshots are reserved before allocation, using Constant.Rank
without a copying shape query. Reservations remain charged conservatively even
when transient snapshots become unreachable. Go object/header/capacity overhead
is excluded, consistent with graph/decode accounting. Static shapes are checked
with backend ValidateShape and finite limits without numerical allocation;
partially known shapes retain checks for Run. Initializer usage never increases
during compilation. RT-09 must charge any later scalar/list Constant conversion
against both load and initializer quotas before constructing storage.

Compilation errors use public Error (an internal shared type alias), retaining
ErrInvalidArgument/ErrInvalidModel/ErrUnsupported/ErrLimit classifications and
the original cause. Node diagnostics include index/name/domain/operator/import,
and concrete input shapes when budget permits. Nil diagnostic shape means
unknown or unavailable; a nonnil empty shape means scalar. Error.Is matches Kind
and Unwrap retains Cause for errors.Is/As, including cancellation. Diagnostics
iterate declared order and sort map-derived required names. The formatter
reports load/run stage through RT-08 while retaining compile as the internal
default.
Cancellation is checked before snapshots, through value/node/shape reservations
and before publication. Cancelling compilation leaves the source and registry
usable. Ordinary Go tests run offline; the pinned Python profile audit also
checks actual schema attributes/defaults and version-selection differences.

## Profile fixture verification

RT-01's existing profile-check fixture can be verified using its pinned Python
3.9.6 environment with ONNX 1.16.2, NumPy 1.26.4, and protobuf 4.25.3:

```sh
python tools/check_profile.py
python tools/check_profile.py --write
```

The first command checks the fixture and selected schemas; the second explicitly
regenerates the original Identity fixture and provenance. Neither command is a
Go test prerequisite. See [the runtime contract](runtime-contract.md) for source
revisions and reference limitations.

## Serial model sessions

`Load(ctx, io.Reader, Options)`, `LoadBytes(ctx, []byte, Options)` and
`LoadFile(ctx, string, Options)` return a validated, compiled `*Model`.
Options/limits are copied and normalized before I/O. Load never closes its reader;
LoadFile closes only its own file. ReadModelWithSize reuses the bounded decoder
and reports consumed original bytes, including noncanonical varints/unknown fields.
Wire, graph and planner causes map to the public Error classifications with load
stage; errors.Is/As retain cancellation and I/O causes.

A private executable plan snapshots Metadata/Inputs/Outputs/Values/Steps only
after reserving `Plan.SnapshotBytes(ctx)`. This no-allocation estimate counts
copied record/index/dimension payloads and a second input/output metadata copy;
constants and immutable string bytes remain shared. It does not multiply prior
load reservations or clone numerical storage. The public Model.Metadata returns
independent records and remains available after Model.Close. Existing sessions
retain their plan; new sessions fail with ErrClosed. A plan mutex bounds session
count, including sessions outliving the model.

Session.Run takes immutable tensors by reference, validates names, required/default
inputs, dtype/rank/fixed axes and shared symbols, and creates a new private table,
binding map and budget for each call. Every value checks both inferred Info and
all original Declarations; Identity refinement must not erase a symbolic constraint.
Every node executes even when its output is unrequested. Requested outputs are
unique declared names returned in request order; empty requests use declaration
order. Zero-node/input/initializer passthroughs and graphs without outputs work.

The first evaluator retains all table entries and charges aliases conservatively.
It counts resident model tensor bytes once, every bound/default source, each node
result and all requested output copies. It reserves before every constant/output
clone. Backend v0.7.0 tensor.go Clone directly copies into owned destination storage
without numeric staging, so Identity/Constant/copies use zero numeric scratch.
The workspace accounting path separately enforces workspace and combined live
reservations, including checked sums; RT-09 must provide audited kernel bounds.
Runtime metadata/Go headers/capacity/GC retention and earlier caller-retained
outputs are outside these logical tensor budgets.

A session's TryLock makes overlapping Run/Close return ErrBusy immediately. Failure
or cancellation drops local state and releases the guard. Successful Close is
idempotent, releases the session slot and makes later Run return ErrClosed. Separate
sessions run concurrently with immutable shared plans/constants. Context checks
cover runtime loops, before/after each node/copy and final publication. Backend
copies remain synchronous with no in-kernel cancellation or partial output result.
ExternalDataResolver defines the reserved Resolve signature, but nonnil options
and external tensors fail as unsupported until RT-14; no resolver is called.

The three original [session fixtures](../testdata/sessions/README.md) have checked-in
bytes, pinned-checker provenance and hand-calculated expectations. Ordinary Go tests
cover branches, multiple IO, shared exact int64 constants, bit-preserving float
copies, default overrides, scalar/empty values, fresh symbol bindings, original
constraints, lifecycle races, concurrent sessions, cancellation at every observed
boundary, exact budgets and recovery. They need neither Python nor native runtime.
