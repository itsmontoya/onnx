# Runtime contract and initial compatibility profile

Status: RT-01 specifies the contract. No Go runtime API or operator is implemented
by this task. [capabilities.json](../capabilities.json) records planned coverage;
only records marked `implemented` may later be advertised as support. The
[roadmap](../ROADMAP.md) identifies the implementation tasks and their acceptance
criteria. This project is an independent Go runtime, not Microsoft's ONNX Runtime.

## Pinned sources and reference tools

The source of ONNX wire and operator definitions is **ONNX v1.16.2**, commit
[`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`](https://github.com/onnx/onnx/tree/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557),
licensed under [Apache-2.0](https://github.com/onnx/onnx/blob/v1.16.2/LICENSE).
RT-03 vendors [the unmodified schema and license](../third_party/onnx/README.md)
and checked-in reproducible Go bindings. This schema has no imports and the
upstream release has no root NOTICE. Schema generation is a development
operation, not part of loading a model.
The [IR](https://github.com/onnx/onnx/blob/v1.16.2/docs/IR.md) and
[version rules](https://github.com/onnx/onnx/blob/v1.16.2/docs/Versioning.md)
are separate from operator schema selection. This source also defines newer IR
and opsets, which this profile does not thereby accept.

The numerical backend is **github.com/itsmontoya/neuralnetwork v0.7.0**, commit
[`6048452c57a0a293db4591cd2e2febd7d6b743a3`](https://github.com/itsmontoya/neuralnetwork/tree/6048452c57a0a293db4591cd2e2febd7d6b743a3)
(MIT). Its published [tensor contract](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/tensor-design.md)
and [handoffs](../NEEDS.md#v070-status) are normative for backend ownership;
ONNX policies remain here. Do not use unpublished changes, a committed local
`replace`, or another module's `internal` packages.

Development references are Python **3.9.6**, ONNX **1.16.2**, NumPy **1.26.4**,
and Python protobuf **4.25.3**. ONNX Runtime **1.19.2**, CPU provider, is a
secondary development oracle for documented evaluator disagreements. Ordinary
Go builds/tests must use checked-in fixtures, without Python, network access,
cgo, C/C++ libraries, Metal, or an external inference service. RT-02 pins the
Go 1.27.2 toolchain and pure-Go `google.golang.org/protobuf v1.36.12` dependency
with a Go 1.26.1 module baseline; see [development](development.md).

`tools/check_profile.py` builds a deterministic original Identity graph, checks
it with the pinned ONNX checker, audits all 25 selected schema versions, and
compares its bytes/hash to `testdata/profile/identity.onnx` and provenance.
It deliberately sets IR 8 and imports opset 13 before checking; it does not
rewrite an existing export's version. Run it using the pinned development
Python environment; `--write` regenerates these RT-01 artifacts. It establishes
that the profile contains a valid tiny ONNX model, not runtime conformance.

## Accepted model envelope

| Boundary | Initial contract |
| --- | --- |
| Execution | CPU inference, deterministic serial topological scheduling |
| Versions | IR exactly 8; one standard empty-string domain import, opset exactly 13 |
| Graph | One top-level acyclic graph, checked in declared order, named values, multiple inputs/outputs and fan-out/joins |
| Values | Dense contiguous row-major float32 and exact int64 tensors; scalar shape `[]`; any concrete zero dimension means an empty tensor |
| Operators | Only implemented records in the manifest, with their listed dtype/rank/attribute restrictions |
| Weights | Inline initializers and Constant attributes first; explicit external resolver only after RT-14 |
| Metadata | Preserve harmless producer/doc/metadata strings; ignore unknown protobuf fields semantically while retaining them in the decoded wire object |

Reject absent/duplicate/incompatible imports, other domains (including the literal
`ai.onnx` spelling), other IR/opsets, sparse initializers/attributes, local
functions, subgraph attributes/control flow, training information, sequence/map/
optional/sparse values, unsupported tensor types, segments, and unsupported
nonempty optional outputs. Unknown attributes are errors. Empty optional input/
output name slots retain their position and are allowed only by the selected
schema and implemented subset. An empty name slot is not an ONNX Optional value.
An operator outside requested output paths is still validated and executed;
there is no silent unsupported-node pruning.

Each value has one producer (input or initializer counts as a source), except
an initializer may also name a graph input as specified below. Node labels may
repeat; node indices and output names identify execution. Duplicate value
producers, graph input/output names, initializer names, and attribute names are
errors. All nonempty inputs must refer to a preceding source. Graph outputs may
refer directly to an input or initializer, so zero-node graphs are useful.
Require nonempty graph/input/output/value names and typed graph input/output
metadata. Intermediate `value_info` is optional; missing shape information never
requires implementing full ONNX shape inference. Reject contradictions that
are statically provable; defer checks that depend on run values.

Public input shapes must have a declared rank. Each dimension is fixed and
nonnegative, except dimension zero (the leading axis) may be a nonempty symbolic
name or unknown. Scalars have no batch axis. Dynamic nonleading input dimensions
and unknown input ranks are unsupported initially. Output/intermediate metadata
may have unknown rank or dimensions at any position; concrete output tensors
always carry their resolved shape. Empty shape and absent shape metadata are
distinct. Symbols with the same nonempty name bind to the same concrete size
within a run across all supplied/defaulted inputs; unnamed unknown dimensions
are independent. Every run binds afresh, including zero batch sizes. Initializer
metadata and dimensions are immutable after compilation.

## Public surface to implement

Module `github.com/itsmontoya/onnx`, package `onnx`, owns the facade and lifecycle.
`internal/onnxpb` contains only generated wire types; `internal/decode` bounded
wire/tensor loading; `internal/graph` named graph validation and immutable plans;
`internal/operators` schema resolution, planning, and adapters. Introduce packages
only when they contain real behavior. Numerical payloads and math stay in
`neuralnetwork/tensor`; do not translate the graph into `model.Sequential`.
Internal packages accept backend tensors and internal metadata records without
importing the public facade, avoiding import cycles. Public metadata can be
converted at the facade boundary; it is not a second tensor storage engine.

These are intended signatures, not declarations of already available APIs:

```go
// Tensor is an immutable facade containing one private backend *tensor.Tensor.
// No method exposes that pointer or a writable destination API.
type Tensor struct { /* private backend value */
}
type DataType = tensor.DataType

const Float32 = tensor.Float32
const Int64 = tensor.Int64

func NewTensor(dataType DataType, shape []int64) (out *Tensor, err error)
func NewFloat32(shape []int64, values []float32) (out *Tensor, err error)
func NewInt64(shape []int64, values []int64) (out *Tensor, err error)
func (t *Tensor) Validate() (err error)
func (t *Tensor) DataType() (dataType DataType)
func (t *Tensor) Shape() (shape []int64)
func (t *Tensor) Rank() (rank int)
func (t *Tensor) Len() (elements int64)
func (t *Tensor) ByteSize() (bytes int64)
func (t *Tensor) Float32Values() (values []float32, err error)
func (t *Tensor) Int64Values() (values []int64, err error)
func (t *Tensor) Clone() (out *Tensor, err error)

type DimensionKind uint8

const (
	UnknownDimension DimensionKind = iota
	FixedDimension
	SymbolicDimension
)

type Dimension struct {
	Kind   DimensionKind
	Value  int64
	Symbol string
}
type ValueInfo struct {
	Name       string
	DataType   DataType
	ShapeKnown bool
	Dimensions []Dimension
	Defaulted  bool
}
type Metadata struct {
	Name      string
	Producer  string
	IRVersion int64
	Opset     int64
	Inputs    []ValueInfo
	Outputs   []ValueInfo
}

type Options struct {
	Limits       Limits
	ExternalData ExternalDataResolver
}

func Load(ctx context.Context, source io.Reader, options Options) (out *Model, err error)
func LoadBytes(ctx context.Context, data []byte, options Options) (out *Model, err error)
func LoadFile(ctx context.Context, path string, options Options) (out *Model, err error)

type Model struct { /* immutable shared plan, guarded lifecycle */
}

func (m *Model) Metadata() (metadata Metadata)
func (m *Model) NewSession() (out *Session, err error)
func (m *Model) Close() (err error)

type RunOptions struct{ Outputs []string }
type NamedTensor struct {
	Name   string
	Tensor *Tensor
}
type Session struct { /* private per-run state, nonblocking busy guard */
}

func (s *Session) Run(ctx context.Context, inputs map[string]*Tensor,
	options RunOptions) (outputs []NamedTensor, err error)
func (s *Session) Close() (err error)
func Capabilities() (capabilities []Capability)
```

`Capability` is the Go representation of a manifest operator record: `Domain`,
`Operator`, `SchemaVersion`, `MinOpset`, `MaxOpset`, `Status`, `Task`, `DataTypes`,
`Inputs`, `Outputs`, `Attributes`, and `Restrictions`. The query includes planned
records with explicit status and returns a deep copy; it does not imply that
planned operators can load. RT-07 may settle the mechanical nested record types
without changing this semantic contract. Manifest format/version is separate
from an ONNX imported opset. Every record lists the exact schema selected from
release 1.16.2 at import 13; schema versions 11/12/1/9 are intentional for CNNs.

Constructors delegate to backend checked constructors and copy shape/payload.
Typed accessors/Clone also copy. A nil or zero Tensor is invalid, never a scalar.
A caller cannot mutate a valid Tensor through the facade; values can therefore
be shared between runs/sessions without another input snapshot. Callers must
not concurrently replace map entries or write Go struct values during Run.
All concrete int64 data stays int64, including values beyond 2^24. Scalar and
empty constructor examples are `NewFloat32(nil, []float32{3.5})`,
`NewTensor(Float32, []int64{0, 3})`, and
`NewInt64([]int64{1}, []int64{16777217})`.

Readers remain caller-owned and are never closed by Load. LoadBytes borrows its
bytes only until returning; callers must not mutate them during the call, and
no compiled value borrows their storage afterward. LoadFile owns/closes only the
file it opens and never guesses an external-data directory. Nil context/reader,
invalid options, and invalid zero receivers return argument/state errors; reject
these before I/O. Check context before loading, between bounded read chunks,
after decoding, and during graph/plan construction. Protobuf unmarshal itself
is one synchronous bounded unit. Validate all nodes before publishing a Model.
Model.Close is idempotent, prevents new sessions, and releases its reference to
the plan; existing sessions hold their own reference and remain usable until
closed. Metadata returns deep copies, remains inspectable after Close, and
returns zero metadata for nil/zero models. Invalid NewSession/Close receivers
return ErrInvalidArgument; an already valid closed object returns nil on Close.

Run validates all names and dtype/rank/fixed dimensions, then shared symbol
bindings, before evaluating nodes. An initializer overlapping a graph input is
its default; callers may override it with a value satisfying that input's type
and shape metadata (not necessarily the default's batch size). Other graph
inputs are required. Initializers absent from graph inputs cannot be overridden.
Reject missing required names, unexpected names, nil/invalid tensors, and duplicate
requested outputs. Nil/empty Outputs requests all declared graph outputs in
declaration order; otherwise return the unique declared names in request order.
Intermediate names are not requestable. A passthrough/default/Identity result
still returns an independent owned copy; outputs never expose shared constants.
No partial outputs are returned on failure.

One session permits only one active Run. Concurrent Run or Close on that session
returns ErrBusy immediately; it neither waits nor cancels the active call. A
successful Close is idempotent; later Run returns ErrClosed. A failed/cancelled
Run releases temporary values and guards, leaving the session usable. Separate
sessions can run concurrently with shared immutable plans/constants and disjoint
scratch. Requested outputs are copied before publication, remain valid after
another run or any Close, and are the caller's responsibility to retain/drop.
Model.Close and NewSession are linearized against each other; a session whose
creation wins the race remains valid. Session counting is shared with the plan
and released on successful Session.Close, including after Model.Close.

For example, after RT-08, LoadFile returns a model, Metadata describes `x`,
NewSession creates execution state, and `Run(ctx, map[string]*Tensor{"x": x},
RunOptions{Outputs: []string{"y"}})` yields `outputs[0].Name == "y"`. Close the
session and model explicitly. RT-11 supplies a checked compiling example;
this contract does not claim that sequence runs yet.

## Limits and cancellation

```go
type Limits struct {
	MaxModelBytes       int64
	MaxLoadBytes        int64
	MaxInitializerBytes int64
	MaxTensorBytes      int64
	MaxLiveBytes        int64
	MaxWorkspaceBytes   int64
	MaxNodes            int
	MaxValues           int
	MaxAttributes       int
	MaxNameBytes        int
	MaxRank             int
	MaxDimension        int64
	MaxRecursion        int
	MaxSessions         int
}
```

Zero fields choose defaults; negative fields fail. Positive values may lower or
raise limits but must fit host arithmetic, and no value disables a limit.
Normalize a copied Options value before use. MaxAttributes counts aggregate
attributes over the graph, MaxValues counts distinct named source/node values,
and MaxNameBytes bounds each name, symbol, domain, producer/doc/metadata string,
and attribute string payload. Ranks/dimensions apply to all metadata, constants,
and runtime values, including otherwise-unused empty dimensions. Initializers
and Constant payloads together count against MaxInitializerBytes.

| Field | Default | Accounted resource |
| --- | ---: | --- |
| MaxModelBytes | 64 MiB | Serialized bytes; reader consumes at most limit+1 to detect oversize |
| MaxLoadBytes | 1 GiB | Encoded input bytes + decoded tensor/attribute payload bytes + owned tensor copies + transient conversion/external buffers |
| MaxInitializerBytes | 256 MiB | Aggregate owned initializer/Constant payload |
| MaxTensorBytes | 64 MiB | Each input, initializer, intermediate, requested output copy |
| MaxLiveBytes | 512 MiB | Resident model tensors + bound inputs + live intermediates + output copies + current workspace reservation |
| MaxWorkspaceBytes | 256 MiB | Per-node adapter scratch/conversion/copy reservation |
| MaxNodes | 100,000 | Nodes in the entire graph |
| MaxValues | 200,000 | Distinct named values |
| MaxAttributes | 400,000 | Aggregate attributes |
| MaxNameBytes | 4,096 | Each string described above |
| MaxRank | 32 | Declared and concrete rank |
| MaxDimension | 2^31 - 1 | Each fixed/concrete dimension |
| MaxRecursion | 64 | Known-message nesting in protobuf decoding |
| MaxSessions | 16 | Live sessions sharing a compiled plan |

These are logical tensor/payload accounting limits, not a promise to cap process
RSS or account Go object headers, allocator/GC retention, kernel stack, or caller-
retained outputs from earlier runs. The finite serialized-size/recursion bounds
also bound the decoder's object input, but Go protobuf allocates its object tree
before semantic graph-count checks. Never describe this as zero-copy decoding
or a hard heap quota. Charge decoded repeated payload lengths at their actual
width and decoded byte fields separately from encoded bytes. Tensor conversion
copies and raw-data staging must fit before allocation; sum using checked
arithmetic. Release wire objects once the immutable plan is built. Many tiny
wire objects can still have substantial allocator overhead; untrusted deployments
may additionally enforce process/container limits.

RT-03 confirms that protobuf v1.36.12's configurable recursion limit applies to
known messages. Unknown groups use the official protowire decoder's independent
fixed allowance of 10,000 recursive descents: it accepts 10,001 nested group
levels and rejects 10,002. Unknown fields/groups remain preserved and subject
to the serialized byte limit; MaxRecursion is not a uniform unknown-group depth
limit. Executable tests retain this distinction without introducing a second
wire parser.

The first executor may conservatively retain all intermediates until Run ends.
Charge shared model tensors once and each bound input reference conservatively;
never hide memory by exempting defaults or aliases. Reserve outputs and node
workspace before calling any allocating backend helper; release reservations
on every exit. An immutable facade avoids snapshot copies, but owned final
output cloning temporarily coexists with its source. Caller accessors allocate
outside Run and its budget. Independently running sessions have separate budgets;
MaxSessions bounds their count, so a configured deployment can bound aggregate
logical reservations by resident model storage plus at most MaxSessions times
the per-run allowance. No global cross-model memory governor is promised.

Backend v0.7.0 has copying bridges and no public workspace. The implementation
must inspect its released allocation paths and use checked conservative charges,
not only output bytes. Minimum reservation rules (numeric bytes, excluding shape
metadata) are:

| Path | Scratch charge in addition to live inputs/destination |
| --- | --- |
| Relu/Sigmoid/Tanh | At least 3 times input bytes (input matrix, output matrix, copied values); any runtime nonfinite patch buffers add their copies |
| MatMul | Full temporary output plus, for each batch block, copied left/right matrices and two product-block buffers; conservatively sum per-block temporaries over all batches when GC retention matters |
| Gemm | Copied A/B, product matrix and copied product values, plus transposed B when both transpose flags are set |
| Shape/Constant/typed/raw conversion | Actual accessor/staging/copy buffers in addition to the owned destination |
| Shape transforms, arithmetic, CNN, Softmax | Charge any actual numerical scratch; zero scratch may be claimed only after auditing the selected kernel |

Budget validation precedes allocation and arithmetic must not overflow. Planning
performs only metadata allocations, never full numeric output tensors. Scratch
bounds must be updated if the pinned backend changes. Conservative rejection
within configured budgets is preferable to an unaccounted large allocation.

Kernels are synchronous and accept no context/check callback in v0.7.0.
Cancellation is observed before/after each node and during runtime-owned bounded
I/O/planning loops, never promised inside a running numerical kernel or protobuf
unmarshal. If cancellation arrives during the final node/output copying, check
again before publishing and return the context error without partial outputs.
RT-15 audits this granularity; a future in-kernel callback needs a published
backend contract/revision. There is no cancellation-latency guarantee for a
single large kernel, and no goroutine is abandoned to simulate cancellation.

## Errors and deterministic diagnostics

Proposed sentinels are ErrInvalidArgument, ErrInvalidModel, ErrUnsupported,
ErrLimit, ErrBusy, and ErrClosed. State errors with no useful additional context
are returned directly. Semantic/operational errors use a structured Error:

```go
type Error struct {
	Kind      error
	NodeIndex int // -1 outside a node
	NodeName  string
	Domain    string
	Operator  string
	Opset     int64
	Value     string
	Shapes    [][]int64
	Cause     error
}

func (e *Error) Error() (message string)
func (e *Error) Unwrap() (cause error)
func (e *Error) Is(target error) (matches bool)
```

Is matches Kind; Unwrap retains Cause so errors.Is reaches context.Canceled,
context.DeadlineExceeded and I/O causes, and errors.As reaches Error. Errors
include load/run stage, value name or node index/name, domain/operator/imported
opset and relevant input shapes where available. Unknown dynamic shapes are
represented by nil shape entries in diagnostics, never by fake concrete zero.
Reject invalid models/unsupported static combinations at Load; defer only
value-dependent violations to Run. Sort unexpected input names and other map-
derived diagnostics to avoid nondeterministic error selection. A kernel failure
has node context and cannot poison other sessions or immutable constants.

## Numerical and empty-value decisions

Float kernels use the backend's float32 accumulation/formulas; general finite
comparison starts at `atol=rtol=1e-5` and fixtures may tighten tolerances. Integers,
shapes, indexing, and storage transforms compare exactly; transforms preserve
float bits. For numerical results compare NaN classification (not NaN payload),
infinity signs, and signed zero where a fixture explicitly requires it. A finite
fixture tolerance is not a universal numerical error bound.

| Case | Chosen runtime behavior and reference record |
| --- | --- |
| Relu NaN | Propagate NaN as ONNX 1.16.2's evaluator does. RT-09 must wrap/patch the backend's NaN-to-positive-zero behavior with accounted temporary storage. Negative zero becomes positive zero as in that evaluator. |
| Softmax-13 | Default axis -1; normalize only that axis, stable maximum subtraction. Scalars/invalid axes fail. Empty tensors succeed after axis validation, including an empty reduction axis; record the evaluator's zero-length reduction rejection as a deliberate extension, never a passing reference comparison. Nonfinite nonempty slices retain backend/evaluator NaNs. |
| Div | IEEE float32 division, including signed infinity/NaN for zero denominators; do not use the older matrix division error contract. |
| MatMul/Gemm | ONNX vector/batch promotion and unidirectional optional Gemm C; empty contractions produce zeros before alpha/beta. Validate supplied bias even when beta is zero. |
| Reshape-13 | 0 copies the same-position input dimension; at most one -1 is inferred. Reject ambiguous zero-product inference; literal target zeros are not the later allowzero mode. |
| Flatten/Gather empty shapes | Use schema shape formulas/concrete products, including prefix + indices shape + suffix for Gather. Preserve pinned evaluator rejections/incorrect `[0]` empty-Gather shortcut as differences; validate with exact independent cases. |
| Pooling | Backend's full-kernel average divisor when including pad, required ceil trimming, NaN propagation, -infinity for max with no samples, average no-sample NaN/zero depending on divisor, and global empty average NaN. Record evaluator/ORT disagreement explicitly; no claim of universal agreement on those edge cases. |
| BatchNormalization-9 | Stored-statistics inference `scale*(x-mean)/sqrt(variance+epsilon)+bias`; rank 2 or 4, one output. Accept valid momentum metadata (finite in [0,1], default 0.9) but do not update statistics. Do not accept later training_mode. Compare independently because the pinned evaluator erroneously applies momentum; retain ORT's zero-variance/zero-epsilon difference. |

The manifest gives exact arity, allowed optional name positions, attribute names,
and rank/type restrictions. Attributes of an operator are not accepted merely
because another schema version defines them. In particular Reshape-13 has no
allowzero, Shape-13 has no start/end, and Squeeze/Unsqueeze-13 use int64 axes
inputs. Runtime-dependent rank/axes/shape checks remain at Run. MaxPool indices
and nonzero storage_order are explicitly unsupported. String/sparse Constants
are unsupported even though schema-13 defines those attribute forms.

## Fixture and handoff obligations

Every implemented operator needs isolated positive/negative fixtures and a
composed graph where useful. Provenance records source/license, model/value
SHA-256, IR/import/schema versions, tool versions, named I/O and comparison
policy. Hand-calculated fixtures prevent decoder/comparator round trips from
being the only oracle. Store real reference failures separately; skips never
count as implemented support. BN-9, nonfinite and empty cases require the
independent/reference distinctions above and in [NEEDS.md](../NEEDS.md).

RT-02 establishes module/CI without pretending operators work. RT-03 regenerates
wire bindings from the pinned source. RT-04 implements the immutable facade and
metadata distinction. RT-05 accounts decoded and owned payloads. RT-06 validates
the entire graph. RT-07 compiles versioned handlers and consults manifest status.
RT-08 adds the lifecycle and limits, RT-09 M1 math, RT-10 differential fixtures,
and RT-11 the first runnable user example. Follow the rest of the roadmap in
dependency order; an optimization or further profile must never weaken these
ownership/error/budget rules without an explicit reviewed contract change.

The external-data API reserved for RT-14 is:

```go
type ExternalDataResolver interface {
	Resolve(ctx context.Context, location string, offset, length int64) (source io.ReadCloser, err error)
}

func NewDirectoryResolver(directory string) (out *DirectoryResolver, err error)

type DirectoryResolver struct { /* confined directory handle */
}

func (r *DirectoryResolver) Resolve(ctx context.Context, location string,
	offset, length int64) (source io.ReadCloser, err error)
func (r *DirectoryResolver) Close() (err error)
```

The requested length is always the validated exact tensor byte count, including
when the model omits its external length. A supplied length must match that
count. Offset is a checked nonnegative int64. The reader starts at the requested
offset; the runtime reads bounded chunks, requires exactly the requested bytes,
and closes resolver-owned resources on every path. The directory helper must
use a race-safe confined filesystem primitive, rejecting absolute/traversal paths
and symlink escapes; platforms without that guarantee require a caller resolver.
No current-directory guesses or automatic URLs. Before RT-14, external tensors
fail as unsupported even if Options contains a resolver. RT-14 must validate
recognized keys/duplicates and location context. Initially reject checksum
metadata as unsupported rather than silently claiming verification; ONNX's
optional SHA-1 checksum applies to the external file, not just a selected tensor
range. A later verified-checksum policy needs its own explicit implementation.
