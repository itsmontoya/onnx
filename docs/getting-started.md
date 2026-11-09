# Run inference in Go

The M1–M4 runtime runs MLPs, named branching graphs, runtime shape subgraphs, 2D CNNs and float32 LayerNormalization on the CPU
entirely in Go. It accepts **ONNX IR 8, empty standard domain, imported opset exactly 13 or 17**
and the implemented operator subsets below. Python, protoc, a C/C++ inference
runtime and a shared library are unnecessary for normal use.

## Try the checked-in examples

Install Go 1.27.2. For the reviewable RT-17 checkout, run:

```sh
git clone --branch rt-17-version-profiles https://github.com/itsmontoya/onnx.git
cd onnx
go mod download
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph exported -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph exported -batch 3
CGO_ENABLED=0 go run ./examples/external -batch 1
CGO_ENABLED=0 go run ./examples/external -batch 3
CGO_ENABLED=0 go test ./...
```

The initial clone and Go dependency download need network access. With the
module cache populated, these commands use checked-in models and values; they
need no network or reference tools. Run from the repository root because the
example resolves `testdata/inference/models/` and `testdata/m3/models/` relative
to the working directory.
For the portable backend path, add `-tags=purego` to `go run` or `go test`.

The MLP is `Gemm → Relu → Gemm → Softmax`. Its sole named input is `features`;
the named output is `probabilities`. The branch adds a named `residual` input,
joins it with the logits, and fans out into `probabilities` and `delta` outputs.
All named inputs/outputs declare float32 `[batch, 2]`. The same loaded model
accepts batch 1 and batch 3; the shared `batch` symbol must agree across inputs.
The example selects these two reproducible input sets; it does not impose a
runtime-wide restriction to those batch sizes.

The branch at batch 3 prints:

```text
branch: IR 8, opset 13, 2 input(s), 2 output(s)
input features: rank 2, leading symbol "batch"
input residual: rank 2, leading symbol "batch"
probabilities [3 2] = [0.984858 0.015142 0.689974 0.310026 0.015712 0.984288]
delta [3 2] = [0.250000 -0.500000 1.000000 0.000000 -0.250000 0.750000]
```

These are rounded display values. Tests compare the original typed goldens with
`atol=rtol=1e-5`; `delta` includes ordinary float32 subtraction rounding.
See [the integration report](integration-rt11.md) for hashes and actual results.

The CNN is `Conv → BatchNormalization → Relu → MaxPool → Flatten → Gemm`.
Input `features` is float32 `[batch,1,4,4]`; outputs `logits` and `pooled`
are `[batch,3]` and `[batch,2,2,2]`. Its independently calculated results match
ORT for both batches; the pinned ONNX evaluator differs. See the
[CNN integration report](integration-rt13.md) and [edge-case record](../testdata/m3/README.md).

## Use the public API

```go
func infer(ctx context.Context) (outputs []onnx.NamedTensor, err error) {
	var (
		model   *onnx.Model
		session *onnx.Session
		input   *onnx.Tensor
	)

	if model, err = onnx.LoadFile(ctx, "testdata/inference/models/mlp.onnx", onnx.Options{}); err != nil {
		return nil, err
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	if session, err = model.NewSession(); err != nil {
		return nil, err
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	if input, err = onnx.NewFloat32([]int64{1, 2}, []float32{1, 2}); err != nil {
		return nil, err
	}

	return session.Run(ctx, map[string]*onnx.Tensor{"features": input},
		onnx.RunOptions{Outputs: []string{"probabilities"}})
}
```

This function imports `context`, `errors` and `github.com/itsmontoya/onnx`.
Read `outputs[0].Tensor.Float32Values()` for a copied `[]float32`. Read
`model.Metadata()` to inspect names, dtype, declared rank/dimensions and
`ValueInfo.Defaulted` before constructing inputs. Do not copy Model or Session
structs after use; use their pointers. The caller must not mutate the Run input
map or output-name slice during a call.


The compiling [example](../examples/inference/main.go) demonstrates the full
sequence: `LoadFile(ctx, path, Options{})`, `model.Metadata()`,
`model.NewSession()`, `NewFloat32(shape, values)`, `session.Run`, reading each
`NamedTensor.Name` and `Tensor.Float32Values()`, then closing both owners.
Its deferred cleanup joins close errors into the returned error. `Load` accepts
an `io.Reader`; `LoadBytes` accepts a borrowed byte slice for the loading call.
Load never closes the caller's reader; LoadFile opens and closes its own file.
LoadBytes borrows bytes during loading only; do not mutate them until it returns.
All loaders publish an owned immutable plan and retain no resolver.

An initializer with a graph-input name makes that input optional. Omit the map
entry to use its default, or supply an override satisfying the input's declared
dtype/rank/fixed dimensions and symbols. The default's concrete batch size does
not restrict an otherwise dynamic leading axis. Other inputs are required;
initializers absent from graph inputs cannot be overridden. Missing/unexpected
names, nil tensors and duplicate/unknown requested outputs fail explicitly.
Each Run binds symbols afresh. Every graph node is validated at Load and executed
at Run, including branches outside the requested outputs.

Supply inputs as `map[string]*onnx.Tensor`. Empty `RunOptions{}` requests all
declared outputs in declaration order. For the branch, request a specific order
with `RunOptions{Outputs: []string{"delta", "probabilities"}}`; names must be
unique declared graph outputs. Intermediate names are not requestable. Output
tensors own independent storage, and their typed accessors copy. Retained
outputs survive later runs, `Session.Close` and `Model.Close`. Tensors require
no `Close` method.

One session permits one active run; overlapping `Run` or `Close` returns
`ErrBusy`. Use separate sessions for concurrent work and close each to release
its finite slot. `Model.Close` prevents new sessions while existing sessions
remain valid. A failed or cancelled run returns no partial outputs and leaves
the session reusable. Cancellation is checked around synchronous kernels and
between nodes; it does not interrupt a running kernel. `Options{}` selects
finite default limits, including 16 sessions; zero limit fields select defaults
and negative values fail. See [the contract](runtime-contract.md#limits-and-cancellation)
for all fourteen defaults and accounting; they limit logical payloads rather
than process RSS, Go object/allocator/GC overhead or previously retained outputs.
For example, `Options{Limits: Limits{MaxTensorBytes: 8 << 20, MaxSessions: 4}}`
lowers just those defaults (qualify these names with `onnx.` in another package).
Constructors/accessors operate outside load/run budgets; caller retention still
uses memory. Independent sessions have separate run budgets.

Context cancellation/deadlines reach `errors.Is`. Checks occur between bounded
reads, through runtime-owned planning/conversion loops, around each synchronous
node/copy, and before publication. A kernel, protobuf unmarshal or backend deep
copy finishes between polls. A custom resolver must cooperate to interrupt its
own blocked Resolve/Read. There is no single-kernel cancellation-latency promise.

## Prepare a compatible model

Choose an exporter that can emit IR 8 and opset 13 or 17 with dense float32 numerical tensors,
inline or explicitly resolved external weights, a known input rank, fixed nonleading input dimensions, and only
the implemented operators/attributes below. Declare a symbolic or unknown
leading axis for variable batch sizes. Repeated nonempty symbols bind to one
concrete size per run. Dense int64 shape/index tensors and type-preserving M2 transforms are supported.
Squeeze/Unsqueeze-13 use axes inputs, and Reshape-13 uses zero-copy-dimension
and single inferred -1 rules; it rejects allowzero even when zero. At import 17,
Reshape-14 accepts allowzero=0 (the same dimension-copy default) or 1 (literal
zeros); a literal zero and -1 together are invalid. Shape-15 accepts start/end
slices with negative indexing and rank clamping. BatchNormalization-15 requires
training_mode=0 and exactly one output; schema-9 at import 13 rejects that
attribute entirely. Float32 CNN/arithmetic restrictions remain unchanged.

Check the resulting `ir_version`, every `opset_import`, every node and all value
types. An opset request alone does not guarantee IR 8 or eliminate custom
domains, shape nodes or external weights. Exporter defaults are not universally
compatible. Do not relabel an arbitrary newer model as an accepted profile: conversion
must preserve its operator and representation semantics, pass the ONNX checker,
and be compared with a reference before use. A checker-valid model can still
exceed this runtime's supported subset or resource limits.

For an exact reproducible authoring example, [the fixture generator](../tools/check_inference.py)
uses ONNX 1.16.2 `helper.make_model(..., opset_imports=[helper.make_opsetid("", 13)])`,
sets `model.ir_version = 8` on its original graphs, declares `["batch", 2]`
I/O, stores inline float32 initializers and runs `checker.check_model(...,
full_check=True)` plus the reference evaluator. These development tools are
optional; ordinary users consume the checked-in model directly.

The [actual PyTorch export](../testdata/profiles/exports/pytorch_mlp.onnx) was
emitted by `torch.onnx.export` 2.2.2 at requested opset 17, with IR 8 and dynamic
leading batches already present. Its bytes are stored unchanged. Run
`CGO_ENABLED=0 go run ./examples/inference -graph exported -batch 3` without
installing PyTorch. [Profile provenance](../testdata/profiles/README.md) records
its source, absent original-fixture license, exact schema selections and
PyTorch/evaluator/ORT comparisons. This one tested module does not establish
compatibility with arbitrary PyTorch exports.

`errors.Is(err, onnx.ErrUnsupported)` identifies unsupported IR/imports,
operators or supported-schema features outside the subset. Missing or duplicate
imports and structurally invalid graphs use `ErrInvalidModel`. Load rejects
known static problems; concrete input dimensions and dynamic node compatibility
are validated at Run. `ErrLimit` identifies a configured budget failure;
`errors.Is` also reaches cancellation/deadline and underlying I/O errors.
`errors.As(err, &diagnostic)` with `var diagnostic *onnx.Error` exposes stage,
node index/name, domain, operator, imported opset, value and relevant shapes
when available. Envelope errors have no invented node context.

## Implemented capability table

This table is generated and checked against [capabilities.json](../capabilities.json)
by `go run ./tools/capabilitiesgen`, using the same embedded records as
`Capabilities()` and the compiler registry. Run `go run ./tools/capabilitiesgen -write`
to update it deliberately; the no-write command fails on drift and runs in CI.
The optional `tools/check_inference.py` reference check verifies it too. Every row
uses the empty domain and its exact tested-import interval. Separate 13 and 17
rows do not claim support for imports 14–16;
optional input slots are zero-based. Only listed attributes are accepted, with
the recorded types/defaults; required inputs/outputs cannot have empty names.
Each operator has exactly one nonempty output. Gemm C may be omitted or have an
empty slot. A supplied C must still be compatible when beta is zero.

<!-- Implemented capabilities start -->

| Operator (schema) | Tested imports | Types | Inputs → outputs | Attributes | Restrictions |
| --- | --- | --- | --- | --- | --- |
| Identity (13) | 13–13 | float32, int64 | 1 → 1 | None | static planning and serial execution; session lifecycle implemented by RT-08; dense float32/int64; arbitrary rank including scalar/empty; output shape/type preserved; requested outputs own independent copies, including Identity aliases |
| Constant (13) | 13–13 | float32, int64 | 0 → 1 | `value:TENSOR`, `value_float:FLOAT`, `value_floats:FLOATS`, `value_int:INT`, `value_ints:INTS` | exactly one dense float32/int64 tensor, float/int scalar or float/int vector attribute; exact int64 storage; arbitrary tensor rank including scalar/empty; reject sparse/string/other forms; scalar/list materialization shares aggregate load and initializer budgets |
| Add (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Sub (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Mul (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Div (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting; IEEE division by zero |
| MatMul (13) | 13–13 | float32 | 2 → 1 | None | float32 rank >= 1; vector promotion and batch broadcasting; empty contractions |
| Gemm (13) | 13–13 | float32 | 2–3; optional slots [2] → 1 | `alpha:FLOAT=1.0`, `beta:FLOAT=1.0`, `transA:INT=0`, `transB:INT=0` | float32 rank-two A/B; optional scalar/vector/row/column/full C with unidirectional broadcast; transA/transB restricted to 0/1; alpha/beta default 1, including nonfinite coefficients |
| Relu (13) | 13–13 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty; propagate NaN; negative zero becomes positive zero |
| Sigmoid (13) | 13–13 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Tanh (13) | 13–13 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Softmax (13) | 13–13 | float32 | 1 → 1 | `axis:INT=-1` | float32 rank >= 1; axis default -1; empty tensor accepted after axis validation as explicit reference extension; nonfinite slices preserve NaN behavior |
| Flatten (13) | 13–13 | float32, int64 | 1 → 1 | `axis:INT=1` | type preserving; split axis in [-rank,rank], default 1; explicit concrete products for empty prefixes; documented evaluator rejection difference |
| Reshape (13) | 13–13 | float32, int64 | 2 → 1 | None | int64 rank-one shape input; schema-13 zero/-1 policy; reject allowzero and ambiguous zero-product inference |
| Transpose (13) | 13–13 | float32, int64 | 1 → 1 | `perm:INTS` | type preserving; perm defaults to reverse; complete unique nonnegative permutation |
| Squeeze (13) | 13–13 | float32, int64 | 1–2; optional slots [1] → 1 | None | optional int64 rank-one axes input; omitted axes remove all unit dimensions; explicit axes must be unique and name unit dimensions; explicit empty axes are identity |
| Unsqueeze (13) | 13–13 | float32, int64 | 2 → 1 | None | required int64 rank-one axes input; unique axes normalized in output rank |
| Shape (13) | 13–13 | float32, int64 | 1 → 1 | None | int64 vector output from concrete runtime shape; no start/end attributes |
| Gather (13) | 13–13 | float32, int64 | 2 → 1 | `axis:INT=0` | float32/int64 rank >= 1 data; arbitrary-rank int64 indices; negative indices supported; exact prefix/indices/suffix shape including empties |
| Concat (13) | 13–13 | float32, int64 | 1–2147483647 → 1 | `axis:INT` | float32/int64 inputs of one type with equal rank >= 1 and equal non-axis dimensions; checked axis sum; required axis; nonempty variadic input list |
| Conv (11) | 13–13 | float32 | 2–3; optional slots [2] → 1 | `auto_pad:STRING=NOTSET`, `dilations:INTS`, `group:INT=1`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; ONNX [M,C/group,kH,kW] weights; positive channels/filters/kernels/strides/dilations; group divides input/output channels; optional [M] bias; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; kernel_shape must match weights when present; dynamic weights/bias; nonpositive floor spatial extents clamp to zero |
| MaxPool (12) | 13–13 | float32 | 1 → 1–2; optional slots [1] | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `dilations:INTS`, `kernel_shape:INTS`, `pads:INTS`, `storage_order:INT=0`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides/dilations; nonnegative pads; values output only; second empty slot allowed, nonempty indices unsupported; storage_order must be 0; ceil_mode must be 0/1; documented empty/nonfinite differences; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| AveragePool (11) | 13–13 | float32 | 1 → 1 | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `count_include_pad:INT=0`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides; nonnegative pads; no dilation attribute; ceil_mode/count_include_pad must be 0/1; full-kernel divisor when including pad; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| GlobalAveragePool (1) | 13–13 | float32 | 1 → 1 | None | float32 NCHW rank four; output [N,C,1,1]; empty spatial reduction NaN |
| BatchNormalization (9) | 13–13 | float32 | 5 → 1 | `epsilon:FLOAT=9.999999747378752e-06`, `momentum:FLOAT=0.8999999761581421` | float32 rank-two or NCHW rank-four input and [C] parameters; exactly one inference output slot; reject all additional output slots and training_mode; finite nonnegative epsilon; finite momentum in [0,1] accepted but no statistics update; independent stored-statistics reference; documented evaluator/ORT differences |
| Identity (16) | 17–17 | float32, int64 | 1 → 1 | None | static planning and serial execution; session lifecycle implemented by RT-08; dense float32/int64; arbitrary rank including scalar/empty; output shape/type preserved; requested outputs own independent copies, including Identity aliases |
| Constant (13) | 17–17 | float32, int64 | 0 → 1 | `value:TENSOR`, `value_float:FLOAT`, `value_floats:FLOATS`, `value_int:INT`, `value_ints:INTS` | exactly one dense float32/int64 tensor, float/int scalar or float/int vector attribute; exact int64 storage; arbitrary tensor rank including scalar/empty; reject sparse/string/other forms; scalar/list materialization shares aggregate load and initializer budgets |
| Add (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Sub (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Mul (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Div (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting; IEEE division by zero |
| MatMul (13) | 17–17 | float32 | 2 → 1 | None | float32 rank >= 1; vector promotion and batch broadcasting; empty contractions |
| Gemm (13) | 17–17 | float32 | 2–3; optional slots [2] → 1 | `alpha:FLOAT=1.0`, `beta:FLOAT=1.0`, `transA:INT=0`, `transB:INT=0` | float32 rank-two A/B; optional scalar/vector/row/column/full C with unidirectional broadcast; transA/transB restricted to 0/1; alpha/beta default 1, including nonfinite coefficients |
| Relu (14) | 17–17 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty; propagate NaN; negative zero becomes positive zero |
| Sigmoid (13) | 17–17 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Tanh (13) | 17–17 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Softmax (13) | 17–17 | float32 | 1 → 1 | `axis:INT=-1` | float32 rank >= 1; axis default -1; empty tensor accepted after axis validation as explicit reference extension; nonfinite slices preserve NaN behavior |
| Flatten (13) | 17–17 | float32, int64 | 1 → 1 | `axis:INT=1` | type preserving; split axis in [-rank,rank], default 1; explicit concrete products for empty prefixes; documented evaluator rejection difference |
| Reshape (14) | 17–17 | float32, int64 | 2 → 1 | `allowzero:INT=0` | int64 rank-one shape input; at most one inferred -1; equal element counts; allowzero=0 copies input dimensions; allowzero=1 uses literal zeros and forbids any -1 with zero; reject ambiguous zero-product inference |
| Transpose (13) | 17–17 | float32, int64 | 1 → 1 | `perm:INTS` | type preserving; perm defaults to reverse; complete unique nonnegative permutation |
| Squeeze (13) | 17–17 | float32, int64 | 1–2; optional slots [1] → 1 | None | optional int64 rank-one axes input; omitted axes remove all unit dimensions; explicit axes must be unique and name unit dimensions; explicit empty axes are identity |
| Unsqueeze (13) | 17–17 | float32, int64 | 2 → 1 | None | required int64 rank-one axes input; unique axes normalized in output rank |
| Shape (15) | 17–17 | float32, int64 | 1 → 1 | `end:INT`, `start:INT=0` | int64 vector output from concrete runtime shape; schema-15 start/end clamp to input rank after negative indexing; start >= end yields an empty vector |
| Gather (13) | 17–17 | float32, int64 | 2 → 1 | `axis:INT=0` | float32/int64 rank >= 1 data; arbitrary-rank int64 indices; negative indices supported; exact prefix/indices/suffix shape including empties |
| Concat (13) | 17–17 | float32, int64 | 1–2147483647 → 1 | `axis:INT` | float32/int64 inputs of one type with equal rank >= 1 and equal non-axis dimensions; checked axis sum; required axis; nonempty variadic input list |
| Conv (11) | 17–17 | float32 | 2–3; optional slots [2] → 1 | `auto_pad:STRING=NOTSET`, `dilations:INTS`, `group:INT=1`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; ONNX [M,C/group,kH,kW] weights; positive channels/filters/kernels/strides/dilations; group divides input/output channels; optional [M] bias; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; kernel_shape must match weights when present; dynamic weights/bias; nonpositive floor spatial extents clamp to zero |
| MaxPool (12) | 17–17 | float32 | 1 → 1–2; optional slots [1] | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `dilations:INTS`, `kernel_shape:INTS`, `pads:INTS`, `storage_order:INT=0`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides/dilations; nonnegative pads; values output only; second empty slot allowed, nonempty indices unsupported; storage_order must be 0; ceil_mode must be 0/1; documented empty/nonfinite differences; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| AveragePool (11) | 17–17 | float32 | 1 → 1 | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `count_include_pad:INT=0`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides; nonnegative pads; no dilation attribute; ceil_mode/count_include_pad must be 0/1; full-kernel divisor when including pad; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| GlobalAveragePool (1) | 17–17 | float32 | 1 → 1 | None | float32 NCHW rank four; output [N,C,1,1]; empty spatial reduction NaN |
| BatchNormalization (15) | 17–17 | float32 | 5 → 1 | `epsilon:FLOAT=9.999999747378752e-06`, `momentum:FLOAT=0.8999999761581421`, `training_mode:INT=0` | float32 rank-two or NCHW rank-four input and [C] parameters; exactly one inference output slot; reject all additional output slots and nonzero training_mode; finite nonnegative epsilon; finite momentum in [0,1] accepted but no statistics update; independent stored-statistics reference; documented evaluator/ORT differences; schema-15 inference only: training_mode=0, all five inputs float32, exactly one output slot |
| LayerNormalization (17) | 17–17 | float32 | 2–3; optional slots [2] → 1–3; optional slots [1 2] | `axis:INT=-1`, `epsilon:FLOAT=9.999999747378752e-06`, `stash_type:INT=1` | IR 8/import 17 only; schema 17; dense float32 inputs and all outputs; stash_type=1 only; rank >=1; axis defaults to -1 and selects the trailing normalized dimensions; nonempty normalized dimensions required; epsilon defaults to float32 1e-5; finite nonnegative values only; Scale and optional B must be unidirectionally broadcastable to X; Y preserves X shape; optional Mean/InvStdDev slots preserve position; reduced dimensions retained as 1; NN-12 ReduceMean/Sqrt/arithmetic composition; synchronous cancellation between kernels; conservative workspace reservations |

<!-- Implemented capabilities end -->

All graphs must be named, acyclic and in topological order, with unique value
producers and resolvable named edges. Input ranks must be known; only a leading
input dimension can be symbolic/unknown. Fixed zero dimensions and scalars stay
distinct from unknown metadata. Intermediate/output shapes may be inferred or
checked at Run. An initializer overlapping a graph input supplies a default
that callers may override within the input's metadata; other initializers
cannot be overridden. Inline constants and all outputs obey the finite limits.

Other IR/opsets/domains, unsupported dtypes, sparse tensors, local functions,
training information, tensor segments, subgraphs/control flow and
sequence/map/optional values are rejected. External float32/int64 data requires
an explicit resolver; unknown metadata keys and checksum metadata are unsupported. Reductions and optimization are future work. `Capabilities()` includes
planned entries, whose presence does not establish execution support. The
current examples work without these future milestones.

## Explicit external weights

The [external MLP fixture](../testdata/external/README.md) has the same weights
and outputs as the inline MLP, stored in one shared file. Every loader, including
LoadFile, requires an explicit resolver for external tensors:

```go
func loadExternal(ctx context.Context) (model *onnx.Model, err error) {
	var resolver *onnx.DirectoryResolver
	if resolver, err = onnx.NewDirectoryResolver("testdata/external/models"); err != nil {
		return nil, err
	}

	defer func() { err = errors.Join(err, resolver.Close()) }()
	return onnx.LoadFile(ctx, "testdata/external/models/mlp.onnx",
		onnx.Options{ExternalData: resolver})
}
```

The caller closes the returned Model. Close the resolver when loading finishes; the model has independent owned
weights. The directory helper rejects traversal/absolute paths and symlink
escapes at open, and supplies separate bounded readers per range. It is
unsupported on js/plan9; [the memory resolver example](../examples/external/memory_resolver.go)
shows a portable explicit allowlist that also works without a directory helper.
Each returned reader is closed by the loader, including readers returned with
resolution errors. Resolve receives exact byte length even when length metadata
is omitted; a short range or close error prevents model publication.
Neither loader nor helper guesses a base directory or fetches URLs. The resolver
is caller-owned and must support concurrent loads if shared. The helper permits
internal relative symlinks but rejects escapes during open. It does not isolate
hard links, mounts or filesystem boundaries, and callers must provide stable file
content; concurrent truncation fails exact reading. Its successful Close leaves
already returned readers usable.

```sh
CGO_ENABLED=0 go run ./examples/external -batch 1
CGO_ENABLED=0 go run ./examples/external -batch 3
CGO_ENABLED=0 go run -tags=purego ./examples/external -batch 3
```

Location is required; offset and length are nonnegative decimal int64 values.
An omitted length still resolves exactly the tensor's byte count, and a supplied
length must match it. Checksum and unknown keys are unsupported. See the
[runtime contract](runtime-contract.md#external-tensor-data) for
budgets, reader ownership, cancellation and filesystem limits.

## Reproduce supported authoring and failures

The tested model origins are the original deterministic graphs generated with
ONNX helper/numpy_helper and the pinned checker/evaluator, plus the separately
identified unmodified upstream rejection fixtures. No PyTorch, TensorFlow or
third-party exporter version has been validated here. Their opset flag alone
cannot establish compatibility.

For optional development authoring, use **Python 3.9.6, ONNX 1.16.2, NumPy
1.26.4 and protobuf 4.25.3** in an isolated environment. The [fixture guide](../testdata/inference/README.md) records hash provenance.
To provision the optional reference tools using a Python 3.9.6 interpreter:

```sh
python3 -m venv /tmp/onnx-reference
/tmp/onnx-reference/bin/python -m pip install onnx==1.16.2 numpy==1.26.4 protobuf==4.25.3 onnxruntime==1.19.2
. /tmp/onnx-reference/bin/activate
python --version
```

The generators require exactly Python 3.9.6 and the listed versions; an interpreter
with another version will be rejected. With that environment active, run:

```sh
python tools/check_profile.py
python tools/check_inference.py
python tools/check_m2.py
python tools/check_m3.py
python tools/check_external.py
```

These no-write commands reproduce the actual authored IR-8/opset-13 bytes,
checker outcomes, typed expected outputs and retained reference differences.
M3 additionally requires pinned **onnxruntime 1.19.2**, CPUExecutionProvider.
Use each generator's `--write` only to replace artifacts intentionally. The
MLP/branch generator constructs original graphs with
`helper.make_opsetid("", 13)` and explicit IR 8, adds float32 initializers and
symbolic batch I/O, then runs the full checker and independent NumPy comparisons.
The CNN generator uses stored-statistics normalization and records the ONNX
and ORT outcomes independently. These tested authoring instructions do not
convert arbitrary newer graphs. For another exporter, inspect its actual output
and compare every supported graph with a trusted independent reference first.

The following probe uses only Go and checked-in models. Run from the repository
root; its expected failures are checked by sentinel identity, not brittle error
strings. It exits unsuccessfully if any documented failure stops being enforced.

```sh
probe_dir=$(mktemp -d)
cat > "$probe_dir/main.go" <<'GO'
package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/itsmontoya/onnx"
)

func main() {
	if err := probe(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func probe() (err error) {
	var (
		ctx   = context.Background()
		model *onnx.Model
	)

	_, err = onnx.LoadFile(ctx, "testdata/conformance/upstream/test_tanh/model.onnx", onnx.Options{})
	if !errors.Is(err, onnx.ErrUnsupported) {
		return fmt.Errorf("original IR-7 fixture: want ErrUnsupported, got %v", err)
	}

	_, err = onnx.LoadFile(ctx, "testdata/external/models/mlp.onnx", onnx.Options{})
	if !errors.Is(err, onnx.ErrUnsupported) {
		return fmt.Errorf("external weights without resolver: want ErrUnsupported, got %v", err)
	}

	_, err = onnx.LoadFile(ctx, "testdata/inference/models/mlp.onnx",
		onnx.Options{Limits: onnx.Limits{MaxModelBytes: 1}})
	if !errors.Is(err, onnx.ErrLimit) {
		return fmt.Errorf("wire byte budget: want ErrLimit, got %v", err)
	}

	if model, err = onnx.LoadFile(ctx, "testdata/inference/models/mlp.onnx", onnx.Options{}); err != nil {
		return err
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	var session *onnx.Session
	if session, err = model.NewSession(); err != nil {
		return err
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	_, err = session.Run(ctx, nil, onnx.RunOptions{})
	if !errors.Is(err, onnx.ErrInvalidArgument) {
		return fmt.Errorf("missing features input: want ErrInvalidArgument, got %v", err)
	}

	fmt.Println("PASS: unsupported IR, absent resolver, byte limit, missing input")
	return nil
}
GO
CGO_ENABLED=0 go run "$probe_dir/main.go"
rm -r "$probe_dir"
```

Use `errors.Is` for `ErrInvalidArgument` (invalid API input), `ErrInvalidModel`
(contradictory graph/wire declarations), `ErrUnsupported` (outside this profile),
`ErrLimit`, `ErrBusy` and `ErrClosed`. Cancellation/deadlines and underlying I/O
causes are also preserved. For node diagnostics:

```go
var diagnostic *onnx.Error
if errors.As(err, &diagnostic) {
	fmt.Printf("stage=%s node=%d operator=%s value=%s\n",
		diagnostic.Stage, diagnostic.NodeIndex, diagnostic.Operator, diagnostic.Value)
}
```

The diagnostic also retains node name, domain, imported opset and relevant
concrete shapes when available; envelope failures do not invent node context.
Load rejects known static issues; input-dependent axes/ranks/shapes fail at Run.
See [existing validation and recovery tests](../session_test.go) and
[external failure tests](../external_integration_test.go) for checksum/unknown-key,
dynamic shape and resolver/read/close failures. The
[release checklist](release-checklist.md) records tested platforms, actual
reference disagreements, licensing status and remaining gaps.


## Audited normalization model

RT-19 runs an unchanged Apache-2.0 normalization export with Y, Mean and
InvStdDev named outputs:

```sh
CGO_ENABLED=0 go run ./examples/normalization
CGO_ENABLED=0 go run -tags=purego ./examples/normalization
```

The [corpus report](model-compatibility.md) records all 33 preserved models,
source hashes/licenses, metadata and unsupported nodes. Nineteen normalization
exports are enabled; fourteen remain required negative tests. This bounded
float32 schema 17 set does not establish general transformer compatibility.
