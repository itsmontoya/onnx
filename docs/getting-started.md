# Run inference in Go

The first working milestone runs MLPs, named branching graphs and runtime shape subgraphs on the CPU
entirely in Go. It accepts **ONNX IR 8, empty standard domain, imported opset 13**
and the implemented operator subsets below. Python, protoc, a C/C++ inference
runtime and a shared library are unnecessary for normal use.

## Try the checked-in examples

Install Go 1.27.2. For the reviewable RT-12 checkout, run:

```sh
git clone --branch rt-12-shape-operators https://github.com/itsmontoya/onnx.git
cd onnx
go mod download
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 3
CGO_ENABLED=0 go test ./...
```

The initial clone and Go dependency download need network access. With the
module cache populated, these commands use checked-in models and values; they
need no network or reference tools. Run from the repository root because the
example resolves `testdata/inference/models/` relative to the working directory.
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

## Use the public API

The compiling [example](../examples/inference/main.go) demonstrates the full
sequence: `LoadFile(ctx, path, Options{})`, `model.Metadata()`,
`model.NewSession()`, `NewFloat32(shape, values)`, `session.Run`, reading each
`NamedTensor.Name` and `Tensor.Float32Values()`, then closing both owners.
Its deferred cleanup joins close errors into the returned error. `Load` accepts
an `io.Reader`; `LoadBytes` accepts a borrowed byte slice for the loading call.
All loaders publish an owned immutable plan.

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
for exact budgets and accounting; they are logical payload limits, not a hard
process heap quota.

## Prepare a compatible model

Choose an exporter that can emit opset 13 with dense float32 numerical tensors,
inline weights, a known input rank, fixed nonleading input dimensions, and only
the implemented operators/attributes below. Declare a symbolic or unknown
leading axis for variable batch sizes. Repeated nonempty symbols bind to one
concrete size per run. Dense int64 shape/index tensors and type-preserving M2 transforms are supported.
Squeeze/Unsqueeze-13 use axes inputs, and Reshape-13 uses zero-copy-dimension
and single inferred -1 rules; it rejects later allowzero metadata.

Check the resulting `ir_version`, every `opset_import`, every node and all value
types. An opset-13 request alone does not guarantee IR 8 or eliminate custom
domains, shape nodes or external weights. Exporter defaults are not universally
compatible. Do not relabel an arbitrary newer model as IR 8/opset 13: conversion
must preserve its operator and representation semantics, pass the ONNX checker,
and be compared with a reference before use. A checker-valid model can still
exceed this runtime's supported subset or resource limits.

For an exact reproducible authoring example, [the fixture generator](../tools/check_inference.py)
uses ONNX 1.16.2 `helper.make_model(..., opset_imports=[helper.make_opsetid("", 13)])`,
sets `model.ir_version = 8` on its original graphs, declares `["batch", 2]`
I/O, stores inline float32 initializers and runs `checker.check_model(...,
full_check=True)` plus the reference evaluator. These development tools are
optional; ordinary users consume the checked-in model directly.

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
by `tools/check_inference.py`. Every row uses the empty domain/imported opset 13;
optional input slots are zero-based. Only listed attributes are accepted, with
the recorded types/defaults; required inputs/outputs cannot have empty names.
Each operator has exactly one nonempty output. Gemm C may be omitted or have an
empty slot. A supplied C must still be compatible when beta is zero.

<!-- Implemented capabilities start -->

| Operator (schema) | Types | Inputs → outputs | Attributes | Restrictions |
| --- | --- | --- | --- | --- |
| Identity (13) | float32, int64 | 1 → 1 | None | static planning and serial execution; session lifecycle implemented by RT-08; dense float32/int64; arbitrary rank including scalar/empty; output shape/type preserved; requested outputs own independent copies, including Identity aliases |
| Constant (13) | float32, int64 | 0 → 1 | `value:TENSOR`, `value_float:FLOAT`, `value_floats:FLOATS`, `value_int:INT`, `value_ints:INTS` | exactly one dense float32/int64 tensor, float/int scalar or float/int vector attribute; exact int64 storage; arbitrary tensor rank including scalar/empty; reject sparse/string/other forms; scalar/list materialization shares aggregate load and initializer budgets |
| Add (13) | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Sub (13) | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Mul (13) | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Div (13) | float32 | 2 → 1 | None | float32 multidirectional broadcasting; IEEE division by zero |
| MatMul (13) | float32 | 2 → 1 | None | float32 rank >= 1; vector promotion and batch broadcasting; empty contractions |
| Gemm (13) | float32 | 2–3; optional slots [2] → 1 | `alpha:FLOAT=1.0`, `beta:FLOAT=1.0`, `transA:INT=0`, `transB:INT=0` | float32 rank-two A/B; optional scalar/vector/row/column/full C with unidirectional broadcast; transA/transB restricted to 0/1; alpha/beta default 1, including nonfinite coefficients |
| Relu (13) | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty; propagate NaN; negative zero becomes positive zero |
| Sigmoid (13) | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Tanh (13) | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Softmax (13) | float32 | 1 → 1 | `axis:INT=-1` | float32 rank >= 1; axis default -1; empty tensor accepted after axis validation as explicit reference extension; nonfinite slices preserve NaN behavior |
| Flatten (13) | float32, int64 | 1 → 1 | `axis:INT=1` | type preserving; split axis in [-rank,rank], default 1; explicit concrete products for empty prefixes; documented evaluator rejection difference |
| Reshape (13) | float32, int64 | 2 → 1 | None | int64 rank-one shape input; schema-13 zero/-1 policy; reject allowzero and ambiguous zero-product inference |
| Transpose (13) | float32, int64 | 1 → 1 | `perm:INTS` | type preserving; perm defaults to reverse; complete unique nonnegative permutation |
| Squeeze (13) | float32, int64 | 1–2; optional slots [1] → 1 | None | optional int64 rank-one axes input; omitted axes remove all unit dimensions; explicit axes must be unique and name unit dimensions; explicit empty axes are identity |
| Unsqueeze (13) | float32, int64 | 2 → 1 | None | required int64 rank-one axes input; unique axes normalized in output rank |
| Shape (13) | float32, int64 | 1 → 1 | None | int64 vector output from concrete runtime shape; no start/end attributes |
| Gather (13) | float32, int64 | 2 → 1 | `axis:INT=0` | float32/int64 rank >= 1 data; arbitrary-rank int64 indices; negative indices supported; exact prefix/indices/suffix shape including empties |
| Concat (13) | float32, int64 | 1–2147483647 → 1 | `axis:INT` | float32/int64 inputs of one type with equal rank >= 1 and equal non-axis dimensions; checked axis sum; required axis; nonempty variadic input list |

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
sequence/map/optional values are rejected. External data and resolver options
are unsupported until RT-14. CNN operators (RT-13), reductions and optimization are future work. `Capabilities()` includes
planned entries, whose presence does not establish execution support. The
current examples work without these future milestones.
