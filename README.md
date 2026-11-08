# onnx

A Go-native ONNX inference runtime built on
[`github.com/itsmontoya/neuralnetwork`](https://github.com/itsmontoya/neuralnetwork).
Load an `.onnx` model, bind named immutable Go tensors, and evaluate named outputs
on the CPU without cgo or an installed native inference runtime.

RT-01–RT-14 implement the [runtime contract](docs/runtime-contract.md), bounded
protobuf and dense tensor loading, named graph validation, versioned planning,
and serial sessions. The executable subset is deliberately narrow: ONNX IR 8,
standard-domain opset 13, float32 Add/Sub/Mul/Div, MatMul/Gemm,
Relu/Sigmoid/Tanh/Softmax, and float32/int64 Identity/Constant. M2 adds Shape, Flatten, Transpose, Reshape,
Squeeze/Unsqueeze, Gather and Concat, preserving exact int64 shape values. M3 adds
float32 NCHW 2D Conv, MaxPool, AveragePool, GlobalAveragePool and stored-statistics
BatchNormalization (rank 2/4). Branching, multiple
inputs/outputs, default overrides,
passthroughs, scalar/empty tensors and symbolic leading batches are supported.
Constant accepts dense tensors and float/int scalar/list forms. Relu propagates
NaN; empty-axis Softmax is an explicit extension of the pinned evaluator.

`Load`, `LoadBytes`, and `LoadFile` return an immutable shared Model. Create a
Session with `NewSession`, then call `Run` with named inputs and optional requested
outputs. Outputs own independent storage and survive later runs and Close.
Overlapping calls on one session return `ErrBusy`; separate sessions can run
concurrently. `Options.Limits` applies finite load, tensor, live, workspace and
session bounds. External float32/int64 tensors require an explicit
`Options.ExternalData` resolver for every loader, including `LoadFile`.
`NewDirectoryResolver` provides confined regular-file ranges on supported Go
platforms; the [portable resolver example](examples/external/main.go) explicitly
permits an in-memory weight file. Checksum metadata is unsupported. `Capabilities()` returns copied manifest records, covering all 25 implemented operators
and their exact restrictions.

The module pins `neuralnetwork v0.7.0`, protobuf v1.36.12, and Go 1.27.2 with a
Go 1.26.1 baseline. Ordinary builds/tests require neither protoc nor Python.
No general ONNX compatibility is claimed beyond the documented subset.

## Run an example

From the repository root with Go 1.27.2 installed:

```sh
go mod download
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 3
CGO_ENABLED=0 go run ./examples/external -batch 3
CGO_ENABLED=0 go test ./...
```

The [example](examples/inference/main.go) loads an original checked-in model,
reads metadata, creates a session, supplies named tensors, reads named results
and closes both owners. All four graphs accept symbolic leading batches and have
independent expected outputs for batch sizes 1 and 3. The branching graph has two
named inputs and two outputs. See [getting started](docs/getting-started.md)
for clean-clone commands, the complete implemented restriction table and compatible export
requirements, and [the RT-11 integration report](docs/integration-rt11.md) for
results, provenance and hashes. Initial dependency download needs network;
cached normal builds/tests need neither network nor reference tooling.

RT-10's [offline conformance harness](testdata/conformance/README.md) verifies
87 reference agreements, one documented empty-Softmax extension, four executed reference
differences and four explicit rejections, using hashed model/typed TensorProto fixtures. The pinned upstream
suite contains no exact IR 8/opset 13 model; selected original upstream bytes
remain unsupported-version tests rather than numerical coverage claims.

## Implementation plans

- [NEEDS.md](NEEDS.md) records the verified `neuralnetwork` v0.7.0 baseline,
  completed NN-01–NN-12 tasks, published handoffs, and integration constraints.
  RT-12 supplies the [runtime inputs for NN-13](docs/handoff-nn13.md); optimization
  remains backend work.
- [ROADMAP.md](ROADMAP.md) defines 19 tasks for this runtime, including ONNX
  decoding, graph validation, versioned operators, execution, conformance,
  examples, and release preparation.
- [TODO.md](TODO.md) provides the starting order and points to the authoritative
  task checklists.

Each task includes its dependencies, implementation scope, acceptance criteria,
and handoff requirements so a separate agent can complete it without this chat.
Use the published v0.7.0 contract and source revision recorded in `NEEDS.md`.
The copying backend is sufficient to begin runtime development; NN-13 does not
block the first working runtime or release readiness.

## Scope and next milestones

The current CPU profile accepts ONNX IR 8, standard-domain opset 13, and dense
float32/int64 values. Named graph inputs/outputs and branching are implemented.
Broadcasting, matrix products, activations and the RT-10 fixture/conformance
harness, user examples and runtime shape transformations are implemented.
The [M2 fixtures](testdata/m2/README.md) retain exact reference differences. The [87 M3 cases](testdata/m3/README.md) execute with independently verified CNN
outputs and retained evaluator/ORT differences. RT-14 adds [explicit external weights](docs/integration-rt14.md); further
resource hardening and a documented release remain later work.

The [roadmap's coverage tables](ROADMAP.md#milestones-and-coverage) specify
operator schema versions, supported subsets, and milestone requirements. Newer
IR/opsets, other data types, recurrent models, transformers, quantization, and
control flow need additional tasks and reference fixtures.

## Architecture

This module owns ONNX protobuf loading, compatibility checks, named graph
planning, sessions, diagnostics, and model execution. `neuralnetwork` provides
concrete tensors and forward numerical kernels through additive APIs that preserve
its existing matrix, training, and serialization behavior.

The runtime works with `CGO_ENABLED=0`, with a portable `purego` path.
Go dependencies such as protobuf bindings are allowed. Python reference tooling
may generate development fixtures, while ordinary builds and tests use
checked-in artifacts without Python or network access.

## Contributing

Read [AGENTS.md](AGENTS.md) and [STYLEGUIDE.md](STYLEGUIDE.md), then the next
unchecked task and its dependency handoffs in `ROADMAP.md`, using the released
backend contract linked from `NEEDS.md`. Implement runtime tasks here and
coordinate later library changes in `neuralnetwork`. Record exact
APIs, dependency revisions, ownership rules, validation results, and limitations
in each completed task's handoff.

Use Go 1.27.2 and the [development commands](docs/development.md) to build and
verify this foundation, including cgo-disabled, `purego`, and race checks.
The [inference guide](docs/getting-started.md) covers the working MLP, shape and CNN examples.
Pull request descriptions must preserve
[the repository template](.github/PULL_REQUEST_TEMPLATE.md).
