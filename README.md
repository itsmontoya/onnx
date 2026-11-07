# onnx

A Go-native ONNX inference runtime built on
[`github.com/itsmontoya/neuralnetwork`](https://github.com/itsmontoya/neuralnetwork).
Load an `.onnx` model, bind named immutable Go tensors, and evaluate named outputs
on the CPU without cgo or an installed native inference runtime.

RT-01–RT-09 implement the [runtime contract](docs/runtime-contract.md), bounded
protobuf and inline tensor loading, named graph validation, versioned planning,
and serial sessions. The executable subset is deliberately narrow: ONNX IR 8,
standard-domain opset 13, float32 Add/Sub/Mul/Div, MatMul/Gemm,
Relu/Sigmoid/Tanh/Softmax, and float32/int64 Identity/Constant. Branching, multiple
inputs/outputs, default overrides,
passthroughs, scalar/empty tensors and symbolic leading batches are supported.
Constant accepts dense tensors and float/int scalar/list forms. Relu propagates
NaN; empty-axis Softmax is an explicit extension of the pinned evaluator.

`Load`, `LoadBytes`, and `LoadFile` return an immutable shared Model. Create a
Session with `NewSession`, then call `Run` with named inputs and optional requested
outputs. Outputs own independent storage and survive later runs and Close.
Overlapping calls on one session return `ErrBusy`; separate sessions can run
concurrently. `Options.Limits` applies finite load, tensor, live, workspace and
session bounds. External tensors and resolver options remain unsupported until
RT-14. `Capabilities()` returns copied manifest records, including planned entries
and the exact implemented restrictions.

The module pins `neuralnetwork v0.7.0`, protobuf v1.36.12, and Go 1.27.2 with a
Go 1.26.1 baseline. Ordinary builds/tests require neither protoc nor Python.
No general ONNX compatibility is claimed beyond the documented subset.

## Implementation plans

- [NEEDS.md](NEEDS.md) records the verified `neuralnetwork` v0.7.0 baseline,
  completed NN-01–NN-12 tasks, published handoffs, and integration constraints.
  NN-13 optimization remains blocked on this runtime's RT-12 milestone, as
  documented in [neuralnetwork PR #114](https://github.com/itsmontoya/neuralnetwork/pull/114).
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
Broadcasting, matrix products and activations are implemented. RT-10 expands the
fixture/conformance harness; RT-11 supplies user examples. Later milestones add
tensor transformations, 2D CNN operators, external weights and a documented release.

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
Runnable inference examples arrive at RT-11. Pull request descriptions must
preserve [the repository template](.github/PULL_REQUEST_TEMPLATE.md).
