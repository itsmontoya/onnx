# onnx

A planned Go-native ONNX inference runtime built on
[`github.com/itsmontoya/neuralnetwork`](https://github.com/itsmontoya/neuralnetwork).
The goal is to load an `.onnx` model, bind named Go tensor inputs, and evaluate
named outputs on the CPU without cgo or an installed native inference runtime.

RT-01 now specifies the [runtime contract](docs/runtime-contract.md) and an
initial [capability manifest](capabilities.json); every operator is still marked
planned. This repository has no runnable inference implementation yet.
RT-02 establishes the Go module, pinned `neuralnetwork v0.7.0` and protobuf
dependencies, and a verification workflow. There is no model loader or inference
implementation yet, and no model compatibility is claimed. RT-03 adds generated
ONNX wire bindings and bounded internal decoding; ordinary builds and tests need
neither protoc nor Python. RT-04 adds immutable float32/int64 tensors and model
metadata records, with owned constructor/accessor copies and exact integer
storage. RT-05 adds shared internal inline TensorProto conversion with checked
payload budgets, exact typed/raw values, and owned backend copies. Public
loading and execution arrive in later roadmap tasks.

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

## Proposed scope

Begin with CPU inference for a deliberately narrow profile: ONNX IR 8,
standard-domain opset 13, float32 computation, and int64 constants and shape
values. Establish named graph inputs/outputs, branching, broadcasting, matrix
products, and activations first. Then add tensor transformations, 2D CNN
operators, external weights, and a documented release.

The [roadmap's coverage tables](ROADMAP.md#milestones-and-coverage) specify
operator schema versions, supported subsets, and milestone requirements. Newer
IR/opsets, other data types, recurrent models, transformers, quantization, and
control flow need additional tasks and reference fixtures.

## Architecture

This module will own ONNX protobuf loading, compatibility checks, named graph
planning, sessions, diagnostics, and model execution. `neuralnetwork` provides
concrete tensors and forward numerical kernels through additive APIs that preserve
its existing matrix, training, and serialization behavior.

The intended runtime works with `CGO_ENABLED=0`, with a portable `purego` path.
Go dependencies such as protobuf bindings are allowed. Python reference tooling
may generate development fixtures, while ordinary builds and tests will use
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
