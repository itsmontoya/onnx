# onnx

A planned Go-native ONNX inference runtime built on
[`github.com/itsmontoya/neuralnetwork`](https://github.com/itsmontoya/neuralnetwork).
The goal is to load an `.onnx` model, bind named Go tensor inputs, and evaluate
named outputs on the CPU without cgo or an installed native inference runtime.

RT-01 now specifies the [runtime contract](docs/runtime-contract.md) and an
initial [capability manifest](capabilities.json); every operator is still marked
planned. This repository has no runnable inference implementation yet.
The tensor storage and
numerical kernels are available in `neuralnetwork` v0.7.0, but there is no Go
module or runtime implementation here yet, and no model compatibility is claimed.

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

Read [AGENTS.md](AGENTS.md) and [STYLEGUIDE.md](STYLEGUIDE.md), then the task and
its dependency handoffs. Start with RT-01 in `ROADMAP.md`, using the released
backend contract linked from `NEEDS.md`. Implement runtime tasks here and
coordinate later library changes in `neuralnetwork`. Record exact
APIs, dependency revisions, ownership rules, validation results, and limitations
in each completed task's handoff.

Build commands and runnable API examples will be added when the module and
first inference milestone exist. Pull request descriptions must preserve
[the repository template](.github/PULL_REQUEST_TEMPLATE.md).
