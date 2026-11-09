# onnx

A Go-native ONNX inference runtime built on
[`neuralnetwork v0.7.0`](https://github.com/itsmontoya/neuralnetwork/tree/6048452c57a0a293db4591cd2e2febd7d6b743a3).
Load an `.onnx` model, bind named immutable tensors, and evaluate named outputs
on the CPU with `CGO_ENABLED=0`, without Python or native runtime libraries.

RT-01–RT-15 and RT-18 complete the documented M1–M4 subset: **IR 8, exactly one
empty standard-domain opset-13 import, dense float32/int64, serial CPU execution**.
The [generated support table](docs/getting-started.md#implemented-capability-table)
records the 25 operators and their dtype, rank, attribute and optional-slot
restrictions. Branching, multiple named inputs/outputs, defaulted initializers,
symbolic leading input batches, scalar/empty tensors, runtime shape subgraphs,
NCHW 2D CNNs and explicitly resolved external weights are implemented.
Compatibility with arbitrary ONNX models is not claimed.

## Run inference

Install Go 1.27.2, then run from the repository root:

```sh
go mod download
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 3
CGO_ENABLED=0 go run ./examples/external -batch 3
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go run ./tools/capabilitiesgen
```

The initial clone/module download needs network. Cached builds and tests use
checked-in models and typed reference values without network, protoc, Python,
C/C++ inference libraries or an external service. The `purego` tag selects the
portable backend CPU path. The module has a Go 1.26.1 language baseline and pins
Go 1.27.2, neuralnetwork v0.7.0 and protobuf v1.36.12.

[Getting started](docs/getting-started.md) covers the public APIs, named I/O,
compatible model authoring/export checks, external weights and reproducible
failure cases. [Development](docs/development.md) gives full verification and
optional regeneration commands.

## Runtime contract

`Load`, `LoadBytes` and `LoadFile` publish immutable shared Models. Create a
Session with `Model.NewSession`, then call `Session.Run` with named tensors.
Requested outputs own independent storage and survive later runs and Close;
constructors and typed accessors copy. Input/initializer overlap supplies an
overridable default within the declared input type/shape contract.

One session permits one active run; overlapping Run/Close returns `ErrBusy`.
Separate sessions can run concurrently. `Model.Close` prevents new sessions;
existing sessions remain usable. All nodes execute even when requesting fewer
outputs. Failed/cancelled runs publish no partial outputs and leave the session
reusable. Errors preserve classifications, structured diagnostics and causes.

`Options.Limits` supplies finite logical load/tensor/live/workspace/session
budgets, rather than a process RSS cap. Cancellation brackets synchronous
backend/protobuf operations; a running kernel has no interruption or latency
guarantee. External tensors require `Options.ExternalData` even for LoadFile.
`DirectoryResolver` confines regular-file ranges on supported Go platforms;
checksum metadata is unsupported. See the [runtime contract](docs/runtime-contract.md)
for defaults, accounting, platform restrictions and exact numerical policies.

## Compatibility evidence and release status

[Fixture provenance](docs/release-checklist.md#reference-evidence) records actual
reference comparisons and exceptions. M1–M4 examples, offline comparisons,
resource/lifecycle tests and bounded fuzz targets pass. Empty-value, pooling
and BatchNormalization evaluator/ORT differences remain explicit; skipped tests
never count as support. No performance improvement or general numerical error
bound is claimed. [NN-13's baseline commands](docs/handoff-nn13.md) support later
measurements; RT-16 optimization remains separate.

The [release checklist](docs/release-checklist.md) records checks and known gaps.
**There is no project LICENSE.** Maintainers must decide and document licensing
for runtime source and original fixtures before public release. The included
[third-party notices](third_party/README.md) cover their identified upstream
artifacts and dependencies; fixture origin alone is not a license. Publishing a
tag is a separate maintainer action; RT-18 does not publish one.

## Contributing and next work

Read [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md) and
[ROADMAP.md](ROADMAP.md). [TODO.md](TODO.md) gives the sequential task order;
[NEEDS.md](NEEDS.md) records the pinned backend and published handoffs.
The [RT-18 handoff](docs/handoff-rt18.md) describes the current documentation
and validation. Later tasks cover measured optimization (RT-16, after NN-13),
additional explicit version profiles (RT-17), and audited model-selected coverage
(RT-19). Backend kernel availability does not broaden runtime compatibility.

Preserve [the PR template](.github/PULL_REQUEST_TEMPLATE.md) and record exact
revisions, APIs, checks and limitations in task handoffs.
