# onnx

A Go-native ONNX inference runtime built on
[`neuralnetwork` v0.8.0](https://github.com/itsmontoya/neuralnetwork/tree/v0.8.0).
Load an `.onnx` model, bind named immutable tensors, and evaluate named outputs
on the CPU with `CGO_ENABLED=0`, without Python or native runtime libraries.

RT-01–RT-19 implement the documented subset: **IR 8, exactly one
empty standard-domain import at exactly opset 13 or 17, dense float32/int64, serial CPU execution**.
The [generated support table](docs/getting-started.md#implemented-capability-table)
records 25 operators at import 13 and 26 at import 17, including float32
LayerNormalization, with their dtype, rank, attribute and optional-slot
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
CGO_ENABLED=0 go run ./examples/inference -graph exported -batch 3
CGO_ENABLED=0 go run ./examples/external -batch 3
CGO_ENABLED=0 go run ./examples/normalization
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go run ./tools/capabilitiesgen
```

The initial clone/module download needs network. Cached builds and tests use
checked-in models and typed reference values without network, protoc, Python,
C/C++ inference libraries or an external service. The `purego` tag selects the
portable backend CPU path. The module has a Go 1.26.1 language baseline and pins
Go 1.27.2, `neuralnetwork` v0.8.0 in [go.mod](go.mod), and protobuf v1.36.12.

[Getting started](docs/getting-started.md) covers the public APIs, named I/O,
compatible model authoring/export checks, external weights and reproducible
failure cases. [Development](docs/development.md) gives full verification and
optional regeneration commands.

The optional [ONNX Runtime comparison tool](tools/compare_runtime/README.md)
runs a pinned pretrained MNIST model through both runtimes, checks numerical
and top-1 agreement, and records CPU inference timings. It also accepts other
compatible local models with named NumPy inputs.
The [MNIST profiling report](docs/performance/mnist/README.md) records the
v0.8.0 comparison and allocation measurements.

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
never count as support. The [RT-16 report](docs/performance/rt16/README.md)
records fixed-graph storage savings and latency tradeoffs. No general performance
or numerical error bound is claimed.

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
The [RT-17 handoff](docs/handoff-rt17.md) records the schema audit, actual
unmodified PyTorch 2.2.2 IR-8/opset-17 MLP export and validation.
[RT-16](docs/handoff-rt16.md) records measured call-local storage reuse. Future
model-selected work extends coverage beyond [RT-19’s audited corpus and first normalization set](docs/handoff-rt19.md).
The [compatibility report](docs/model-compatibility.md) and independent task files
keep unselected families explicit. Backend kernel availability does not broaden runtime compatibility.

Preserve [the PR template](.github/PULL_REQUEST_TEMPLATE.md) and record exact
revisions, APIs, checks and limitations in task handoffs.
