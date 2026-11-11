# onnx

A Go-native ONNX inference runtime built on
[`neuralnetwork` v0.10.4](https://github.com/itsmontoya/neuralnetwork/tree/v0.10.4).
Load an `.onnx` model, bind named immutable tensors, and evaluate named outputs
on the CPU with `CGO_ENABLED=0`, without Python or native runtime libraries.

RT-01–RT-19 implement the documented subset: **IR 8, exactly one
empty standard-domain import at exactly opset 13 or 17, dense float32/int64, serial CPU execution**.
The [generated support table](CAPABILITIES.md)
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
```

The initial clone/module download needs network. Cached builds and tests use
checked-in models and typed reference values without network, protoc, Python,
C/C++ inference libraries or an external service. The `purego` tag selects the
portable backend CPU path. The module requires
Go 1.27.2, `neuralnetwork` v0.10.4 in [go.mod](go.mod), and protobuf v1.36.12.

Detailed guides and historical reports below link to their last tracked
versions; the `docs/` and `tools/` directories are no longer included in this repository.

[Getting started](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/getting-started.md) covers the public APIs, named I/O,
compatible model authoring/export checks, external weights and reproducible
failure cases. [Development](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/development.md) gives full verification and
optional regeneration commands.

The archived [ONNX Runtime comparison tool](https://github.com/itsmontoya/onnx/blob/c2d5123/tools/compare_runtime/README.md)
runs a pinned pretrained MNIST model through both runtimes, checks numerical
and top-1 agreement, and records CPU inference timings. It also accepts other
compatible local models with named NumPy inputs.
With published `neuralnetwork` **v0.10.4**, seven alternating precompiled
benchmark pairs on Apple M3 (Go 1.27.2, `CGO_ENABLED=0`, `GOMAXPROCS=1`,
`GOARM64=v8.0`) measure **144.422 → 125.644 µs/op** versus v0.10.3:
**13.0% lower MNIST latency**, with unchanged **179,440 B/op / 174 allocations**.
Samples span 141.940–150.267 µs before and 124.096–130.709 µs after, with
sample CVs of 1.90% and 1.99%. The median paired latency reduction is 13.0%,
with individual reductions of 10.2–13.4%.

The backend's ARM64 residual convolution tiles share weights across two to four
columns, including padded boundaries and interior tails. The second prepared
Conv falls **67.912 → 47.690 µs (29.8% lower latency)**; the first remains
approximately 16.3 µs. Other prepared operators retain similar timings.
The unchanged 19-case comparison, repeated seven times per version, measures
Go at **147.042 → 128.167 µs**, versus ORT at **44.500 → 44.417 µs**:
a **3.30× → 2.89×** median gap. ORT uses one thread, sequential execution and
disabled graph optimization. Each case retains ten warmups and 100 timed calls;
the Go benchmark uses a separate deterministic input. These are workload- and
hardware-specific results, not a general performance guarantee.

All numerical and top-1 checks pass; outputs remain bit-identical to v0.10.3.
An additional 131 archived probes also match bit for bit across v0.10.3,
v0.10.4 and v0.10.4 `purego`. Fresh profiles confirm the residual SIMD path is
selected. Convolution remains the largest measured cost: its two prepared
calls total about 63.9 µs after the upgrade, approximately 51% of inference.
Pooling is the next largest operator family at approximately 17%. Default,
cgo-disabled, `purego` and race suites, bounded fuzzing, and module/build checks
validate the integration. No ONNX API, operator coverage, numerical tolerance
or model-specific branch changed.

The archived [comparison results](https://github.com/itsmontoya/onnx/blob/c2d5123/tools/compare_runtime/README.md),
[fifth-pass investigation](https://github.com/itsmontoya/onnx/blob/c2d5123/tools/compare_runtime/pass5/README.md) and
[v0.10.2 validation](https://github.com/itsmontoya/onnx/blob/c2d5123/tools/compare_runtime/pass5/v0.10.2/README.md)
retain the earlier measurements and profiling evidence.

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
checksum metadata is unsupported. See the [runtime contract](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/runtime-contract.md)
for defaults, accounting, platform restrictions and exact numerical policies.

## Compatibility evidence and release status

[Fixture provenance](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/release-checklist.md#reference-evidence) records actual
reference comparisons and exceptions. M1–M4 examples, offline comparisons,
resource/lifecycle tests and bounded fuzz targets pass. Empty-value, pooling
and BatchNormalization evaluator/ORT differences remain explicit; skipped tests
never count as support. The [RT-16 report](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/performance/rt16/README.md)
records fixed-graph storage savings and latency tradeoffs. No general performance
or numerical error bound is claimed.

The [release checklist](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/release-checklist.md) records checks and known gaps.
**There is no project LICENSE.** Maintainers must decide and document licensing
for runtime source and original fixtures before public release. The included
[third-party notices](third_party/README.md) cover their identified upstream
artifacts and dependencies; fixture origin alone is not a license. Publishing a
tag is a separate maintainer action; RT-18 does not publish one.

## Contributing and next work

Read [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md) and
[ROADMAP.md](ROADMAP.md). [TODO.md](TODO.md) gives the sequential task order;
[NEEDS.md](NEEDS.md) records the pinned backend and published handoffs.
The [RT-17 handoff](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/handoff-rt17.md) records the schema audit, actual
unmodified PyTorch 2.2.2 IR-8/opset-17 MLP export and validation.
[RT-16](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/handoff-rt16.md) records measured call-local storage reuse. Future
model-selected work extends coverage beyond [RT-19’s audited corpus and first normalization set](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/handoff-rt19.md).
The [compatibility report](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/model-compatibility.md) and independent task files
keep unselected families explicit. Backend kernel availability does not broaden runtime compatibility.

Preserve [the PR template](.github/PULL_REQUEST_TEMPLATE.md) and record exact
revisions, APIs, checks and limitations in task handoffs.
