# Development and verification

RT-02 establishes module `github.com/itsmontoya/onnx`, package `onnx`, with a
Go 1.26.1 language baseline and the selected Go 1.27.2 toolchain. The pinned
`neuralnetwork v0.7.0` module requires Go 1.26.1; its SIMD and `x/sys`
dependencies require Go 1.25.0. The pure-Go protobuf dependency is
`google.golang.org/protobuf v1.36.12` (Go 1.23). No local path replacements
or workspace are required.

The root package currently contains documentation and a dependency smoke test.
That test exercises exact int64 tensor storage and known protobuf wire bytes;
it verifies the dependency/toolchain foundation, not ONNX model execution.
Every operator in [the capability manifest](../capabilities.json) remains planned.

## Ordinary builds and tests

Install Go 1.27.2, then run these commands from the repository root:

```sh
go mod download
go build ./...
go fmt ./...
go vet ./...
go test ./...
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go test -race ./...
git diff --check
```

Dependency download initially requires network access. Once dependencies are
cached, ordinary builds/tests use only Go and checked-in artifacts. They need
neither Python nor protoc, Metal, a native inference runtime, or shared libraries.
The race detector is a development check and requires a supported platform and
cgo toolchain; cgo-disabled runtime builds do not acquire that requirement.
The `purego` tag selects the backend's portable CPU path.

[The CI workflow](../.github/workflows/go.yml) runs formatting, module tidiness,
build, vet, normal tests, both cgo-disabled configurations, and race tests on
Linux. Local baseline checks also run on Darwin/arm64. A separate linter is not
configured. Use `go mod tidy -diff` to check module metadata without rewriting it.

## Package boundaries

The [runtime contract](runtime-contract.md#public-surface-to-implement) defines
the following boundaries. Create each internal package when its implementation
task introduces real behavior, rather than adding empty packages in advance.

| Location | Responsibility | First task |
| --- | --- | --- |
| Root `onnx` package | Public values, metadata, load/model/session facade | RT-04/RT-08 |
| `internal/onnxpb` | Generated ONNX wire types, isolated from the public API | RT-03 |
| `internal/decode` | Bounded protobuf and tensor decoding | RT-03/RT-05 |
| `internal/graph` | Named graph validation and immutable planning | RT-06/RT-07 |
| `internal/operators` | Versioned schema planning and numerical adapters | RT-07/RT-09 |
| `testdata` | Checked-in model/value fixtures and provenance | RT-01/RT-10 |

Internal packages use backend tensors and internal metadata records without
importing the public facade. Concrete numerical storage and kernels remain in
`neuralnetwork/tensor`.

## Regeneration

Regeneration is a development operation, separate from ordinary builds. RT-03
will introduce pinned protobuf generation tools, vendored schema/license files,
and checked-in generated bindings. There are no protobuf generation commands to
run yet. Do not require generation during `go build` or `go test`.

RT-01's existing profile-check fixture can be verified using its pinned Python
3.9.6 environment with ONNX 1.16.2, NumPy 1.26.4, and protobuf 4.25.3:

```sh
python tools/check_profile.py
python tools/check_profile.py --write
```

The first command checks the fixture and selected schemas; the second explicitly
regenerates the original Identity fixture and provenance. Neither command is a
Go test prerequisite. See [the runtime contract](runtime-contract.md) for source
revisions and reference limitations.
