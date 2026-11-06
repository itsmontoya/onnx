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
RT-03 adds checked-in ONNX wire bindings and bounded internal protobuf decoding.
There is no public model loader or execution API yet.
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

Regeneration is a development operation, separate from ordinary builds. Install
the official [protoc 33.2 release](https://github.com/protocolbuffers/protobuf/releases/tag/v33.2)
for your platform, then run this command from the repository root with Go 1.27.2:

```sh
go run ./tools/onnxgen
git diff --exit-code -- internal/onnxpb
```

The tool verifies the [vendored schema](../third_party/onnx/README.md) SHA-256,
requires protoc 33.2 and the module's exact protobuf v1.36.12, builds that
version's official protoc-gen-go in a temporary directory, and invokes protoc.
It splits the generated Go AST into one file per type, retaining that type's
methods and enum values/maps together; descriptor initialization is separate.
It preserves the generated protobuf logic instead of maintaining a wire parser.
No schemas need importing. Repeated generation leaves all 36 generated files
unchanged. Generated code retains upstream implementation style; the handwritten
generation and decoding code follows this repository's style guide.

Only optional regeneration requires protoc. Go builds and tests use checked-in
bindings and require neither protoc nor Python. The generator itself can run
offline after module dependencies have been downloaded.

## Internal wire decoding

`internal/decode` provides ParseModel/ReadModel and ParseTensor/ReadTensor.
Each accepts a context and Limits: zero values select 64 MiB serialized bytes
and 64 levels of known-message recursion. Byte input is checked before
unmarshalling. Readers are consumed in chunks of at most 32 KiB, with at most
limit+1 bytes consumed to detect excess, and are never closed. I/O errors retain
their causes, repeated empty reads fail, and cancellation is checked between
reads and before/after the synchronous protobuf call.

Unknown fields are retained. In protobuf v1.36.12, MaxRecursion bounds known
messages; unknown groups instead use protowire's fixed recursive allowance of
10,000, permitting 10,001 nested group levels and rejecting the next level.
Tests exercise both limits. Serialized bytes remain bounded in either case.
This is the official decoder's behavior, not a uniform configurable depth bound.

Decoding verifies wire syntax, not ONNX compatibility. Empty protobuf messages,
unsupported versions/types, graph attributes, and other semantic features remain
wire-decodable for later validation; none establish inference support. The byte
limit bounds serialized input, not protobuf object-tree overhead or total heap.
RT-05 and later tasks add payload ownership/accounting and semantic validation.

## Profile fixture verification

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
