# Third-party sources and notices

This inventory identifies included upstream artifacts and the dependencies in
`go.mod`. It does not assign a license to this repository's runtime source or
original fixtures: **no root project LICENSE is present**. Maintainers must
resolve that licensing decision before public release. Provenance/original
authorship does not supply a license. Preserve the listed license texts and
source notices when packaging a source archive or a binary distribution.

## Included ONNX artifacts

The unmodified `onnx.proto`, its Apache-2.0 LICENSE, and three unmodified upstream
backend model/value cases, plus the RT-19 audited corpus, come from **ONNX v1.16.2**, commit
[`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`](https://github.com/onnx/onnx/tree/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557).
[The schema attribution](onnx/README.md) records original paths, hashes, generator
versions and the absence of a root NOTICE in the pinned release archive.
[The included Apache-2.0 text](onnx/LICENSE) applies to these identified ONNX
artifacts; it does not license unrelated original runtime files.

Generated [Go bindings](../internal/onnxpb/doc.go) retain schema attribution and
generator headers. [Conformance provenance](../testdata/conformance/provenance.json)
records each copied upstream model/value path and SHA-256. These unchanged IR-7
models are rejection tests, not numerical support claims. Original M1/M2/M3,
inference, session and external fixtures have no separately declared license;
their provenance records this gap explicitly.

RT-19 additionally includes [33 byte-original models and the enabled models’
values/source export](../testdata/corpus/README.md) under this same Apache-2.0
license. The pinned archive has no NOTICE; exact source paths/hashes and notices
are in [the corpus index](../testdata/corpus/index.json). Nineteen normalization
models execute; fourteen remain negative fixtures. The separate original
normalization helper graphs have no redistribution grant or project license.

## Go runtime dependencies

License copies below are byte-identical to the LICENSE in each pinned downloaded
Go module; neuralnetwork v0.10.6 was rechecked on 2026-11-11. Module zip/content and
module-file checksums remain in [go.sum](../go.sum). No dependencies, replacements
or backend changes were introduced for RT-18.

| Module | Version | Primary upstream license | Included text |
| --- | --- | --- | --- |
| github.com/itsmontoya/neuralnetwork | v0.10.6, commit `b0f1aa2f6716e32b3fae610528ec00ebd33ab19e` | [MIT, copyright 2026 Josh Montoya](https://github.com/itsmontoya/neuralnetwork/blob/b0f1aa2f6716e32b3fae610528ec00ebd33ab19e/LICENSE) | [neuralnetwork-LICENSE](licenses/neuralnetwork-LICENSE) |
| google.golang.org/protobuf | v1.36.12 | [BSD 3-Clause, copyright 2018 The Go Authors](https://github.com/protocolbuffers/protobuf-go/blob/v1.36.12/LICENSE) | [protobuf-LICENSE](licenses/protobuf-LICENSE) |
| github.com/tphakala/simd (indirect) | v1.3.0 | [MIT, copyright 2025 Tomi P. Hakala](https://github.com/tphakala/simd/blob/v1.3.0/LICENSE) | [simd-LICENSE](licenses/simd-LICENSE) |
| golang.org/x/sys (indirect) | v0.45.0 | [BSD 3-Clause, copyright 2009 The Go Authors](https://cs.opensource.google/go/x/sys/+/v0.45.0:LICENSE) | [x-sys-LICENSE](licenses/x-sys-LICENSE) |

The selected **Go 1.27.2** toolchain's BSD 3-Clause text is included as
[go-LICENSE](licenses/go-LICENSE), copied from that toolchain's root LICENSE;
[the upstream source](https://cs.opensource.google/go/go/+/refs/tags/go1.27.2:LICENSE)
is the primary notice. The module language baseline is Go 1.27.2. Go's source
files retain any more specific embedded notices; review the actual artifacts
when deciding what to redistribute. A `purego` build does not change the module
versions pinned in go.mod.

| Included license | SHA-256 |
| --- | --- |
| Go / x/sys (identical texts) | `911f8f5782931320f5b8d1160a76365b83aea6447ee6c04fa6d5591467db9dad` |
| neuralnetwork | `97dec3b67a1c49a984ea92dba47e806df00c3a782d3625bb41c48630eededd05` |
| protobuf | `4835612df0098ca95f8e7d9e3bffcb02358d435dbb38057c844c99d7f725eb20` |
| simd | `da8b7883943f165928cd796cfb001f5161cec768f10ef5fd2a841d2907c0f71c` |

Recheck with `go mod verify`, `go list -m all` and
`shasum -a 256 third_party/licenses/*`. To compare a license copy directly, use
`cmp third_party/licenses/neuralnetwork-LICENSE "$(go env GOMODCACHE)/github.com/itsmontoya/neuralnetwork@v0.10.6/LICENSE"`
and the corresponding paths for the other modules. The module graph also retains
checksums for upstream testing dependencies; that does not make those packages
runtime imports. Inspect the target build with `go list -deps ./...`.

## Optional development tools

protoc 33.2 and protoc-gen-go v1.36.12 generate bindings; Python 3.9.6, ONNX
1.16.2, NumPy 1.26.4, Python protobuf 4.25.3 and ORT 1.19.2 generate/check reference
fixtures. They are not runtime dependencies and their packages/binaries are not
vendored here. Their versions and actual observations are recorded in each
[fixture provenance set](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/release-checklist.md#reference-evidence).
If distributing development tooling itself, inspect the corresponding upstream
release notices separately. ONNX Apache-2.0 attribution remains included for the
upstream schema and copied test artifacts even though normal users need no
Python ONNX installation.
