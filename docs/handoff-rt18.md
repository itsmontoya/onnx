# RT-18 usage and compatibility handoff

Completed on 2026-11-08 against RT-15 base
[`3a5b5b3baa0bb1bce1e6a7d378157ab2f1c8aac0`](https://github.com/itsmontoya/onnx/tree/3a5b5b3baa0bb1bce1e6a7d378157ab2f1c8aac0).
The documentation/tooling revision is
[`928496894ca25fa0dd2edbca939abb9081e54a72`](https://github.com/itsmontoya/onnx/tree/928496894ca25fa0dd2edbca939abb9081e54a72).
The completion revision is the commit introducing this handoff and the RT-18
checkbox in ROADMAP.md. The dedicated branch is
[rt-18-usage-compatibility](https://github.com/itsmontoya/onnx/tree/rt-18-usage-compatibility),
stacked on **rt-15-runtime-hardening**. Keep its PR open; no merge/tag/publication
is part of RT-18.

## Changes and APIs

[README](../README.md), [getting started](getting-started.md),
[development](development.md) and the [runtime contract](runtime-contract.md)
now describe the actual M1–M4 implementation. Package/Capability GoDoc removes
stale Identity-only/static-planning descriptions. The guide includes compiling
load/session/named-Run/external examples, metadata/default overrides, constructor/
accessor copying, independent output lifetime, concurrency/ErrBusy, Close
behavior, logical limits and synchronous cancellation. It documents exact
external ranges/reader closure, required explicit resolver, unsupported checksum
metadata and DirectoryResolver's platform/filesystem boundaries.

The public API and behavior remain unchanged: Load/LoadBytes/LoadFile, Options/
Limits/ExternalDataResolver/DirectoryResolver, Model.Metadata/NewSession/Close,
Session.Run/Close, Tensor constructors/accessors, RunOptions/NamedTensor,
Capabilities and Error/sentinels. IR remains exactly 8, standard empty-domain
import exactly opset 13, dense float32/int64, CPU serial scheduling and the same
25 operators. No additional feature, version profile, optimizer, backend bridge,
performance optimization or dependency is introduced.

[tools/capabilitiesgen/main.go](../tools/capabilitiesgen/main.go) renders the marked
support table directly from `Capabilities()`. The compiler registry consumes
those same embedded capabilities.json records; there is no handwritten support
list or second renderer. `CGO_ENABLED=0 go run ./tools/capabilitiesgen` checks
without Python, and `-write` updates only that table. Attributes sort by name;
RawMessage defaults keep exact numeric text and string defaults display their
value. Missing/repeated/out-of-order markers fail. The reference generator
[tools/check_inference.py](../tools/check_inference.py) delegates to it;
[CI](../.github/workflows/go.yml) checks table drift separately. The inference
fixture README explains this Go requirement for optional reference generation.
No runtime model/value/provenance bytes changed.

[The release checklist](release-checklist.md) records actual technical checks,
reference outcomes, known gaps and separate maintainer actions.
[Third-party notices](../third_party/README.md) identify the included ONNX
Apache-2.0 schema/upstream rejection fixtures and exact unmodified LICENSE copies
for backend/SIMD MIT and protobuf/x/sys/Go BSD notices, with source links and
SHA-256. There is still **no project LICENSE**, and original fixtures still have
no separately declared license. Origin is not a licensing grant. Maintainers
must make that decision before public release; no license was invented.

README/TODO/NEEDS and ROADMAP current-status records now agree. Historical task
handoffs, original dependency-review history, numerical exception records and
fixture provenance remain intact. The backend remains v0.7.0 at
`6048452c57a0a293db4591cd2e2febd7d6b743a3`; protobuf remains v1.36.12; module
language baseline Go 1.26.1 and selected toolchain Go 1.27.2 are unchanged.
The adjacent neuralnetwork checkout was not modified.

## Validation

Local verification used **Go 1.27.2, Darwin/arm64**, cached offline modules and
`GOCACHE=/private/tmp/onnx-runtime-gocache`. Passed `go mod verify`,
`go mod tidy -diff`, `go fmt ./...`, support-table check, `go build ./...`,
`go vet ./...`, and uncached `go test -count=1 ./...` in normal,
`CGO_ENABLED=0`, `CGO_ENABLED=0 -tags=purego` and race configurations. Full
commands are in the release/development guides. No standalone benchmark result
is published; the historical one-iteration smoke remains explicitly a smoke.
NN-13/RT-16 measurement commands and their limitations remain in the existing
[NN-13 handoff](handoff-nn13.md).

A deliberate changed table row and missing end marker each caused a nonzero
no-write check. `-write` restored the exact original guide, including all
surrounding text. Newly documented API snippets compiled; the Go-only failure
probe verified ErrUnsupported for unchanged IR-7 input and unresolved external
weights, ErrLimit for a one-byte wire limit, and ErrInvalidArgument for missing
`features`. These checks assert error identity with errors.Is, preserving
freedom to improve diagnostic text.

A fresh clone at `928496894ca25fa0dd2edbca939abb9081e54a72` used `env -i`, a new
HOME/build cache, PATH containing only the selected Go toolchain's bin,
`GOTOOLCHAIN=local`, `GOWORK=off`, `GOPROXY=off`, `GOSUMDB=off`, cached module
sources and `CGO_ENABLED=0`. All **20** commands passed: MLP/branch/shape/CNN
at batches 1/3 and external MLP at batches 1/3, each in default and purego modes.
The table check, documented failure probe and inference binary build also passed.
The clone remained clean. `go version -m` confirmed CGO_ENABLED=0 and pinned
module sums; Darwin `otool -L` showed only system libSystem/libresolv, with no
native inference library. The guide documents a repeatable cleared-environment
command; initial clone/toolchain/module provisioning still requires installation
and network or preloaded caches.

`go run ./tools/onnxgen` with protoc 33.2 and all seven no-write Python checks
(profile, M1, M2, M3, inference, external and conformance) reproduced unchanged
artifacts using the pinned reference environment. Conformance used the existing
pinned source archive with SHA-256
`d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5`.
Actual agreement/extension/difference/rejection classifications remain in the
[release evidence table](release-checklist.md#reference-evidence); expected ORT
errors and shape warnings reproduced those records and are not new agreements.
No third-party exporter has been tested. The tested positive origins are original
helper/numpy_helper-authored IR-8/opset-13 graphs; unchanged upstream IR-7 bytes
remain rejection cases.

All four bounded native Go fuzz targets passed with `GOMAXPROCS=2`,
`-parallel=2`, `-fuzztime=3s`: protobuf **134,867**, tensor **81,227**, graph
**145,346**, planning **143,674** executions. Bounds/seeds remain those from RT-15;
fuzzing does not expand compatibility. License byte comparisons/hashes,
relative Markdown links/anchors, handwritten Go style/type-colocation review
and `git diff --check` pass. Hosted Linux checks cover the final PR head and
include the new table check plus inherited full tests/fuzzing; verify their
actual result on the PR, rather than treating local Darwin evidence as Linux.

## Remaining boundaries and next task

The [runtime contract](runtime-contract.md), generated restrictions and reference
exception records remain authoritative. Limits are logical payload reservations,
not process RSS; backend/protobuf calls do not support in-call cancellation;
DirectoryResolver is unsupported on js/plan9 and does not isolate hard links or
mounts. No skipped test establishes model/platform support. No universal exporter,
ONNX-compliance or performance claim is made.

RT-18's technical documentation/checklist acceptance is complete. Licensing,
review/merge and tag publication remain explicit maintainer actions. The next
roadmap task in document order is **RT-16**, after NN-13 supplies measured backend
results. RT-17 and RT-19 follow separately. NN-13 already has the RT-12 baseline
handoff and does not wait for RT-16; no later task was started here.
