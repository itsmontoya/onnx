# RT-11 first working runtime integration report

Verified on 2026-11-07 with Go 1.27.2 on Darwin/arm64, based on RT-10 revision
`dc9898b48e23c6fab82429f8f0039718342b51bc`. Implementation revision is
`e527d778ace196ce6ddb8362cc349501ec6859bf`; the final handoff revision is
recorded in [the RT-11 handoff](../ROADMAP.md). This task adds fixtures,
examples, tests and documentation; no production implementation, public API,
capability status, dependency or numerical policy changes.

## Implemented profile and dependencies

M1 accepts IR 8, exactly empty-domain opset 13, dense float32 numerical operators,
and float32/int64 Identity/Constant. All 12 implemented operators and their
arity/type/rank/attribute/empty-value restrictions are listed in the
[generated M1 table](getting-started.md#implemented-capability-table), sourced from
[capabilities.json](../capabilities.json). Planned capability records establish
no runtime support. This report verifies concrete cases, not arbitrary ONNX
models or complete upstream profile coverage.

The unchanged backend is published `github.com/itsmontoya/neuralnetwork v0.7.0`,
commit `6048452c57a0a293db4591cd2e2febd7d6b743a3`; protobuf is v1.36.12.
The module language baseline is Go 1.26.1 with a Go 1.27.2 toolchain. No local
replacement, unreleased backend API, Python or native inference runtime is
required for ordinary use.

## Reproducible graphs and provenance

Both original graphs store four inline float32 weight/bias initializers and
declare `["batch", 2]` named I/O. The MLP has a three-unit hidden layer:
`Gemm(features) → Relu → Gemm → Softmax(probabilities)`. The branch reuses these
affine layers, adds the second named `residual` input to logits, and fans out
into `Softmax(adjusted) → probabilities` and `Sub(adjusted, logits) → delta`.
It retains shared logits and a join rather than flattening into a sequential
network. Inputs `features` and `residual` share the same batch symbol.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| [MLP model](../testdata/inference/models/mlp.onnx) | 498 | `04df066d0a12b0e6ddf2ba89e9e452768939a745ddd574cfb39780d0e14c06f9` |
| [Branch model](../testdata/inference/models/branch.onnx) | 662 | `3c8710155dd18231cdfcc730d0a98b2d6c158dfeb8c7a47c1f3d0765982f4446` |
| [Provenance manifest](../testdata/inference/provenance.json) | 9,097 | `9fa5307cf7e12b632d1057f6d6cac8c8f799ae8dcc1fd21ef5267ea95058feb7` |

The manifest records exact model/value hashes, named raw TensorProto input/output
artifacts, IR/import/schema versions, origin, license gap, tolerances and actual
reference observations for all four cases. The source is original
`tools/check_inference.py`; no upstream model is copied. This repository has no
LICENSE and these original fixtures have no separately declared license. No
license is inferred from their origin; release preparation must resolve that gap.

Development references are Python 3.9.6, ONNX 1.16.2 from source revision
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, NumPy 1.26.4 and protobuf 4.25.3.
The generator enforces exact versions and full ONNX checker acceptance. Every
golden comes from an actual `onnx.reference.ReferenceEvaluator` return, then is
cross-checked using independent NumPy affine/ReLU/stable-softmax expressions.
No RT-11 ORT comparison is claimed. [The fixture guide](../testdata/inference/README.md)
documents verification/regeneration; repeated generation reproduced all bytes
and checked the guide's M1 table against the canonical manifest.

## Actual numerical and lifecycle results

| Model | Batch | Named inputs | Named outputs | Result |
| --- | ---: | --- | --- | --- |
| MLP | 1 | features | probabilities | Reference agreement |
| MLP | 3 | features | probabilities | Reference agreement |
| Branch | 1 | features, residual | probabilities, delta | Reference agreement |
| Branch | 3 | features, residual | probabilities, delta | Reference agreement |

Types, shapes, names and counts compare exactly. Finite floats use
`abs(got-want) <= 1e-5+1e-5*abs(want)`; NaN classification, infinity signs and
signed zeros use RT-10's explicit comparator. These fixtures are finite.
The branch `delta` reflects ordinary float32 rounding, including reference
values `-0.5000000596046448` and `0.7499999403953552`; the CLI rounds display to
six places. This tolerance is a fixture policy, not a universal error bound.

[Offline integration tests](../inference_integration_test.go) verify model/value
hashes and envelopes, metadata with symbolic leading dimensions, and both batch
sizes through one session per model. They also reverse requested output order
by name and compare previous outputs after later runs and both Close calls.
The separate concurrency case starts two sessions together on each shared
numerical model, at batches 1 and 3, joins both calls and compares goldens. It
passes the race detector. Existing lifecycle/cancellation/budget tests remain
passing; failed runs cannot poison a session, shared constants remain immutable
and owned output storage survives closure.

These are **four additional reference agreements**. RT-09 remains 81 models:
80 actual reference agreements plus one explicit empty-axis Softmax extension.
At the RT-11 revision, RT-10 had 86 agreements, one extension and nine explicit unsupported-model
rejections. Its pinned source audit still contains 1,422 backend models and
zero exact IR-8/opset-13 envelopes; original upstream bytes remain rejection
fixtures. No skips, evaluator rejection or deferred shape/CNN observation count
as implemented numerical support.

## Verification

All of the following passed locally using
`GOCACHE=/private/tmp/onnx-runtime-gocache`, cached Go dependencies,
`GOPROXY=off` and `GOSUMDB=off`:

```sh
go fmt ./...
go mod tidy -diff
go mod verify
go build ./...
go vet ./...
go test ./...
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go test -race ./...
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph branch -batch 3
CGO_ENABLED=0 go run -tags=purego ./examples/inference -graph branch -batch 3
```

The optional pinned Python profile/schema checker, 81-model M1 checker,
96-case conformance regeneration/audit and RT-11 generation check also passed.
Normal Go tests call none of those tools and require no network, Python, protoc
or native shared runtime. Formatting, local Markdown targets, handwritten Go
if/var spacing and type/method colocation, contract Go-block formatting and diff
whitespace checks passed. A fresh clone of the implementation revision passed
cached dependency download, the full cgo-disabled suite and all four examples
offline with a clean tree. [Hosted Linux verification](https://github.com/itsmontoya/onnx/actions/runs/42425356391)
passed formatting, tidy, build, vet, normal, cgo-disabled, purego and race checks
at that revision. Final stacked-PR status is verified before the next task.

## Limits and next work

Outputs own copies; model/session cleanup releases their respective references
and finite session slots. Default load, initializer, tensor, live, workspace and
session budgets remain enabled. Cancellation is checked around kernels and
between nodes, without an in-kernel interruption or latency promise. Logical
payload accounting is not a hard process-memory quota.

Shape/transform operators are RT-12, CNN adapters are RT-13, external data is
RT-14, broader lifecycle auditing is RT-15, and optimization is later work.
Other versions/domains/dtypes, quantization and control flow remain unsupported.
Exporters must be configured and validated for the exact subset; changing
version fields alone does not convert a model. No benchmark measurements,
speedups or latency claims were produced by RT-11.

RT-12 must add its shape-heavy graph and publish its own accessible revision,
hashes, named I/O and reference outputs for at least two batches. Together with
these reproducible MLP models, that enables the
[NN-13 measurement plan](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-13.md).
RT-11 alone does not unblock NN-13. The first working M1 inference demonstrated
here already runs without those later tasks.

RT-12 subsequently adds all M2 operators and a shape-heavy graph at both batch
sizes. Its [current handoff](handoff-nn13.md) supplies NN-13's runtime inputs.
The conformance corpus now executes the former Gather rejection as one explicit
shape difference, leaving 86 agreements, one extension and eight unsupported
rejections. The report above preserves the RT-11 measurements and fixture bytes.
