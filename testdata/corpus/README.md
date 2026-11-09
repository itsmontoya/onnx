# Audited redistributable model corpus

All 33 `.onnx` files and the 19 enabled models' input/output TensorProtos are
**unmodified Apache-2.0 ONNX v1.16.2 artifacts**, copied from revision
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`. The original
[LICENSE](../../third_party/onnx/LICENSE) is retained byte-for-byte; the pinned
archive has no NOTICE file. [The source export](source/layernormalization.py)
retains its original SPDX header. No pretrained weights, external-data file or
framework licensing assumption is involved. Repository source and the separate
original test-only normalization fixtures still have no declared project license.

[index.json](index.json) records each model's source URL/path/SHA-256, license,
actual IR/imports, ordered nodes and applicable schemas, named types/shapes,
attributes, initializers, external requirements, and deterministic before/after
unsupported-feature findings with counts and node indices. The source archive
SHA-256 is `d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5`.
Its 1,422 backend models contain zero exact IR 8/import 13 and 65 exact
IR 8/import 17 exports. This is a selected small corpus, not an industry coverage
percentage. All original model bytes and version metadata are preserved.

RT-19 enables 19 normalization exports with one float32 LayerNormalization-17
adapter, including all three named outputs. The 14 remaining models are
**negative fixtures**. They represent expanded primitive normalization,
Slice/Expand/Pad/Resize, Where/bool, low precision, recurrent, quantization,
control flow and structured values. Several also require separate IR/import
audits; their envelopes are never relabeled. Go tests require their actual
`ErrUnsupported` failures. No skipped model counts as support.

The deterministic static report covers envelope, public graph value types,
operator records, allowed attributes and output slots for this selected corpus.
It is not a second full runtime validator or a public compatibility API. The
runtime's actual Load validation and contextual first diagnostic are exercised
separately. Findings can overlap: a model can fail both profile and operator.

| Scope | Enabled | Unsupported | Findings |
| --- | --- | --- | --- |
| Before RT-19 | 0 | 33 | 63 |
| After RT-19 | 19 | 14 | 44 |

Enabled model values are checked against independent NumPy centered population
variance and both pinned oracles in
[normalization provenance](../normalization/provenance.json). Source values keep
their original bytes and hashes. Normal Go tests are offline/Python-free.

Reproduce with the existing reference environment and separately downloaded
pinned official source archive:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_corpus.py --source-archive /private/tmp/rt10-onnx-source.tar.gz
CGO_ENABLED=0 go test . -run 'TestAuditedCorpus|TestNormalization' -count=1
CGO_ENABLED=0 go run ./examples/normalization
```

The generator checks Python 3.9.6/ONNX 1.16.2/NumPy 1.26.4/protobuf 4.25.3/ORT 1.19.2,
source hashes, exact LayerNormalization schema/defaults and all artifacts. It
never downloads or installs anything. `--write` deliberately regenerates only
new corpus/normalization artifacts; no-write rejects missing/extra/drifted files.
Original helper test graphs are kept in a separately labeled directory.
