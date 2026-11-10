# Pinned ONNX wire schema

`onnx.proto` and `LICENSE` are unmodified files from ONNX **v1.16.2**, commit
[`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`](https://github.com/onnx/onnx/tree/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557).
The schema is licensed under [Apache-2.0](LICENSE), as stated by its SPDX header.
The upstream release does not contain a root NOTICE file. This attribution
documents the included source; no additional upstream notice was omitted.
The scoped Git whitespace attribute preserves the schema's upstream trailing
blank line rather than changing the pinned bytes.

RT-10 also includes three unmodified backend model/value cases under
[`testdata/conformance/upstream/`](../../testdata/conformance/README.md).
Their provenance records exact upstream paths, this revision, Apache-2.0
attribution and each file's SHA-256. The license above covers these artifacts;
no additional root NOTICE exists in the pinned archive. These IR-7 cases are
explicit unsupported-version tests, not claimed runtime numerical support.

| File | Upstream path | SHA-256 |
| --- | --- | --- |
| `onnx.proto` | `onnx/onnx.proto` | `1023c7f39a568f309fdd0b445a81a7a0d90f74046e5967c2e817d4e4e06a5f1b` |
| `LICENSE` | `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |

This proto2 schema has no imports. Other ONNX schemas, such as the separate
operator-set serialization schema, are not needed to decode ModelProto or
TensorProto and are not included. The source defines newer features and versions
than this runtime supports; successful wire decoding does not accept them.

The [generation instructions](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/development.md#regeneration) use
protoc **33.2** (generated version string **6.33.2**) and
protoc-gen-go **v1.36.12**. The Go package mapping is supplied as a compiler
option so this vendored source remains byte-identical to upstream. Generated
declarations retain upstream/generated implementation style, with an automatic
AST split placing each type and its methods in one file.


RT-19 adds the [audited corpus](../../testdata/corpus/README.md):33 unchanged
Apache-2.0 model exports, enabled models' input/output values and their original
normalization source export. Exact upstream paths/SHA-256 and preserved versions
are in the corpus index. The same unmodified LICENSE above applies; the archive
has no root NOTICE. Nineteen LayerNormalization models execute and fourteen
other cases remain explicit rejections. Original helper graphs live separately
and do not inherit ONNX's license.
