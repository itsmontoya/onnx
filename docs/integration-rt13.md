# RT-13 CNN integration report

Verified on 2026-11-07 using Go 1.27.2, Darwin/arm64, from the RT-12 base
`007b8abe53b63f35ae4d09abaed93015b7256b4d`. Implementation revision:
[`fa06c9d58814727adfa97ac31646d16a101c4fca`](https://github.com/itsmontoya/onnx/tree/fa06c9d58814727adfa97ac31646d16a101c4fca). Backend remains published
`github.com/itsmontoya/neuralnetwork v0.7.0`, commit
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, with protobuf v1.36.12 and the
Go 1.26.1 module baseline. No local replacement or backend change is used.

All 25 manifest records are implemented within their exact restrictions. The
five new adapters are Conv-11, MaxPool-12, AveragePool-11,
GlobalAveragePool-1 and inference BatchNormalization-9. The accepted envelope
remains IR 8, standard empty domain and imported opset exactly 13. The generated
[restriction table](getting-started.md#implemented-capability-table) comes
from the same canonical manifest checked by the pinned schema tool.

## Behavior and ownership

The fixed-size `CNNConfig` parses actual schema defaults, validates attributes
and resolves concrete automatic padding without float conversions or unchecked
ceil additions. Conv uses NCHW and ONNX `[M,C/group,kH,kW]` weights directly,
including groups/depthwise multipliers/dilation and optional `[M]` bias.
Dynamic weights, bias and spatial shapes resolve each invocation. Explicit pads
are nonnegative; non-NOTSET auto_pad conflicts with a pads attribute. SAME_UPPER
and SAME_LOWER split checked effective-kernel padding according to the schema.

MaxPool produces values with storage_order 0; a second empty output slot is
preserved as -1. A named indices output or another storage order is rejected.
AveragePool has no schema-11 dilation attribute; count_include_pad selects full
kernel area or valid sample count. Ceil mode removes windows starting in the
right padded region. Nonpositive spatial extents clamp to zero. Pooling NaNs,
empty/all-padding windows and pinned oracle differences are recorded explicitly.

BatchNormalization accepts rank 2 or NCHW rank 4 and exactly one output slot.
Scale, bias, mean and variance are float32 `[C]`. Epsilon is finite/nonnegative;
momentum is finite in [0,1], retained as metadata and ignored by inference.
Supplied statistics never update. Later training_mode and additional output
slots (including empty training slots) are rejected. Negative variance and
nonfinite arithmetic follow the literal stored-statistics float32 formula.

The compiler checks known parts of ranks/channels/kernel/bias/geometry and
propagates metadata without numerical node outputs. Unknown intermediate shapes
or ranks defer to contextual Run checks. Fixed numerical step metadata and deep
snapshot accounting allow 256 bytes per step for embedded configs; copied shape
records remain separately charged. CNN calls reserve the complete destination
before allocation. The released direct Into loops need zero numerical scratch;
no numerical accessors, packing, layers, training state or caches are created.
Inputs/weights/statistics remain immutable, and public outputs are independent
copies. Cancellation brackets synchronous kernels; in-kernel interruption,
storage reuse and NN-13 optimization are outside this task.

## Fixtures and actual reference outcomes

[87 M3 fixtures](../testdata/m3/README.md) use independent NumPy window/einsum
convolution, indexed window reductions and stored-statistics normalization.
Actual ONNX 1.16.2 results classify as 49 agreements, 26 differences and 12
rejections. Actual ORT 1.19.2 results give 66 agreements, six differences and 15
rejections. Every returned oracle shape/float32 bit pattern or error is retained;
none of the differences/rejections is advertised as agreement.

The symbolic-batch CNN composes Conv → BatchNormalization → Relu → MaxPool →
Flatten → Gemm with named `features` input and `logits`/`pooled` outputs, no
intermediate value_info and multiple outputs. Independent expectations match
ORT for batches 1 and 3; the ONNX evaluator differs. Its schema-9 wrapper applies
momentum to inference statistics. Zero-variance/zero-epsilon additionally differs
from ORT's three NaNs. The fixture guide lists ceil trimming, nonfinite/no-sample,
SAME/dilation and empty-value differences separately.

The inherited 96-case conformance set now executes all retained CNN observations:
ceil average is an actual agreement; max NaN and two normalization cases are
executed differences. Current counts are 87 agreements, one empty-Softmax
extension, four differences (including Gather) and four unsupported models
(three unchanged upstream IR-7 models plus the opset-18 reduction). The original
1,422-model upstream envelope audit remains unchanged.

| Artifact | SHA-256 |
| --- | --- |
| CNN model | `eb3189e921756a9138588cc1247c4563faa8745418220f56a457a21bd2056796` |
| M3 provenance | `be7505137619c7ea8402eb3573bf678788b7c97b88c2ac1c6cfe12b16667ecfb` |
| Updated conformance provenance | `bd1b7d7d8659ef7289207d33772d0651226f5df7cc5b6639a4800a404e69849e` |
| Unchanged M1 provenance | `19227f9920883b065788cc8d1a2acdfa803c632a8bc0b4bc24ff720157d315ce` |
| Unchanged M2 provenance | `4bd43bb170feb5427a4f596ba4a171ff1da03cd251558e80e22adb1121fba59f` |
| Unchanged MLP/branch/shape provenance | `547933bf4e3be8e5e57e537820a1529822fcf3ddda17d953a83e98eca887ff2c` |

Every named input/output hash appears in M3 provenance. Model/value bytes for
M1, M2 and the three prior inference models are unchanged. Python 3.9.6,
ONNX 1.16.2 source `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, NumPy 1.26.4,
protobuf 4.25.3 and ORT 1.19.2 reproduce the fixtures with `tools/check_m3.py`.
These are development tools; ordinary Go tests remain offline and Python-free.
The repository has no LICENSE; no license was invented for original fixtures.

## Validation

Passed formatting, tidy-diff, module verification, build, vet, normal tests,
CGO_ENABLED=0, CGO_ENABLED=0 purego and race tests with
`GOCACHE=/private/tmp/onnx-runtime-gocache` and cached offline Go dependencies.
Tests cover isolated reference cases and rejection mutations, independent
concurrent sessions, 1 → 3 → 1 composed execution, output selection/ownership,
exact zero-scratch destination/publication live bounds, pre-allocation overflow,
all cancellation boundaries and recovery. Reshape-fed unknown ranks/shapes test
dynamic weight/bias/kernel/channel failures, changing SAME geometry and recovery.
All inherited profile, M1/M2, conformance and inference generators reproduce.

Reproduce public CNN inference from the repository root:

```sh
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph cnn -batch 3
CGO_ENABLED=0 go run -tags=purego ./examples/inference -graph cnn -batch 3
```

External data, further robustness/fuzzing, broad profiles/types/layouts,
MaxPool indices, training normalization, reductions and optimization remain
separate tasks. RT-14 is next. No benchmark improvement or universal ONNX
compatibility is claimed.
