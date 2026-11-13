# Offline conformance fixtures

RT-10 runs 96 classified cases in ordinary Go tests: **87 reference agreements**,
**one documented extension**, **four executed reference differences**, and
**four explicit unsupported-model rejections**. RT-12 enables the empty Gather
case with its independently expected shape. RT-13 executes the four retained CNN
cases: ceil averaging agrees with both oracles, while max NaN and normalization
retain actual differences.
No test skips contribute to coverage. The accepted IR 8/standard-domain opset 13 envelope remains unchanged.

[The provenance manifest](provenance.json) records each case's origin, actual
license status, IR/imports, selected operator schemas, named TensorProto inputs
and outputs, SHA-256 for every model/value, comparison policy and actual reference
observations. It reuses all 81 RT-09 [M1 models](../m1/provenance.json) without
changing their bytes or repeating their generation. Eighty agree with ONNX
1.16.2; empty-axis Softmax retains the evaluator's rejection as an extension.

Six original hand-calculated cases cover addition, a two-output branch, exact
int64 extrema and `16777217`, an empty int64 tensor, scalar negative zero, and
float NaN/infinity/zero identity. Inputs include both raw and typed TensorProto
encodings; expected outputs use typed fields. Reference checks independently
verify the literal expectations. The Go harness uses the runtime's shared
bounded TensorProto decoder; it never routes integer values through floats.
See the project [LICENSE](../../LICENSE) for these original fixtures.

## Pinned upstream selection

The official ONNX v1.16.2 source at commit
[`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`](https://github.com/onnx/onnx/tree/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557)
contains **1,422** backend `model.onnx` files. The complete
[envelope/hash audit](upstream_audit.json) finds **zero** with exactly IR 8 and
standard-domain opset 13. Its 352 opset-13 models use IR 7; IR 8 models import
later opsets. Altering version numbers would not establish compatibility.

Three upstream node cases (`test_softmax_axis_0`, `test_gemm_default_no_bias`,
`test_tanh`) retain their original model **and value bytes**. All use IR 7 and
opset 13; tests require an explicit unsupported-version error. They count as
rejection coverage, never successful runtime conformance. Original checker-
validated IR 8/opset 13 fixtures supply the positive numerical cases. This is
the available pinned suite's envelope boundary; accepting other IR versions
belongs to a later profile-expansion task.

Copied upstream artifacts are Apache-2.0 licensed. The required unmodified
[license](../../third_party/onnx/LICENSE) is included in the repository; the
pinned upstream source has no root NOTICE file. Each case links its upstream
directory and records file hashes. See the official
[backend testing documentation](https://onnx.ai/onnx/repo-docs/OnnxBackendTest.html)
for how backend cases represent selected tests rather than universal support.

## Comparison and failure reporting

Names, output order, dtype, rank, dimensions, element counts and int64 payloads
must match exactly. Validated tensor type and shape establish equal counts
before element comparison. Floats use
`abs(actual-expected) <= 1e-5 + 1e-5*abs(expected)` for finite values. A tolerance
is a fixture bound, not a universal numerical guarantee. NaNs compare by
classification without a payload/sign promise. Infinities must have matching
signs; an unexpected infinity or NaN always fails a finite expectation.

Both zero signs are exact when actual and expected are zero, except the
explicitly recorded nonempty MatMul/Gemm product policy. Empty MatMul
contractions require positive zeros. Small finite differences near zero still
use the finite tolerance. Identity's special-value case explicitly checks
negative zero. Integer extrema remain exact, independent of float tolerances.

The reusable runner loads the model through the public API, binds named
inputs, executes every output, compares results, closes both owners and
compares retained outputs again. Diagnostics include case/input names,
output name, flat mismatch offset and row-major coordinates, values/bits and
tolerances. `TestConformanceSuiteDetectsNumericError` injects `[4,7]` into the
real addition fixture expecting `[4,6]` and requires a diagnostic identifying
`inputs [left right]`, `output "y"`, `flat[1]` and `index [1]`.

## Executed reference differences and remaining reduction boundary

Gather and all four CNN models execute against independent expectations. Only
the reduction model in this table remains unsupported; the three upstream
IR-7 models above remain version rejections.

| Case | Independent expectation and retained reference behavior | Owner |
| --- | --- | --- |
| Empty Gather indices | Shape `[2,0,2]`; evaluator returns `[0]` | RT-12 implemented difference |
| AveragePool ceil window | Including padding uses full-kernel divisors; `[3,2.25,3.75,2.25]`; actual evaluator and ORT outputs retained | RT-13 executed agreement |
| MaxPool NaN | Independent output propagates NaN; evaluator filters it; ORT output retained | RT-13 implemented difference |
| BatchNormalization-9 | Stored statistics give `[2.5,4.5]`, agreeing with ORT; evaluator applies momentum and differs | RT-13 implemented difference |
| BatchNormalization zero domain | Literal formula gives `[-Inf,NaN,+Inf]`; ORT gives three NaNs; evaluator also retained | RT-13 implemented difference |
| NN-12 reduction boundary | Omitted ReduceMean axes express identity with opset 18 and `noop_with_empty_axes=1`, matching NN-12; no opset-13 reduction support implied | Later profile/operator task |

BN expectations follow `scale*(x-mean)/sqrt(variance+epsilon)+bias`, with the
supplied statistics unchanged. ORT 1.19.2 uses CPUExecutionProvider, one
intra/inter-op thread and disabled graph optimizations. These observations
preserve the distinctions in [NEEDS.md](../../NEEDS.md) and the published
v0.7.0 handoffs; they do not replace those broader backend fixture sets.

Primary pinned reference implementations are
[Gather](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_gather.py),
[pooling](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_pool_common.py),
[BatchNormalization](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_batch_normalization.py)
and [ORT normalization](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/providers/cpu/nn/batch_norm.cc).
Reference observations are outputs of these development tools, not copied
implementation code or an assertion that future adapters are implemented.

## Reproduction

Ordinary `go test ./...` needs only Go, cached modules and checked-in artifacts.
It invokes no Python, network or shared runtime. Optional fixture regeneration
requires Python 3.9.6, ONNX 1.16.2, NumPy 1.26.4, protobuf 4.25.3 and ONNX
Runtime 1.19.2. First download the pinned official archive:

```sh
curl -fsSL https://codeload.github.com/onnx/onnx/tar.gz/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557 -o /private/tmp/rt10-onnx-source.tar.gz
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_conformance.py --source-archive /private/tmp/rt10-onnx-source.tar.gz
```

Run commands from the repository root. `--write` deliberately regenerates;
without it, byte differences, extra/missing artifacts, version drift or archive
drift fail. The archive SHA-256 is
`d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5`.
Generation checks every selected model with ONNX's full checker and reproduces
the complete upstream audit. It verifies borrowed RT-09 models/expectations and
serializes all new values deterministically. Repeated generation must leave the
working tree unchanged.
