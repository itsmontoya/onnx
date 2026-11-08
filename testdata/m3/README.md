# M3 CNN reference fixtures

[provenance.json](provenance.json) contains 87 original IR 8/opset 13 cases:
49 ONNX 1.16.2 evaluator agreements, 26 executed differences and 12 evaluator
rejections with independently expected outputs. All 87 execute in ordinary Go
tests. ORT 1.19.2 agrees on 66, differs on six and rejects 15. Actual shapes,
float32 bits and errors from both oracles are retained for every case. No
rejection or difference counts as unqualified agreement.

The isolated graphs cover Conv-11 groups, depthwise channel multipliers,
dilation, optional bias, dynamic weights, asymmetric pads, VALID and both SAME
modes; MaxPool-12 negative padding, dilation, empty optional output slot, ceil
geometry and no-sample windows; AveragePool-11 included/excluded divisors;
GlobalAveragePool-1 empty reductions; and BatchNormalization-9 rank 2/4,
nondefault/zero epsilon, momentum metadata, nonfinite domains and empty values.
Go mutation tests cover invalid attributes, ranks/channels, kernel/weight
mismatches, unsupported indices/storage order, training outputs and later-schema
attributes. Dynamic Reshape-fed CNN inputs exercise missing intermediate shapes,
changing spatial sizes/weights/bias, runtime failures and recovery.

The symbolic-batch [cnn.onnx](models/cnn.onnx) composes
Conv → BatchNormalization → Relu → MaxPool → Flatten → Gemm with no intermediate
value_info. Named input is `features:float32[batch,1,4,4]`; outputs are
`logits:float32[batch,3]` and `pooled:float32[batch,2,2,2]`. Batches 1 and 3
match an independent stored-statistics NumPy formula and ORT; the ONNX evaluator
differs. Tests reuse one session for 1 → 3 → 1, run independent sessions
concurrently, check requested output order, and compare retained results after
later runs, failures, cancellation and closure. The public example uses the
same input sequence, scaled by 1/8.

Expectations use independent NumPy padded-window/einsum convolution, indexed
pooling windows, and the literal float32 normalization expression
`scale*(x-mean)/sqrt(variance+epsilon)+bias`. Finite comparisons use
`atol=rtol=1e-5`, NaNs by classification, and infinity signs exactly. Zero tie
signs and NaN payloads have no cross-backend promise. Model and every named
TensorProto value have SHA-256 records verified by the existing Go harness.
There is no repository LICENSE; original fixtures have no separately declared
license. No upstream model is copied into this set.

Important retained oracle boundaries:

| Boundary | Executed behavior and actual observations |
| --- | --- |
| Conv automatic padding | Resolve each concrete axis using checked integers, including effective dilated kernels. The evaluator differs on selected SAME/VALID cases. ORT rejects SAME with dilation; the independent formula covers it. |
| Ceil average divisor | Include-pad uses the full kernel area even beyond declared padding: input 1..9, kernel/stride [2,2], ceil mode gives [3,2.25,3.75,2.25]. Both pinned oracles agree. |
| Ceil trimming | Remove starts at or beyond input+beginPad. For input [1,1,3,3], kernel/stride [3,3], pads [0,0,2,2], the runtime returns spatial [1,1]; ORT returns [2,2]. Average evaluator results also differ. |
| SAME pooling | Per-schema attribute text specifies ceil(input/stride); selected SAME_LOWER/ceil/dilated results differ from actual evaluator or ORT results. Their actual shapes/bits remain in provenance. |
| Padding/empties | Pinned schemas permit nonnegative pads without a kernel-size upper bound; ORT rejects pads at least the kernel, zero spatial sizes and some negative raw output geometries. The runtime clamps nonpositive extents to zero. Max with no samples is -Inf; average is NaN or zero depending on divisor; global empty average is NaN. |
| Nonfinite max | Runtime propagates input NaNs; the evaluator filters them. ORT uses the lowest finite float32 for a selected dilated no-sample window; runtime returns -Inf. |
| BatchNormalization | Preserve supplied statistics and accept valid momentum metadata without updates. The evaluator applies momentum. Literal zero-variance/zero-epsilon gives signed infinities/NaN where ORT gives three NaNs. Both composed CNN cases match ORT. |

Pinned tools: Python 3.9.6, ONNX 1.16.2 at
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, NumPy 1.26.4, protobuf 4.25.3,
and ORT 1.19.2. ORT uses CPUExecutionProvider, one intra/inter-op thread and
optimizations disabled. Regeneration is an optional development operation:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_m3.py
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_m3.py --write
```

The no-write command reproduces every byte and rejects unexpected artifacts.
Normal Go builds/tests need no Python, network or native runtime.

Primary sources are the pinned [schema definitions](https://github.com/onnx/onnx/blob/3bf92c03a9f27eba3bda1e5b9e63ea20ec213557/onnx/defs/nn/old.cc),
[convolution evaluator](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_conv.py),
[pooling evaluator](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_pool_common.py),
[normalization evaluator](https://github.com/onnx/onnx/blob/v1.16.2/onnx/reference/ops/op_batch_normalization.py),
and [ORT normalization](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/providers/cpu/nn/batch_norm.cc).
The [NN-09](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-09.md),
[NN-10](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-10.md),
and [NN-11](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-11.md)
handoffs describe the direct kernel contracts and their broader independent
backend fixture sets. Their counts are distinct from this runtime's 87 cases.
