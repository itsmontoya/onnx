# M2 shape fixtures

RT-12 supplies 66 original checker-validated IR-8/standard-opset-13 models with
hashed named TensorProto inputs/outputs in [provenance.json](provenance.json).
The 60 evaluator agreements, two empty-prefix Flatten evaluator rejections and
four empty-Gather shape disagreements are classified separately. The latter
retain the actual evaluator `[0]` shape; independent NumPy `take` confirms the
schema's data-prefix + full-indices-shape + data-suffix expectation. Empty
Flatten uses explicit prefix/suffix products. These exceptions consume the
published [NN-07](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-07.md)
and [NN-08](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-08.md)
contracts and do not count as evaluator agreements.

Both float32 and int64 paths cover Shape, Flatten, Transpose, Reshape,
Squeeze/Unsqueeze, Gather and Concat. Data includes int64 extrema and values
above 2^53, float32 NaN payloads, signed zeros and infinities. Runtime tests
compare transform float bits and integer values exactly. Cases cover default
and explicit permutations, split endpoints/scalars, zero/-1 reshape, omitted
versus explicit-empty squeeze axes, output-rank unsqueeze axes, scalar/general/
negative/empty Gather indices and empty Concat. Static negative and dynamic
recovery tests complement these positive models without weakening rejections.

The independent expected values use NumPy shape/reshape/transpose/squeeze/
expand_dims/take/concatenate operations; the Go runtime does not generate its
own goldens. The generator records actual pinned ReferenceEvaluator outcomes
and checks exact shapes/values for agreements. Normal Go tests verify all
artifacts offline and invoke no Python, network or native runtime.

Regenerate with Python 3.9.6, ONNX 1.16.2 (source commit
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`), NumPy 1.26.4 and protobuf 4.25.3:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_m2.py --write
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_m2.py
```

See the project [LICENSE](../../LICENSE) for these original fixtures.
The [shape-heavy composed graph](../inference/README.md) and
[NN-13 handoff](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/handoff-nn13.md) reuse the existing public example
and fixture harness.
