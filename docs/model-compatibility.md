# Audited model compatibility report

RT-19 selects **float32 LayerNormalization-17**, enabling nineteen unchanged
Apache-2.0 normalization exports at IR 8/import 17 with one additional operator.
This is a bounded normalization workload, not a trained network or general
transformer compatibility claim. The byte-original source exports are actual
ONNX backend test models; no upstream model is converted or relabeled.

## Corpus and license audit

The pinned ONNX v1.16.2 archive at revision
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557` has SHA-256
`d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5`.
Its1422 backend models contain no exact IR 8/import 13 model and 65 exact
IR 8/import 17 models. RT-19 copies 33 small models: 19 fused normalization models
and 14 representative negative workloads. Source model/value/source bytes,
versions, hashes, license, named types/shapes, schemas, attributes and external
requirements are retained in [the machine-readable index](../testdata/corpus/index.json).
None of these models requires external data or separately licensed learned weights.

These upstream artifacts are Apache-2.0. The byte-original
[LICENSE](../third_party/onnx/LICENSE) and original SPDX source header are
preserved; the pinned archive has no NOTICE file. No PyTorch/framework license
is used to license weights. The repository still has no project LICENSE;
original test-only helper graphs remain separately labeled without a grant.

## Deterministic unsupported findings

The static audit reports all selected models' envelope/value-kind/dtype/operator/
attribute/output-slot findings and failing node indices. It is intentionally a
selected-corpus audit, not a public replacement for runtime validation. Ordinary
Go tests assert actual loading and mandatory rejections separately. Models can
have multiple failures; model counts and finding counts are distinct.

| State | Enabled models | Unsupported models | Findings | Operator | Profile | Dtype | Value kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Before RT-19 |0|33|63|43|12|5|3|
| After RT-19 |19|14|44|24|12|5|3|

Each enabled model previously had one failing node 0, LayerNormalization. Each
now executes all three named Y/Mean/InvStdDev outputs. The preserved negatives:

| Original model | Actual IR/import | Findings | Failing operator node indices |
| --- | --- | --- | --- |
| [test_layer_normalization_default_axis_expanded](../testdata/corpus/models/test_layer_normalization_default_axis_expanded.onnx) | 8/17 | 11 | 1, 3, 6, 7, 8, 11, 12, 14, 18, 21, 27 |
| [test_slice](../testdata/corpus/models/test_slice.onnx) | 7/13 | 2 | 0 |
| [test_expand_dim_changed](../testdata/corpus/models/test_expand_dim_changed.onnx) | 7/13 | 2 | 0 |
| [test_constant_pad](../testdata/corpus/models/test_constant_pad.onnx) | 10/21 | 2 | 0 |
| [test_resize_downsample_scales_nearest](../testdata/corpus/models/test_resize_downsample_scales_nearest.onnx) | 9/19 | 2 | 0 |
| [test_where_example](../testdata/corpus/models/test_where_example.onnx) | 8/16 | 3 | 0 |
| [test_cast_FLOAT_to_FLOAT16](../testdata/corpus/models/test_cast_FLOAT_to_FLOAT16.onnx) | 10/21 | 3 | 0 |
| [test_simple_rnn_defaults](../testdata/corpus/models/test_simple_rnn_defaults.onnx) | 7/14 | 2 | 0 |
| [test_gru_defaults](../testdata/corpus/models/test_gru_defaults.onnx) | 7/14 | 2 | 0 |
| [test_lstm_defaults](../testdata/corpus/models/test_lstm_defaults.onnx) | 7/14 | 2 | 0 |
| [test_quantizelinear](../testdata/corpus/models/test_quantizelinear.onnx) | 10/21 | 4 | 0 |
| [test_if](../testdata/corpus/models/test_if.onnx) | 6/11 | 3 | 0 |
| [test_sequence_map_identity_1_sequence](../testdata/corpus/models/test_sequence_map_identity_1_sequence.onnx) | 8/17 | 3 | 0 |
| [test_optional_get_element_optional_tensor](../testdata/corpus/models/test_optional_get_element_optional_tensor.onnx) | 8/18 | 3 | 0 |

A profile failure blocks the original model even after adding its missing
operator. Future work must audit an additional profile or select another actual
export; it must preserve this fixture's original bytes. Expanded normalization
needs Cast/Size/Slice/Neg/ConstantOfShape/ReduceMean/Sqrt/Reciprocal and stays
unsupported. Backend ReduceMean availability does not establish ONNX reduction
schema 13/17 support. No bool/int32/low-precision/recurrent/quantized/control-flow/
structured-value production feature is added.

## Selected implementation and numerical evidence

The [independent selected task](tasks/layer-normalization.md) defines exact
schema 17 scope, input/output/attribute rules, enabling fixture, library revision
and acceptance. Import13 has no LayerNormalization schema and remains rejected.
Float32 X/Scale/optional B use full-X unidirectional broadcasting; rank must be
positive, normalized suffix dimensions nonempty, stash_type=1, epsilon finite and
nonnegative. Mean and InvStdDev retain unit reduced dimensions. Optional slots
are positional and every named output is implemented.

Published neuralnetwork v0.7.0 at
`6048452c57a0a293db4591cd2e2febd7d6b743a3` already supplies NN-12's required
ReduceMeanInto/SqrtInto plus arithmetic. No library task, dependency bump,
backend edit, local replacement or second numerical engine is required.
The adapter follows centered population variance:
`mean=mean(X); D=X-mean; variance=mean(D*D); inv=1/sqrt(variance+epsilon); Y=(D*inv)*Scale+B`.
Internal reduction axes are always explicit, nonempty suffix axes with kept
dimensions. This bypasses no ONNX schema adapter: no primitive ReduceMean,
ReduceSum/Max, Exp/Log/Sqrt/Pow, functions or new import record is advertised.

All nineteen source goldens pass an independent NumPy float32 centered-variance
cross-check and both pinned oracles. Nine original test graphs add a residual
DAG at batches 1/3, bias omission, optional Mean hole, output subset/order,
axis 0, scalar broadcasting, empty prefix and epsilon/nonfinite cases.
All 28 executed cases agree with ONNX 1.16.2;27 agree with ORT 1.19.2. ORT actually
rejects scalar Scale broadcasting even though the schema/evaluator accept it.
[Provenance](../testdata/normalization/provenance.json) retains each actual return
or error separately; no skip is represented as agreement. Float comparison is
`1e-5 + 1e-5*abs(expected)` with exact type/shape and NaN/infinity classification;
signed-zero equality is not promised for composite reductions.

The adapter reserves all destination tensors independently and all numerical
workspace before allocating. It keeps disjoint Mean/D/DD/variance/epsilon-sum/
stddev/inverse/normalized/scaled/Y buffers and two scalar tensors and their staging payloads. Backend
ReduceMean/Sqrt are direct loops without numeric accessor staging. Matching-shape
Sub/Mul/Add can allocate up to four additional payload copies per call. The
conservative workspace `20*X_bytes + 16*statistics_bytes + 16` covers retained
temporaries and copying paths, including unit normalized suffixes; it is not a
measured peak. Resident inputs/initializers, each of Y/Mean/InvStdDev, statistics
planning snapshots and requested output copies retain their separate bounds.
Even omitted stats slots conservatively reserve internal destinations.
Cancellation brackets synchronous subkernels; no in-kernel latency promise is made.

## Bounded next initiatives

These are separate handoff-ready task files with named failing models, exact
contract decisions, profile/dependency blockers, required backend work and
acceptance fixtures. They remain future gaps; RT-19 implements only the selected
normalization set.

| Initiative | Independent task |
| --- | --- |
| Slice | [Feature cropping](tasks/slice.md) |
| Expand | [Broadcast grids](tasks/expand.md) |
| Pad | [Constant-border preprocessing](tasks/pad.md) |
| Resize | [Nearest image downsampling](tasks/resize.md) |
| Bool/int32, comparisons/Where | [Typed selection](tasks/typed-selection.md) |
| Float16/bfloat16 | [Raw-bit value and conversion design](tasks/low-precision.md) |
| RNN | [Sequence/state contract first](tasks/rnn.md) |
| GRU | [Gate/reset mapping after RNN](tasks/gru.md) |
| LSTM | [Gates and cell state after RNN](tasks/lstm.md) |
| Quantization | [Rounding/saturation/per-axis conversion](tasks/quantization.md) |
| Functions/control flow | [Nested scopes and bounded execution](tasks/functions-control-flow.md) |
| Sequence/map/optional/sparse | [Explicit value-model design](tasks/structured-values.md) |

Map/sparse, bfloat16, function and bounded-Loop increments require further named
licensed models before selection. Their absence is an explicit acceptance blocker
in the appropriate task file, rather than an implicit support claim.

Reproduction is in [the corpus guide](../testdata/corpus/README.md); ordinary Go
tests remain offline and Python-free. [The RT-19 handoff](handoff-rt19.md)
records exact revisions and local/hosted checks. RT-16 remains unchecked until
published NN-13 results satisfy its independent acceptance conditions.
