# Selected RT-19 set: float32 LayerNormalization-17

Status: **implemented by RT-19**. This independent task records the selected
contract; [the RT-19 handoff](../handoff-rt19.md) records completion evidence.

## Enabling model and failure

The byte-original Apache-2.0
[`test_layer_normalization_default_axis.onnx`](../../testdata/corpus/models/test_layer_normalization_default_axis.onnx)
is IR 8/import 17, with three float32 inputs and three named outputs. Before
RT-19 its node 0 fails registry resolution for LayerNormalization. All nineteen
upstream 2D/3D/4D normalization exports are enabled without editing version or
model bytes. Their source export and exact hashes are in
[the corpus index](../../testdata/corpus/index.json).

## Exact contract and dependencies

RT-17 supplies the accepted IR 8/import 17 envelope. ONNX LayerNormalization schema
17 accepts rank >=1 float32 X, Scale and optional B, float32 Mean/InvStdDev,
stash_type=1, axis=-1 by default and finite nonnegative float32 epsilon=1e-5.
Axis normalizes against rank and reduces the entire trailing suffix with kept
unit dimensions; normalized dimensions must be nonempty. Scale/B use
unidirectional broadcasting to the full X. Optional input/output holes retain
position; all named outputs execute and have independently checked metadata.
Import13, lower precision, other stash types and arbitrary local functions remain
unsupported. LayerNormalization is the fused standard operator, even though
its ONNX schema also describes a function body.

NN-12 is already published in neuralnetwork v0.7.0 at
`6048452c57a0a293db4591cd2e2febd7d6b743a3`. Its ReduceMeanInto/SqrtInto and the
existing arithmetic kernels implement centered population variance followed by
normalization/affine scaling. Internal axes are explicit, nonempty trailing axes
and keepdims=true; the runtime introduces no ONNX ReduceMean adapter, no
attribute-versus-input axes ambiguity, no noop_with_empty_axes behavior and no
new primitive reduction/math capability record. No further library task or
backend change is needed for this bounded set.

## Acceptance evidence

The 19 unchanged upstream models and original isolated cases exercise axis
signs/zero/default, epsilon/default/zero, no bias, scalar broadcasting, empty
prefix, constant/nonfinite inputs and all three outputs. A residual Add →
LayerNormalization → branched Mul/Add DAG runs at batches 1/3. Independent NumPy
centered-variance expectations are checked separately from both oracles. ORT's
scalar-broadcast rejection stays explicit, with schema/evaluator agreement.

Load checks known ranks/shapes/attributes, Run defers dynamic checks and recovers.
All numerical storage is reserved before allocating. Workspace is the conservative
`20*X_bytes + 16*statistics_bytes + 16`, not a measured peak. Each of the three
destination tensors is independently limited/reserved even if an optional slot
is omitted. Inputs stay read-only, published results own copies, cancellation
brackets synchronous kernels and independent sessions remain concurrent.

Completed tests and exact revision/check results belong in the handoff; this
selection does not imply general transformer or expanded primitive-graph support.
