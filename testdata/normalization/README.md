# LayerNormalization reference cases

[provenance.json](provenance.json) has **28 executed cases**: 19 unchanged
Apache-2.0 upstream models from [the corpus](../corpus/README.md) and nine
original, test-only helper graphs. Original graphs have no separate
redistribution license; the repository has no project LICENSE. The exporter
framework's license is not used to license original artifacts.

The 19 upstream cases agree with independent NumPy centered population variance,
ONNX 1.16.2's evaluator and ORT 1.19.2. They cover all positive/negative axes of
2D/3D/4D inputs, default axis, default/explicit epsilon and three named outputs.
The helper graphs cover a residual Add → normalization → branched Mul/Add DAG at
batches 1/3, no bias, positional omitted Mean, only Y, epsilon 0, scalar affine
broadcasting, empty prefix, axis 0, constant epsilon 0 and NaN/infinity inputs.

All **28** agree with the evaluator; **27** agree with ORT. The scalar-broadcast
case executes and is rejected by ORT 1.19.2 because it requires scale size equal
to the normalized suffix. The schema explicitly allows unidirectional
broadcasting to X; the evaluator agrees with the runtime. This actual rejection
is retained in the observations and does not count as ORT agreement.

Each oracle's actual float32 output bits/type/shape or error is retained
separately from expectations. Finite comparisons use `1e-5 + 1e-5*abs(expected)`;
type/shape are exact, NaNs compare by classification and infinities by sign.
Signed-zero equality is not promised for this composite reduction. No reference
skips are used. Empty normalized dimensions are outside the selected contract;
empty nonnormalized prefixes execute. Runtime tests cover dynamic rejection and
recovery, exact conservative reservations, cancellation between kernels,
independent concurrent sessions and owned outputs after closure.

[schema_audit.json](schema_audit.json) pins all schema-17 attributes/defaults and
full T/U constraints. Runtime scope is float32 only, stash_type=1 and finite
nonnegative epsilon. The adapter invokes released backend numerical kernels;
no ONNX primitive reduction/math adapter or arbitrary function executor is added.
Reproduction commands and source archive hash are in the corpus guide.
