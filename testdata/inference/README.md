# RT-11 inference fixtures

Two original deterministic models use IR 8, the empty standard-domain opset 13,
inline float32 initializers, and `["batch", 2]` named I/O. `mlp.onnx` is
Gemm/Relu/Gemm/Softmax. `branch.onnx` extends its logits with Add/Softmax/Sub,
accepts `features` and `residual`, and returns `probabilities` and `delta`.
Each has actual ONNX reference goldens for batch sizes 1 and 3. No upstream
model is copied. No repository LICENSE is present; these original artifacts
have no separately declared license. Origin does not establish a license.

[provenance.json](provenance.json) records each model and named TensorProto
value's SHA-256, exact IR/import/schema versions, original source, license gap,
comparison policy and actual reference observation. Float32 values retain raw
bits in the TensorProto payload. All artifacts are used by normal offline Go
tests through RT-10's bounded typed-value decoder and hash/comparison helpers.

`tools/check_inference.py` requires Python 3.9.6, ONNX 1.16.2, NumPy 1.26.4 and
protobuf 4.25.3. The ONNX source revision is
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`. Run from the repository root in
that optional development environment:

```sh
python tools/check_inference.py
python tools/check_inference.py --write
python tools/check_inference.py
```

Without `--write`, generation verifies existing bytes and the getting-started
capability table without mutating them. `--write` deliberately regenerates
both models, all goldens/provenance, and that table. The generator enforces its
exact tool versions, validates both models with the full ONNX checker, records
actual `onnx.reference.ReferenceEvaluator` returns and independently checks
their composition using NumPy affine/ReLU/stable-softmax expressions. It does
not generate expected values through this Go runtime.

The root `TestInferenceFixtures` loads one model/session for both batches,
checks symbolic metadata, hashes/envelopes and golden outputs, reverses requested
output order by name, and rechecks retained outputs after later runs and closing
both owners. Runtime and reference floats compare with
`abs(got-want) <= 1e-5+1e-5*abs(want)`, exact types/shapes/names/counts,
NaN classification, infinity signs and signed zeros. These four agreements are
additional to RT-10's unchanged 86 agreements, one extension and nine rejections;
neither suite claims full ONNX profile coverage.

See [the runnable command](../../examples/inference/main.go),
[getting started](../../docs/getting-started.md) and
[the integration report](../../docs/integration-rt11.md).
