# Two exact version profiles

RT-17 adds **IR 8 / empty-domain opset 17** to the original **IR 8 / empty-domain
opset 13** profile. Imports 14–16, 18 and other untested versions remain errors.
Dense float32/int64, finite limits, CPU serial scheduling and all existing
ownership/recovery rules remain the same. No new operator family is added.

[The schema audit](schema_audit.json) records every one of the 25 operators at
both imports using ONNX 1.16.2, source commit
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`. It includes actual applicable
schema versions, arities, attribute required/default values, type constraints,
standard-function flags, documentation hashes and the bounded runtime decision.
The nine changed selections are Identity-16, Add/Sub/Mul/Div-14, Relu-14,
Reshape-14, Shape-15 and BatchNormalization-15. Sixteen operators keep the same
applicable schema, with disjoint tested-import records. The manifest and its
Go-generated table do not advertise the gap as tested.

Identity/arithmetic/Relu type extensions and standard-function definitions do
not expand supported values: Identity still accepts only dense float32/int64;
arithmetic and Relu still require float32. At import 17 Reshape's allowzero=0
copies input dimensions, while 1 keeps literal zeros. Any literal zero with -1
is invalid. Shape's start/end use negative indexing and clamp to the input rank;
start >= end returns an empty vector. BN-15 accepts only training_mode=0 and
exactly one output; all five inputs remain float32 and stored statistics remain
unchanged. Old imports reject new attributes even at their default values.

## Actual exporter selection

The demanded target is inference of a dynamic-batch float32 MLP exported through
[PyTorch 2.2.2's torch.onnx.export](https://github.com/pytorch/pytorch/tree/v2.2.2/torch/onnx)
at source commit `39901f229520a5256505ec24782f716ee7ddc843`,
with requested opset 17. The original module is defined in
[tools/check_profiles.py](../../tools/check_profiles.py); all its weights are
explicit literals. The exporter emits IR 8 and the single empty-domain opset-17
import without conversion. [pytorch_mlp.onnx](exports/pytorch_mlp.onnx) stores
those emitted bytes unchanged, with SHA-256 in [provenance](provenance.json).
No prior export's version field is rewritten. The graph is
`Gemm → Relu → Gemm → Softmax`, with named float32 `features` and `probabilities`,
each shaped `[batch,2]`.

This is one tested original module, not a model-corpus survey or universal
PyTorch compatibility claim. The optional exporter is BSD-3-Clause licensed;
no exporter implementation or third-party model/weights are copied. This
repository has no LICENSE and the original module/fixtures have no separately
declared license. An exporter tool license does not license these original
artifacts. No Go runtime dependency or backend version changes.

## Offline cases and observations

The 67 cases include equivalent isolated graphs for all 25 operators at both
imports, exact int64 extrema/16777217 and empty/special float identities,
Reshape zero modes, Shape slices including int64 boundary values, stored-statistic
BN in both schemas and the actual export at batches 1 and 3. All 67 agree with
ORT 1.19.2. ONNX 1.16.2's evaluator agrees on 66 and differs on the BN-9 case
because it applies momentum to stored statistics; its BN-15 evaluator agrees.
Actual oracle outputs retain float32 bits, int64 values, shapes and types.
Independent NumPy/literal expectations and PyTorch eager outputs are recorded
separately from these observations. The inherited exception policies remain.

Normal Go tests hash every model/value/schema-audit artifact, check exact
applicable schema provenance, run named inputs through public APIs and compare
owned results after session/model closure. Int64, shapes and dtypes compare
exactly. Floats use `1e-5 + 1e-5*abs(expected)`, matching NaN classification,
infinity signs and signed zeros. New tests also cover version gaps, training,
unsupported types/functions, optional output slots, dynamic failures/recovery,
workspace/live budgets, cancellation and concurrent sessions. No Python runs.
All old fixture bytes and hashes remain unchanged.

## Reproduction

Use the existing pinned Python 3.9.6 environment with ONNX 1.16.2, NumPy 1.26.4,
protobuf 4.25.3 and ONNX Runtime 1.19.2. Install the additional optional exporter
outside the checkout, then run from the repository root:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python -m pip install --target /private/tmp/onnx-rt17-exporter torch==2.2.2
PYTHONPATH=/private/tmp/onnx-rt17-exporter /private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_profiles.py
CGO_ENABLED=0 go run ./tools/capabilitiesgen
CGO_ENABLED=0 go run ./examples/inference -graph exported -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph exported -batch 3
```

The audited exporter wheel was `torch-2.2.2-cp39-none-macosx_11_0_arm64.whl`.
Its installed support packages were filelock 3.19.1, fsspec 2025.10.0, Jinja2
3.1.6, MarkupSafe 3.0.4, mpmath 1.3.0, networkx 3.2.1, sympy 1.14.0 and
typing_extensions 4.16.0. The generator pins the exporter and oracle versions,
uses one Torch/ORT thread and disables ORT graph optimizations. Tool provisioning
needs network or preloaded packages; fixture execution needs only Go.

`--write` deliberately regenerates only this new fixture directory. No-write
mode rejects missing, extra or changed artifacts and any selected schema/default
or tool-version drift. It stores the raw exported buffer before serialization
and separately checks every model with ONNX's full checker. Regeneration is a
development operation; there is no runtime converter or evaluator.
