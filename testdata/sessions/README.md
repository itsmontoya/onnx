# RT-08 session fixtures

These are original tiny models for lifecycle and serial execution tests, generated
with ONNX 1.16.2's `helper.make_graph`, `make_tensor`, `make_node`, and `make_model`.
Each uses producer `itsmontoya/onnx RT-08`, IR 8, and exactly one empty-domain
opset-13 import, then passes `onnx.checker.check_model(model, full_check=True)`.
The models use deterministic protobuf serialization. The pinned versions, original
fixture license, byte lengths and SHA-256 hashes are in [provenance.json](provenance.json).
Ordinary Go tests read these checked-in files without Python or native libraries.

| Model | Graph and hand-calculated behavior |
| --- | --- |
| [branch_defaults.onnx](branch_defaults.onnx) | `x: float32[N,2]` fans out through `middle → a` and `b`; graph input `weight: float32[W,2]` defaults to `[7,11]` at `[1,2]` and feeds `weight_out`. A tensor-valued Constant `[16777217, -9223372036854775808]` feeds an Identity `integer_out`. Outputs are declared `b, a, weight_out, integer_out, x`. All Identity values equal their inputs exactly. Overrides may change W independently of N. |
| [passthrough.onnx](passthrough.onnx) | No nodes; scalar int64 input `x` and float32 initializer `stored: [2] = [3,-0.0]` are returned in order `stored, x`. `stored` is not a graph input and cannot be overridden. |
| [empty.onnx](empty.onnx) | Named graph with no nodes, inputs, initializers or outputs. Run succeeds with no results. |

The explicit expectations and metadata variations live in
[session_test.go](../../session_test.go) and [model_test.go](../../model_test.go).
They cover default overrides, fresh shared symbols, bit-preserving float copies,
exact integers, unknown/scalar/empty shapes, output subsets, races, cancellation,
and resource boundaries. The suite also executes the existing
[profile Identity fixture](../profile/identity.onnx).
These models establish only the implemented Identity/dense Constant subset;
RT-10 owns the broader differential conformance harness.
