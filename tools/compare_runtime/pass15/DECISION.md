# Evidence before production selection

Baseline ONNX `752a8ac`, neuralnetwork v0.10.8 (`be95e90`), Go 1.27.2,
darwin/arm64. These are initial diagnostics, not the final paired estimator.

1. **22 allocations:** each channel Add creates three output-rank broadcast
   shapes, two input-rank contiguous strides and two output-rank broadcast
   strides: 216 B / 7 objects each. Coordinates do not escape. The equal-shape
   Add creates a 16-byte shape, three 48-byte Matrix objects and four 48-byte
   float32 buffers: 352 B / 8 objects. Profile counts confirm every site.
2. **Unnecessary objects:** all 15 shape/stride objects can use invocation-owned
   temporary storage. The seven matrix objects/buffers are also adapter overhead,
   but contain values or adapt values; removing them is a separate kernel/API
   decision, not static broadcast preparation.
3. **Model invariant:** all operand/output shapes, rank alignment, span geometry
   and source strides for these fixed MNIST nodes. The output shape/byte/workspace
   charge is already prepared by Pass 12.
4. **Run dependent:** dynamic/symbolic signatures, override shape validity,
   destination identity/shape/storage, context and mutable coordinates/offsets.
   Arithmetic depends on actual values. No session-only broadcast metadata exists.
5. **Planning:** initial medians 256.6 + 260.6 + 47.01 ns. These include shape
   validation/allocation and stride construction, not an isolated allocator cost.
6. **Arithmetic:** unchanged span loops 1,636 + 921.2 ns; diagnostic direct
   equal-shape scalar loop 6.61 ns. The last deliberately omits the matrix adapter.
   Input validation costs 27.79 + 28.51 + 22.86 ns; destination validation
   20.07 + 20.87 + 16.29 ns. These scopes overlap, so do not add them as savings.
7. **Latency ceiling:** roughly 0.56 µs of measured current planning across the
   three calls, with validation still required. The static prepared diagnostic
   saves ~0.52 µs on the channel pair. Its equal-shape result is a separate
   copy-removal ceiling. Ownership prototype saves ~0.18 µs across all three.
   Small differences require complete-runtime confirmation.
8. **Allocation ceiling:** 448 B / 15 objects for metadata without changing
   arithmetic; 784 B / 22 objects would also require removing matrix adapters.
9. **Boundary:** neuralnetwork owns the repeated allocations. Its public
   BroadcastShape must return owned storage, but internal consumers only need
   temporary metadata. A public prepared Add descriptor would add a signature,
   concurrency and memory contract for a small additional timing indication.
10. **Smallest candidate:** internal buffer-accepting shape/stride helpers,
    stack storage for ranks up to eight and heap fallback for arbitrary ranks.
    Preserve public owned shapes, validation order/errors and matrix dispatch.
    Build broadcast strides directly rather than creating/discarding contiguous
    strides. Copy error-message shape arguments only in the failing branch so
    success-path scratch does not escape. No persistent state or new public API.

The ownership prototype measures **0 / 0 / 336 B** and **0 / 0 / 7 objects**.
This selects a production candidate, conditional on semantic tests and whole-Run
allocation transfer. No static preparation is required: it would remove dynamic
work too, but add retained metadata and API surface for a small latency ceiling.
Repeated validation remains deliberate at the general backend API boundary.
