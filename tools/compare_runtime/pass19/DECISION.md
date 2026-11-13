# Pass 19 production selection gate

Recorded after the fresh merged-runtime investigation and before editing
production source. **No NeuralNetwork production change is required.**

Select only evaluation-local argument storage. Keep destination retention,
validation, publication, cancellation, dispatch, and accounting unchanged.

1. **Unnecessary work:** allocating a new pointer-array backing store for each
   synchronous node invocation. MNIST allocates 160 B in twelve objects.
2. **Why unnecessary:** a call finishes before the next node constructs its
   arguments. Prepared descriptors copy the slice into a temporary call, and
   dynamic handlers do not retain arguments beyond synchronous execution.
   Neither the shared plan nor any returned output owns these arrays.
3. **Component cost:** fresh argument construction is 0.2411 µs, including
   argument binding and the existing per-argument context checks. That is an
   upper bound on recovery; binding/checks must remain.
4. **Complete-Run indication:** broad diagnostic control 49.304 → 49.024 µs;
   focused ordinary ramp 49.124 → 48.851 µs. Focused unchanged fixed-policy
   report medians 51.6455 → 51.125 µs, difference 0.5205 µs. Fixed-policy ranges
   overlap and CVs are 1.94%/1.98%; do not promise that entire difference.
   All repetitions, including the 63.086-µs broad control, remain in evidence.
5. **Bytes:** 160 B removed, evaluation size-class grows by 16 B in the measured
   overlay; net warm 1,409 → 1,265 B/op, a 144-B reduction.
6. **Objects:** twelve removed, warm 36 → 24 allocs/op. No new heap object.
7. **Contract:** nil optional slots, actual argument ordering/length, dynamic
   signatures, arbitrary supported arity, cancellation check positions, lifetime,
   errors, owned outputs, limits, Busy, and independent evaluations/Sessions.
8. **Smallest fix:** an inline three-pointer array in each evaluation; clear its
   current slice before filling it. Allocate an ordinary fresh slice for larger
   arities. Use the same helper in ordinary execution and normalization.
   Do not reuse the evaluation or any state between Runs.
9. **Owner:** ONNX `evaluation.go`; no module or backend edits.
10. **Expected recovery:** approximately 0.1–0.3 µs of complete warm execution,
    subject to fresh production comparison and correctness/Metal checks. Twelve
    fewer allocations are structurally demonstrated; latency is less certain.

Retention compatibility bypass indicates 0.473 µs in the noisy broad matrix,
but removes a required guard. The corrected lookup component and immutable
metadata proof deserve future investigation; this pass does not replace that
guard. Removing value validation indicates 1.793 µs but also removes binding,
declaration, symbolic, rank and limit obligations. It is not a valid fix.
Removing ONNX context checks indicates 0.388 µs and weakens cancellation.
Publication is only 0.1163 µs as a component and must own the output.

Backend MatMul planning is 0.0622 µs, with arithmetic dominating its 1.690-µs
complete call; the staging ceiling does not establish error-atomicity equivalence.
Pool validation is about 0.10 µs per call and includes required validation.
The final Add adapter ceiling is about 0.09 µs and belongs in NeuralNetwork.
None demonstrates a larger, contract-preserving complete-Run opportunity than
the selected ONNX mechanism. No backend implementation or handoff is selected.
