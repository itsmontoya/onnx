# Pass 17 production decision

Decision: **make no production change; investigate bounded ONNX destination reuse
in Pass 18**. No NeuralNetwork production change is required for this direction.
No backend fix was selected, so no `NEURALNETWORK_PROMPT.md` is generated.
The numerical gates do not justify another backend optimization pass now.

1. **Exact potentially unnecessary work:** constructing nine private destination
   tensors anew on every serial inference, including zeroing their payloads
   before operators overwrite them. Current intra-Run reuse already removes two
   ReLU destinations. Those two reuses must not be counted again.
2. **Measured cost:** unchanged diagnostic control 56.943 µs; preallocated ordinary
   destination topology 48.940 µs. Difference of medians: **8.003 µs**; paired
   median: 7.845 µs. A second controlled probe that disables operator bodies
   measures 10.089 → 3.210 µs, a 6.879-µs difference. These scopes overlap and
   have different working sets/GC behavior; they do not establish an 8-µs
   complete-contract production improvement.
3. **Bytes:** 91,449 B per Run of allocator-rounded destination storage,
   representing 83,664 logical payload bytes plus headers, shapes and identities.
4. **Objects:** 36 per Run: nine payloads, nine tensor headers, nine shapes,
   nine storage identities.
5. **Why potentially unnecessary:** serialized sessions could retain private
   destinations while still cloning public outputs. The diagnostic demonstrates
   the allocation mechanism; safe retention under current limits and lifecycle
   rules is not yet demonstrated.
6. **Contract:** returned outputs must remain independently owned; inputs and
   defaults remain immutable. Independent sessions share an immutable plan.
   Run/Close overlap returns Busy. Live/workspace accounting, cancellation,
   deadlines, failed-Run recovery and Close cannot change. Retained payloads
   must be charged and released, including unused shape variants.
7. **Smallest correct fix:** not established. A bounded session-owned workspace
   with dynamic-shape fallback is a hypothesis, not an approved design. Current
   diagnostics hold global references and reserve logical tensor lifetimes as
   if buffers were fresh; they are not a correct implementation of retained
   storage accounting or concurrency. Do not transplant them into production.
8. **Expected component recovery:** allocator/capacity reuse can remove up to
   the measured destination allocations in this static fixture. Practical
   retained-memory, initialization and fallback costs must be measured first.
9. **Expected complete-Run recovery:** unknown under the complete contract.
   The controlled 6.9–8.0-µs signals are ceilings/diagnostic indications, not
   a forecast, a lower bound, or additive recoverable savings.
10. **Owner:** ONNX owns session/evaluation/destination/lifetime/accounting policy
    (`session.go`, `evaluation.go`, `run_storage.go`, `run_budget.go`,
    `value_lifetimes.go`, `execution_plan.go`). No backend API is required just
    to investigate private destination retention. Any later discovery requiring
    backend production work must stop at the repository boundary and produce a
    standalone backend handoff.

The argument-buffer diagnostic removes twelve objects and 128 net B, but its
0.262-µs whole-Run difference is small against control movement, and adds almost
no measured benefit once destinations are preallocated. It is a valid smaller
future allocator objective, not evidence to distract from destination policy.
Run-local stack storage across generic escaping call constructors has not been
proved; the measured array is embedded in the heap-owned evaluation.

Publication measures 0.1166 µs, 193 B and six objects, with only a 40-byte final
payload copy. Its complete-Run ablations do not show a durable latency gain.
Public output survival must remain intact. The 1-KiB activation Reshape copy
and final 40-byte MatMul staging copy are much smaller than destination
allocation. MatMul's validated direct-product ceiling recovers about 0.051 µs
in its synthetic fixture, and Pool's validation costs about 0.10 µs each.
Neither gate justifies reopening kernels or crossing the backend boundary.

Pass 18 acceptance must include a complete-contract before/after comparison with
the unchanged fixed policy, explicit retained bytes/objects and initialization
cost, bounded behavior for changing shapes, quota boundary equivalence, output
survival, immutable defaults/overrides, aliases, cancellation/deadlines and failed
Run recovery, Busy/Close, independent concurrent sessions, race/purego/CGO-disabled
suites and supported cross-builds. Native Metal validation is required if the
chosen design affects lifetime/staging/retirement. Keep the historical anomaly
visible. Reject the design if the demonstrated benefit depends on uncharged
retained memory or a weakened contract.
