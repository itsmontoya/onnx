# Pass 20 production decision

Recorded after fresh baseline, profiles, full metadata inventory, accessor
instrumentation and seven unfiltered controlled diagnostic repetitions; before
editing production source.

**Select one focused ONNX optimization: extract the actual dtype once per
`evaluation.checkValue` invocation and reuse that scalar across declaration checks.**

**No NeuralNetwork production change is required.**

1. Recomputed metadata: the same backend tensor's validated dtype, once for each
   original declaration and once for inferred Info. MNIST has 34 such reads in
   21 checks: nine binding checks and twelve produced-value checks.
2. Reason: each independent `checkInfo` argument currently calls `DataType`, whose
   backend implementation repeats full shape/product/type/storage validation.
3. Immutability: no operation inside `checkValue` can change the actual tensor's
   dtype, shape, identity or storage lengths. Backend metadata fields are private;
   constructors own them, accessors copy shapes and Into operations write payloads.
4. Available proof: the first **actual tensor** dtype extraction within this
   invocation. Prepared Info supplies expected dtype, which cannot stand in for
   validation of the produced tensor. The local actual result can serve all
   original declarations and inferred Info.
5. Invalidating conditions: a different tensor, a future metadata-mutating API,
   or unsupported concurrent mutation during the synchronous validation call.
   No result is reused between values or Runs, including dynamic values/overrides.
6. Guards: every check still calls actual Rank, Shape and DataType, in the same
   order before checking declarations. All declaration/Info checks remain.
7. Remaining validation: actual tensor validation, rank/dimension limits, original
   dtype/rank/fixed/symbolic declarations, inferred Info, backend operator checks,
   retention checks, accounting, alias checks and every context boundary.
8. Removed validation: thirteen deterministic repetitions of DataType's backend
   Validate on the identical, unmodified object in the measured warm Run.
9. Equivalence: each declaration receives the same actual dtype it previously
   received. Check order, error kind/location, symbols and cancellation boundaries
   remain unchanged. Empty/scalar/dynamic/default/override values use fresh local
   metadata per check. No success is inferred solely from expected prepared Info.
10. Component recovery: produced-value checks 0.9100 -> 0.7901 us, **0.1199 us**,
    measured in separate seven-observation scopes. Binding is also affected.
11. Complete-Run diagnostic: control median 49.052 -> 48.907 us, **0.145 us**,
    CVs 0.1275% / 0.1883%; seven alternating/reversed-order observations, all kept.
    Final fixed-policy transfer still requires independent control/candidate pairs.
12. Allocation change: **zero**; all diagnostic samples remain 1,265 B / 24 objects.
13. Retained memory: **zero increase**; no fields, slices, tables or caches added.
14. Owner: ONNX `evaluation.go`; backend v0.10.10 remains unchanged.

The prepared-retention diagnostic reuses the existing call-descriptor table and
checks actual tensor compatibility on admission, requested compatibility at
lookup, and normal backend destination/alias validation every execution. It
reduces nine-lookup components 0.2977 -> 0.05233 us, but requires a new Session
reference plus an admission invariant. Its complete-Run median reduction is
0.132 us and CV 3.592%, with 53.348/52.224-us samples retained. Keep it diagnostic;
do not combine it with the selected local change.

Requested geometry is already prepared; an additional requested-geometry cache
would repeat existing state. An unconditional direct-table lookup is explicitly
invalid because it discards actual/requested compatibility without replacing
the proof. Its diagnostic movement is not equivalent recovery.

Conv/Pool/Reshape geometry allocation remains a backend-owned hypothesis.
No equivalent backend prepared-operation API or complete-contract recovery was
demonstrated. A backend implementation handoff is not justified by those helper
costs alone. No backend implementation, branch, release or dependency change is
selected, and no unrelated operator or allocation work is included.
