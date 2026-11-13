# Pass 18 ownership and production gate

Written before production implementation. The diagnostic uses Go overlays only;
its first revision and failed full-suite log remain in the evidence. Revision 2
passes the full CGO-disabled suite and focused race tests. Warm ramp medians are
55.072 µs control and 48.580 µs prototype (five observations each, including the
60.679-µs prototype observation). These are prototype measurements, not the final
fixed-policy production result. First-use payload allocation remains necessary.

## Existing contract and selected owner

`Session.Run` holds `mu.TryLock` for the complete synchronous evaluation and
publication. Overlap with Run or Close returns ErrBusy immediately; no new
serialization is introduced. Different Sessions share immutable plan metadata,
constants, prepared geometry and the prepared weight reshape, and execute
concurrently. Run values, symbols and budgets remain invocation-local. Model.Close
drops its reference and disallows new sessions; existing sessions remain valid.

Each Session owns one exact-shape tensor reference per prepared destination
creation slot. No mutable tensors enter the execution plan. Only wholly static
graphs composed of Identity, Constant, Conv, MaxPool, Add, Relu, MatMul and immutable
shape-target Reshape qualify. Every prepared output shape/type must match inferred
fixed Info. Unknown/symbolic inferred geometry, overridable reshape targets,
unsupported handlers, unavailable preparation, and insufficient headroom decline
retention for the entire graph. This deliberately narrow first implementation
covers the MNIST graph. Existing dynamic evaluation and intra-Run reuse remain.

The single retired candidate gets first refusal. A cache slot is borrowed only
when that path would otherwise construct a new tensor. Only fresh, successful,
validated results are captured. A borrowed/retired tensor is never captured in
another slot. Immutable prepared Reshape neither borrows nor captures. Output
cloning is unchanged, including graph outputs and Identity aliases.

## Accounting proof

Limits currently count logical payload/reservations, excluding Go headers, shape
objects, pointer tables, allocator rounding and outputs retained by earlier callers.
MaxLoadBytes bounds load work, snapshots/preparation and prepared payload;
residentBytes is immutable numerical payload used to seed each Run budget.
MaxLiveBytes bounds resident payload plus live invocation reservations and scratch;
MaxWorkspaceBytes separately bounds scratch. Input/default binding retains its
legacy logical reservations even when the immutable payload is already resident.
MaxTensorBytes applies to every tensor. MaxSessions bounds sessions sharing a plan.
None is a process RSS or allocator-byte quota.

Use the existing conservative `preparedLiveBudget`: subtract resident payload,
maximum permitted workspace, every value's fixed payload, maximum tensor bytes
for each graph input (including invalid bindings before declaration validation),
and all possible output clones from MaxLiveBytes. Require enough remainder for
ALL eligible static destination capacities, including slots normally satisfied
by intra-Run reuse. Decline LayerNormalization and every nonstatic value. Prepared
output geometry is additionally checked against the declared sizes used by this
bound. Thus no unbounded actual allocation can bypass the proof before validation.

Retained bytes are charged ON TOP OF unchanged invocation reservations. At entry,
add current retained bytes to the Run budget; after a successful fresh output,
reserve its additional residency BEFORE retaining its reference. All later
retirement releases only invocation charges. The residency never disappears
because the tensor becomes idle. This deliberately counts an active retained
tensor twice, like existing default/prepared constant reservations. The static
headroom proof makes that conservative charge harmless to supported Run outcomes,
including invalid inputs and original error precedence. No quota is raised.

A computed minimum-live floor prevents optional retention if an internal test
lowers the immutable plan's Run budget after loading. Revision 1 lacked this
admission check: exact-limit tests exposed publication failures after optional
capture. Revision 2 checks the floor before attaching storage; it discards any
references/counters when admission no longer holds. Public limits are immutable.

No Session payload is load work, so it is not falsely charged to MaxLoadBytes.
Its category is retained Session numerical payload, included in each owning
Session's MaxLiveBytes. Go metadata follows the existing documented exclusion.
Allocator capacity is not exposed by Tensor; logical capacity is ByteSize of the
exact immutable shape, and physical size classes are reported separately.

For MNIST: 83,664 B retained in nine tensors after first success; 91,449 allocator
bytes including nine payloads/headers/shapes/identities. The slot table adds 96 B.
The conservative capacity bound is 121,296 logical B / eleven potential tensors
and twelve table entries; usual ReLU reuse removes two allocations/retentions.
Across a plan, total Session retention is bounded by MaxSessions times this bound.
No shape history, replacement allocations, eviction policy or capacity resizing.
Zero-payload destinations are not retained. Tight models retain zero payload.

## Isolation and overwrite proof

Session mutex ownership spans acquisition, kernel execution, publication and
release of Run-local state. Slot references are immutable after insertion and
never escape as public outputs. Disjoint creation establishes distinct tensor
identities. Inductively, a borrowed slot first enters this Run at its original
creation position; intra-Run retirement can subsequently lend it only to later
nodes after all consumers/Identity aliases die. No slot can be acquired a second
time during that traversal, and fresh-only capture prevents duplicate ownership.
Other Sessions allocate their own slots and tensors. Plan lifetime pins inputs,
constants, graph outputs and immutable prepared results exactly as before.

Conv kernels assign every output element from local accumulators (including the
released scalar, channel, interior and residual/tail branches); no accumulation
starts from destination. Broadcast Add assigns every element; matching Add copies
a completed independent matrix result. MaxPool assigns each output from its own
window accumulator, including empty-window/NaN handling. Reshape copies the whole
validated contiguous payload. MatMul commits the complete independent staged
product, or explicitly clears a zero-contraction result. ReLU writes every lane;
ONNX retains the existing NaN clone/legacy correction and cancellation restoration.
Zero-sized outputs have no bytes to initialize. No preclear is needed.

Backend returned errors preserve destination except the explicitly cancellable
ReLU path; ONNX already restores its legacy result. Context errors can follow
complete synchronous writes. All errors return before value installation and
subsequent traversal/publication; cached storage can contain old/partial values,
but the next successful producer fully overwrites them. Failure never installs a
cached tensor into the next Run's new value table. Polling and clone boundaries
are unchanged. Cancellation does not release the Session mutex until synchronous
work returns. There is no separate Busy flag or outstanding lease to strand.

Close zeros references AND logical retained counters while holding the existing
mutex, before releasing the plan slot. Logical accounting does not await GC;
physical Go heap reclamation still does. Model.Close alone intentionally leaves
live Sessions' resources valid. Previously returned clones survive both closes.

## Candidate comparison

| Design | Retained bytes / objects | Concurrency | Dynamic geometry | Accounting | Expected benefit |
| --- | --- | --- | --- | --- | --- |
| Fixed exact static creation slots (selected) | Sum of static capacities, ≤ steps tensors; MNIST 83,664 B / 9 tensors usual, bound 121,296 B / 11 tensors | Existing per-Session Busy; Sessions independent | Original path, no retention | Conservative extra residency with complete static headroom proof | Captures nine MNIST allocations; no new resizing API |
| Bounded Session capacity cache | Explicit byte/object caps required; replacement transient must also fit | Same isolation possible | Replace/discard on incompatibility | Needs shape/type/layout and replacement accounting; Tensor has immutable shape and no resizing abstraction | Broader workloads; no demonstrated additional MNIST benefit |
| Reusable Run-storage instances | One per existing serial Session, plus bounded destinations | Same isolation possible | Must clear values/symbols/leases and evict incompatible payload | Retains additional invocation metadata and input/output reference hazards | Also saves small metadata allocations; exceeds this pass's scope |

## Fourteen-question gate

1. Retain exact backend tensor payload/header/shape/identity and a finite pointer table.
2. Session owns them; plan contains only immutable admission bounds.
3. At most one tensor per eligible static creation slot, bounded aggregate bytes.
4. Retained numerical payload is an additional MaxLiveBytes reservation; original
   live/workspace/tensor limits and load charges remain intact.
5. Session.Close clears references/counters synchronously; failed admission drops
   retention before Run; model Close follows existing Session-retains-plan contract.
6. Existing Session TryLock excludes overlap; each Session has independent objects.
7. No output publishes on cancellation; cached bytes remain charged and next
   successful producer overwrites every element; mutex always unwinds.
8. Returned errors follow the same rule; fresh failed/unvalidated results are not
   captured. Poisoned-slot and late handler-error recovery are tested.
9. Dynamic/unknown/symbolic inferred geometry or mutable targets disable retention;
   exact-shape compatibility rejects mismatches; no replacement or accumulation.
10. Close releases logical retention immediately, Busy still rejects active Close.
11. Cold execution still constructs nine tensors; revision-2 lifecycle benchmark
    includes Session creation/Close and separately records 1/2/10/100-call batches.
12. Warm prototype removes 91,417 net allocated B and 36 objects per call; observed
    Go-only median improvement 6.492 µs, subject to complete production validation.
13. About 91 KB of allocator storage per warm MNIST Session plus pointer metadata;
    it is a lifetime trade, not eliminated memory.
14. One Session storage type, conservative plan admission, existing destination
    selection and validated-result capture; no backend production/API change.

Decision: pass the prototype gate and implement this focused ONNX candidate.
Final acceptance remains conditional on production correctness, native Metal and
fresh unchanged fixed-policy measurements. No backend production work is needed.
