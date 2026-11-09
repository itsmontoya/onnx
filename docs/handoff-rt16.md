# RT-16 storage measurement and reuse handoff

RT-16 is complete as a separately reviewable optimization follow-up from merged
`main` (`4d5a000c20ffae3e053190a4cd04322529851800`). The branch is
[`rt-16-live-tensor-storage`](https://github.com/itsmontoya/onnx/tree/rt-16-live-tensor-storage),
with PR base `main`; it is not merged or tagged by this task.

## Sources and dependency resolution

- `ab619af3501a6705820ee020aadf08160d2fb169`: pin the published NN-13 backend and extend the fixed-graph benchmark to CNN and Parse before changing runtime ownership.
- `6e58b352b529b73b8eb2daf520e31c7265d90ae3`: implement consumer/Identity groups, conservative private-storage reuse, scratch audit and focused tests.
- `fde7144bff7cb8b15b4e6f9a5e2a6c5a2e97abbe`: align embedded backend provenance; this is the exact after-measurement execution source.
- Later commits add validation tooling, a reused-destination zero-contraction test, raw evidence and this handoff. The final handoff revision is discoverable with `git log -1 --format=%H -- docs/handoff-rt16.md`.

The runtime consumes `github.com/itsmontoya/neuralnetwork
v0.7.1-0.20261109154142-a64fa687e269`, implementation
[`a64fa687e269c3a1bf3ac345434b46c8ed4a59fc`](https://github.com/itsmontoya/neuralnetwork/commit/a64fa687e269c3a1bf3ac345434b46c8ed4a59fc),
from [NN-13 PR #115](https://github.com/itsmontoya/neuralnetwork/pull/115).
Its [report/ownership contract](https://github.com/itsmontoya/neuralnetwork/blob/458b64d18a2e974a1bd51313d040060a85f1f70a/docs/performance/nn13/README.md)
meets the original dependency resume conditions. Source is publicly fetchable;
normal tests/timings use no local replacements or adjacent source imports.
The [November 8 blocker assessment](blocker-rt16.md) is retained with a resolution
note rather than rewriting its historical evidence. Neither optimization PR
needs a new tag or merge to provide an exact source dependency.

## Ownership and behavior

`valueLifetimes` records consumer occurrences, including duplicate inputs, and
groups Identity aliases into immutable ownership sets. With serial scheduling,
final-consumer positions can be precomputed. Every source, default/initializer
clone, Constant and declared output owner is pinned, including unrequested
outputs. Other groups retire after the final consumer's output checks and context
boundary; every alias slot is cleared together.

`runStorage` holds one call-local retired private tensor. A matching concrete
type/shape permits use as a disjoint destination. Replacement or a mismatch
releases the cached candidate; scratch/destination reservations trim it before
rejecting a limit. Cached numeric storage remains charged until dropped or
transferred. The cache is cleared before publication and cannot survive Run.
All public outputs still own independent clones. Cancellation, failure, changing
batches, defaults/overrides and separate sessions cannot mutate prior outputs or
shared constants. No parallel node scheduling, session-wide pool, packed weights,
constant folding or public API change is added.

Identity slots now share a reservation, so some previously limited calls succeed.
Separate input names remain conservatively charged even if callers supply the
same Tensor. The plan's additional bounded lifetime metadata can require a larger
MaxLoadBytes setting: exact-capacity arrays/flags plus construction work reserve
at most 41 bytes/value and 8 bytes/step, excluding Go headers/objects. Cached
buffers remain inside live limits; this is not a hard Go heap/RSS cap.

NN-13 CPU MatMul/Gemm scratch is one output-sized staging buffer (MatMul empty
contraction uses none). The Metal build retains the old conservative Matrix-copy
reservations. Other copying activation/arithmetic/normalization paths remain
explicitly reserved. No numerical operation or supported ONNX profile changed.

## Measurements and tradeoffs

The [full report](performance/rt16/README.md) publishes all sources, environment,
fixture/input/output hashes, 48 ten-sample timing rows, raw logs, actual executed
numeric copies, allocated payload, function-scoped payload bounds, explicit
reservation/workspace high-water and allocation-profile cross-checks.

Both sides already use NN-13, so its earlier backend gains are not attributed to
runtime reuse. Fixed MLP/CNN/shape graphs run batches 1 and 3. Parse, Load,
First and Warm are separate, with fixture setup/comparison outside timing.

- CNN First/Warm eliminates four allocations; Warm B/op drops 9288→9064 and 11880→11400 at batches 1/3. One destination is reused per call.
- CNN newly allocated numeric payload falls 1464→1336 /4032→3648 bytes; Warm function-scoped peaks fall 1236→1108 /2988→2604. Reservation peaks fall 1576→1448 /4008→3624.
- MLP/shape allocation counts and numeric copies remain unchanged, with +32 B/run evaluation-header overhead. Shape reservation peaks fall 328→216 /712→456 while its conservative function-scoped payload bounds remain unchanged. Those are distinct metrics.
- Default Warm medians rise 1.6–6.0%; purego medians rise 2.3–5.3% on this host. The explicit retention objective is lower bounded CNN allocation/storage, without a speedup or universal zero-allocation claim.
- Rate-1 CNN batch-3 profiles show about 512 B/run fewer tensor allocations and 32 B/run extra evaluator/table storage, matching the measured net 480 B/run reduction.

The observer is a development-only temporary source overlay; it never patches
production or module-cache source. Its function-scoped bound may retain locals
beyond compiler last use and excludes wire/metadata/Go objects/allocator overhead
and RSS. Explicit reservation hooks observe actual budget high-water separately.
All 48 before/after observations reproduced exactly. No absent metric is claimed
as zero; Parse's zero numeric counts do not imply zero protobuf memory/copying.

## Verification

On Go 1.27.2, macOS 26.5.2, Darwin/arm64 Apple M3:

- Formatting, module verification/tidiness, build, vet, generated capability-table check and diff whitespace passed.
- Normal, cgo-disabled, cgo-disabled purego and race suites passed against the public pseudo-version. Full `go test -tags=metal ./...` also passed with its retained copying bounds.
- Focused tests prove duplicate consumers, exact-capacity metadata, alias retirement, clearing every retired slot, source/output protection, replaced/mismatching candidates, dtype/empty-shape matching, trim-before-live/workspace limits, default overrides, branch/joins, output subsets/order and independently cloned aliases.
- Changing/empty batches, retained outputs after subsequent Run/Close, every cancellation boundary, late-failure recovery and independent concurrent reuse sessions pass. A zero-contraction MatMul clears a reused nonzero destination to positive zero.
- All four bounded fuzz targets passed: protobuf entrypoints (133880 executions), tensor conversion (147170), graph references (23548) and shape/attribute planning (107721). These short robustness runs are not exhaustive proofs.
- MLP/CNN/shape examples passed at batches 1 and 3 with cgo disabled. Existing offline typed goldens and all numerical/reference exceptions remain unchanged; no fixture bytes or support records were broadened.
- The PR's final hosted Linux push/pull-request jobs must be green before handoff; exact final head and run URLs are recorded in the verified PR/agent ledger.

## Remaining scope

All original RT-01–RT-19 tasks are now checked complete. Future operator/type,
version, GPU, recurrent/control-flow and broader corpus work stays in its separate
model-selected contracts. The runtime still implements its documented subset.
Project/source-fixture licensing and release tagging remain maintainer decisions;
this task preserves their existing checklist entries and publishes no release.
