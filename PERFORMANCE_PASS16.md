# Performance Pass 16 — Repository Boundary and Backend Handoff

Published integration is now complete with NeuralNetwork **v0.10.10**. See
[the release validation](PERFORMANCE_PASS16_RELEASE.md) for fresh transfer results.
The report below preserves the original investigation and repository handoff.

The measured optimization belongs in **NeuralNetwork**. Implementation is
delegated through the standalone [NeuralNetwork agent prompt](tools/compare_runtime/pass16/NEURALNETWORK_PROMPT.md),
ready for the user to run separately against that repository. Pass 16 production
optimization stops here. Neither repository's production source or dependencies
were changed; no backend branch, commit, PR, release or local production fix
was created.

## Boundary decision

ONNX supplies owned backend Tensor inputs and an existing disjoint destination
directly to `tensor.AddInto` and `tensor.MatMulInto`. The equal-shape Add matrices,
copied buffers, MatMul plan and staging output are constructed **inside
NeuralNetwork**, after that boundary. ONNX neither constructs nor owns those
matrices. Its public tensor accessors return copies; duplicating arithmetic or
using unsafe storage access in ONNX would bypass the backend's contract.

The strongest concrete opportunity is the equal-shape Add adapter. MatMul
already reads host input payloads directly and requires a separate investigation
of staging and its guarantee that all returned errors leave the destination
unchanged. A new Matrix view API is a hypothesis, not a demonstrated requirement;
`Matrix.ValuesInto` already offers a smaller backend option for one copied result.
Internal backend changes may suffice without any public API addition.

## Fresh evidence

2026-11-12, Apple M3, 24 GB, darwin/arm64, macOS 26.5.2, Go 1.27.2.
ONNX `0803d3ea12ae02597592d014e4792c95ec5ca364`; released NeuralNetwork v0.10.9
(`740a244bf56214bb3443a69287a2d800674620e4`). Performance processes used
GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off, GOARM64=v8.0, default optimization.
Seven 200-ms repetitions per benchmark, all samples retained. Profiling,
compiler diagnostics and race instrumentation were separate from latency runs.

Backend diagnostics use deterministic finite operands; ONNX adapters use actual
ramp intermediates. Destinations and setup are outside timing. These scopes
overlap and are not additive costs or demonstrated production savings.

| Scope | Median µs | Range µs | B/op | allocs/op |
| --- | ---: | --- | ---: | ---: |
| Backend equal-shape Add `[1,10]` | 0.176100 | 0.172100–0.207200 | 336 | 7 |
| Its matrix adapter, excluding outer tensor validation | 0.101500 | 0.098290–0.102800 | 336 | 7 |
| Add validation + direct arithmetic diagnostic | 0.079250 | 0.078450–0.093400 | 0 | 0 |
| Four preallocated ten-float copies | 0.005786 | 0.005777–0.005868 | 0 | 0 |
| Backend MatMul `[1,256] × [256,10]` | 1.681000 | 1.643000–1.789000 | 144 | 3 |
| MatMul plan construction | 0.063140 | 0.062650–0.063500 | 96 | 2 |
| MatMul batch execution with staging, existing plan | 1.564000 | 1.556000–1.578000 | 48 | 1 |
| Host product into existing storage | 1.497000 | 1.492000–1.526000 | 0 | 0 |
| MatMul tensor validation + pre-resolved direct product diagnostic | 1.577000 | 1.571000–1.592000 | 0 | 0 |
| One preallocated ten-float copy | 0.001843 | 0.001833–0.001852 | 0 | 0 |
| ONNX equal-shape Add numerical adapter | 0.195800 | 0.194600–0.198200 | 336 | 7 |
| ONNX MatMul numerical adapter | 1.690000 | 1.667000–1.695000 | 144 | 3 |
| Complete ordinary-ramp inference | 55.841000 | 55.514000–56.453000 | 92,874 | 73 |

The two channel Add adapters remain allocation-free: 1.929 µs for
`[1,8,28,28] + [8,1,1]`, and 1.177 µs for `[1,16,14,14] + [16,1,1]`.
The Add adapter sum is 3.3018 µs; most of that arithmetic is outside this task.
Seven fresh ramp repetitions reproduce the released allocation baseline exactly.
No new before/after or fixed-policy Go/ORT experiment is claimed.

The Add diagnostic retains input/broadcast/destination validation but changes
dispatch and establishes only this finite fixture. The MatMul diagnostic omits
shape-plan construction, signature guards, batch handling and staging, retaining
input/destination validation against already resolved geometry. Neither is a
complete general implementation. Their median differences are 0.096850 and
0.104000 µs, respectively; the 0.200850-µs sum is a small diagnostic indication,
not a whole-Run promise. Add complete/validated CVs are 6.37% / 6.24%; MatMul
complete/validated CVs are 3.19% / 0.49%. All ranges/samples are preserved.

## Allocation and copying ledger

| Backend-owned site, per call | B | Objects |
| --- | ---: | ---: |
| Add: two Matrix input structs and copied payloads | 192 | 4 |
| Add: Matrix output struct and payload | 96 | 2 |
| Add: copied `Values()` payload | 48 | 1 |
| **Equal-shape Add total** | **336** | **7** |
| MatMul: 72-byte plan, allocator-rounded | 80 | 1 |
| MatMul: result-shape array | 16 | 1 |
| MatMul: temporary output payload | 48 | 1 |
| **MatMul total** | **144** | **3** |

Each Matrix is 48 bytes. Each ten-float buffer requests 40 bytes and occupies
48 allocator bytes. Filtered rate-one profiles of 10,000 complete calls account
for exactly 70,000 Add objects / 3,360,000 B and 30,000 MatMul objects /
1,440,000 B at those sites. Benchmark setup is excluded from the filtered ledger.

Source-derived logical bytes copied: Add 160 B (four 40-B copies), MatMul 40 B
(staged result to destination). Copy read/write traffic is twice those numbers;
this is not a DRAM bandwidth measurement. The preallocated-copy probes exclude
allocation, validation and arithmetic. Their small latency does not explain the
entire adapter cost. Large owned tensor destinations still dominate full-Run bytes.

Removing every observed allocation in both paths would conditionally give
92,394 B / 63 objects, a maximum 480-B / 10-object opportunity. No implementation
has achieved that, and necessary error-atomic staging may remain.

## Compiler and semantic findings

Matrix constructor structs/data and `Values` copies escape; the Add adapter's
tensor arguments and operation function do not. Generated code retains the
constructor calls, indirect matrix operation and result copying. The MatMul plan
and shape escape. Its runtime-sized staging slice is reported as non-escaping,
yet generated code and profiling show an actual allocation; escape analysis
alone is not an allocation counter. Small shape literals are not extra allocations.

Tensor destinations reject pointer and shallow-copy identity aliases, including
empty results, even though Matrix elementwise destinations may alias inputs.
Inputs and shared initializers remain read-only. MatMul's all-errors destination
atomicity is stronger than Add's argument-error guarantee. Host product kernels
are synchronous and return nil after validation, but Metal has fallible adapter,
execution and synchronization paths. Dynamic shapes, batch/vector promotion,
empty contractions, float32 accumulation, concurrent readers, output lifetime and
all existing build paths remain backend obligations.

ONNX checks cancellation/deadlines around these synchronous kernels; backend
Add/MatMul have no context parameter. ONNX's conservative workspace accounting
currently charges 160 B for this equal Add and 40 B for host MatMul. Those logical
charges can stay during later integration if still sound. No persistent state or
initialization work was added in this pass. No retained-heap/RSS experiment or
new initialization saving is claimed.

The handoff includes the exact call paths, validation order, ownership/alias
rules, dynamic/default/override behavior, cancellation, lifetime, race and device
contracts; rejected explanations and API hypotheses; independent backend
measurement and full acceptance requirements; repository branch/commit/PR
workflow; and the required return report for later ONNX integration.

## Validation and next step

Fresh ONNX complete suites pass with CGO disabled, purego, race and purego+race;
vet and module verification pass. The test-only finite Add/MatMul ceilings match
complete results bitwise and preserve inputs in all four modes. These validate
the diagnostics and unchanged runtime, not a future optimization. Cross-builds
and complete backend before/after validation are required from the next agent;
they are not represented as newly executed in this pass.

Evidence and replay instructions are in [the Pass 16 directory](tools/compare_runtime/pass16/README.md).
After the backend change is reviewed, merged and published, a separate ONNX step
must consume the released API/version and perform component/whole-Run transfer
validation. There is no ONNX workaround, dependent implementation or later
optimization candidate in this change.
