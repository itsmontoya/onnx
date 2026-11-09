# RT-19 audited corpus and normalization handoff

Completed on 2026-11-08 against RT-17 base
[`4ded3992742ff4dfa33f12c3c7122544e1f6d503`](https://github.com/itsmontoya/onnx/tree/4ded3992742ff4dfa33f12c3c7122544e1f6d503).
The implementation is
[`1744afa991a88468aeff708c8b0a12e19a9df405`](https://github.com/itsmontoya/onnx/tree/1744afa991a88468aeff708c8b0a12e19a9df405),
with conservative scalar-staging accounting completed at
[`a1b83943dae6fe38b2435a6f4ab73fb792961422`](https://github.com/itsmontoya/onnx/tree/a1b83943dae6fe38b2435a6f4ab73fb792961422).
The completion revision is the commit introducing this handoff and RT-19's
roadmap checkbox (`git log -1 --format=%H -- docs/handoff-rt19.md`). Dedicated
branch **rt-19-model-corpus** is stacked on **rt-17-version-profiles**. Keep its
PR open; no merge, tag, backend change or later initiative is part of this task.

## Corpus decision and actual evidence

[The report](model-compatibility.md) and [corpus index](../testdata/corpus/index.json)
record 33 byte-original Apache-2.0 ONNX v1.16.2 models, actual IR/imports,
source/license/hash, named types/shapes, nodes/schemas/attributes and external
requirements. The pinned source revision remains
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`; archive SHA-256 remains
`d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5`.
Of its 1,422 backend models, zero use exact IR 8/import 13 and 65 use exact
IR 8/import 17. No source versions or bytes were rewritten to enable a model.

Nineteen fused LayerNormalization exports form the smallest useful selected
normalization workload: one missing operator enables all 19. At the exact
RT-17 base, the unchanged default-axis model was executed as a Load probe and
failed with ErrUnsupported at node 0, operator LayerNormalization, import 17.
RT-19 runs that model and all three Y/Mean/InvStdDev outputs. These are upstream
backend test exports, not pretrained networks or general transformer evidence.
The 14 remaining corpus cases stay actual required unsupported-model rejections.
The deterministic static report changes from 0 enabled/33 unsupported/63 findings
to 19 enabled/14 unsupported/44 findings, retaining all failing node indices.
Its documented audit scope does not replace runtime validation.

All model/value/source artifacts retain the upstream Apache-2.0 grant and
[unmodified LICENSE](../third_party/onnx/LICENSE); the pinned archive has no
NOTICE. There are no separately licensed learned weights or external-data needs.
The original source export retains its SPDX header. The repository still has
no project LICENSE, and original test-only helper graphs have no separate
redistribution grant. No framework license was inferred to license weights.

## Implementation, APIs and contracts

[The selected independent task](tasks/layer-normalization.md) defines exact
schema/type scope and enabling/failing fixtures. The canonical manifest adds
one implemented **LayerNormalization-17 / import 17** record: 51 records total,
25 operators at import 13 and 26 at import 17. Float32 X/Scale/optional B,
float32 Y/Mean/InvStdDev, stash_type=1, rank >=1, trailing axis suffix, finite
nonnegative epsilon and nonempty normalized dimensions are required. Scale/B
broadcast unidirectionally to X. Optional input/output holes retain position;
all named outputs execute. Unsupported types, attributes, modes and import 13
remain deterministic errors. Known failures are checked at Load; dynamic ones
remain contextual Run errors with session recovery.

[Planning](../internal/operators/normalization.go), the
[compiler](../internal/operators/compiler.go), [registry](../internal/operators/registry.go)
and [Step](../internal/operators/step.go) select defaults and infer separate
statistics shapes. Statistics dimension snapshots are reserved before copying;
Step's Epsilon remains inside the existing 256-byte fixed scalar metadata bound.
[Evaluation](../evaluation.go) validates/stores each named output independently.
[normalizationCall](../normalization_call.go) composes published NN-12
ReduceMeanInto/SqrtInto and arithmetic kernels for centered population variance.
It does not implement another numerical engine, expose ONNX primitive reductions
or execute arbitrary function bodies. Explicit internal axes are nonempty suffix
axes with keepdims=true, so no ONNX attribute/input-axes or noop conversion is
implied by the backend calls.

Disjoint numerical destinations are reserved individually before allocating,
including internal stats whose optional output slot is omitted. Conservative
workspace is `20*X_bytes + 16*statistics_bytes + 16`: retained full/statistic
buffers, matching-shape arithmetic bridge copies and two scalar staging/owned
payloads are covered, including zero-byte empty-prefix inputs. This bound is not
a measured peak. Resident input/default copies and public output copies retain
their existing bounds. Cancellation brackets each synchronous kernel; no
in-kernel interruption promise is made. Inputs are read-only and returned
outputs remain independent after later runs/Close.

No public function/type/signature changed. IR profiles, dense float32/int64
values, ownership, serial scheduling and external-data contracts are unchanged.
The published backend remains neuralnetwork v0.7.0 at
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, protobuf v1.36.12 and Go 1.27.2 /
language baseline 1.26.1. No local replace or adjacent backend edit was used.
NN-12 already supplies this bounded set; no further library task is needed.

## Fixtures and validation

[The generator](../tools/check_corpus.py) audits source/license/schema/defaults
and reproduces all new artifacts with pinned Python 3.9.6, ONNX 1.16.2,
NumPy 1.26.4, protobuf 4.25.3 and ORT 1.19.2. Ordinary Go tests remain offline
and Python-free. All **1,211 inherited testdata artifacts** have identical Git
blob hashes to the RT-17 base. All old numerical exception classifications and
all old fixture bytes remain intact.

[Normalization provenance](../testdata/normalization/provenance.json) contains
28 executed cases: 19 source models independently cross-checked with NumPy
float32 centered population variance, plus nine original isolated/DAG cases.
All 28 agree with the ONNX evaluator; 27 agree with ORT. ORT actually rejects
scalar affine broadcasting, which the schema/evaluator/runtime allow. Its error
is retained separately, not skipped or counted as agreement. Cases cover axes,
epsilon, no bias, optional Mean hole, three outputs, scalar scale, empty prefix,
constant epsilon 0, NaN/infinity and a residual Add/normalization/branched Mul/Add
DAG at batches 1/3. Exact types/shapes, finite tolerance 1e-5 + 1e-5*abs(expected),
NaN classes and infinity signs are checked; signed-zero equality is not promised.

Go tests additionally cover profile/stash/attribute/axis/shape rejection,
dynamic empty-reduction recovery, explicit empty bias slot, output subsets and
order, read-only inputs, retained outputs, concurrent independent sessions,
cancellation before and between kernels, exact and one-byte-short workspace/
combined-live reservations and empty-prefix scalar staging. Corpus tests hash
and validate source metadata and execute every positive or negative Load.

Local checks used **Go 1.27.2 Darwin/arm64** and
`GOCACHE=/private/tmp/onnx-runtime-gocache`. Passed fmt, tidy -diff, mod verify,
build, vet, generated support-table check, uncached full normal/cgo-disabled/
cgo-disabled purego/race tests, wire regeneration with protoc 33.2, all eight
inherited reference generators and the new corpus generator. The final staging
reservation change was followed by all four full Go test configurations.
Four bounded fuzz smokes passed with GOMAXPROCS=2, -parallel=2, -fuzztime=3s:
protobuf **69,836**, tensor **163,001**, graph **31,151**, planning **138,380**
executions. These observations do not expand the target bounds or compatibility.
Go style/type-method colocation, all 397 local Markdown links and git diff
--check pass. Deliberate support-table row drift and a missing end marker fail;
-write restores the complete guide’s exact bytes.

A fresh clone at `a1b83943dae6fe38b2435a6f4ab73fb792961422` uses a new HOME/build
cache, PATH containing only Go's selected bin, cached module sources,
GOTOOLCHAIN=local, GOWORK=off, GOPROXY=off, GOSUMDB=off and CGO_ENABLED=0.
All **30** commands pass: inherited MLP/branch/shape/CNN/exported MLP and external
MLP at batches 1/3 under default and purego paths, both new normalization example
paths, table check, example binary build and both uncached full Go test modes.
The clone stays clean. The new example runs all three named outputs from an
unchanged licensed model. Initial provisioning still requires installed Go and
modules; no Python/protoc/native inference runtime is on the execution PATH.
No benchmark or performance improvement claim is made.

Hosted Linux/amd64 checks cover the final dedicated PR head, including the
support table, normal/cgo-disabled/purego/race suites and bounded fuzzing. Their
actual result and exact PR head are verified before handoff; local Darwin
checks alone are not treated as hosted Linux evidence.

## Unselected work and remaining boundaries

The [report's independent task table](model-compatibility.md#bounded-next-initiatives)
links twelve separate future initiatives: Slice, Expand, Pad, Resize, typed
bool/int32/comparison/Where, float16/bfloat16, RNN then GRU/LSTM, quantization,
functions/control flow and sequence/map/optional/sparse value design. Each has
contracts, dependencies, named failing fixture/workload, required library work
and acceptance. New licensed map/sparse/bfloat16/function/bounded-Loop selections
remain explicit prerequisites in their tasks. These files do not implement
those production features or claim their models run.

RT-19's corpus/report/first selected set acceptance is complete. The existing
logical-budget, copying-backend, synchronous-cancellation and external-data
limitations remain. The 14 negative models, primitive ONNX reductions/math,
other profiles/dtypes, recurrent/quantized/control-flow and structured values
remain gaps. **RT-16 remains unchecked**, awaiting published NN-13 measurement/
optimization results. License decisions, review/merge and publication remain
maintainer actions; no later task was started.
