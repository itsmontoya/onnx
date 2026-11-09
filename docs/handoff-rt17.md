# RT-17 exact version-profile handoff

Completed on 2026-11-08, stacked on **rt-16-storage-performance** at
`e801cf3cdc1d75d0f82319da03fa229147d208fb` ([PR #17](https://github.com/itsmontoya/onnx/pull/17)).
The implementation revision is `5840f3ed99e43d04bbfef36a890d7e44862c5c30` on
[rt-17-version-profiles](https://github.com/itsmontoya/onnx/tree/rt-17-version-profiles).
The completion revision is the commit introducing this handoff and checking
RT-17 in ROADMAP.md. Keep the dedicated PR open; no merge or tag is part of this
task. RT-16 remains unchecked and blocked on unpublished NN-13 measurements.

## Selected profile and evidence

The only additional accepted profile is **IR 8, one empty standard-domain
opset-17 import**. The existing IR-8/opset-13 behavior remains. Imports 14–16,
18 and every other untested import/IR/domain still fail deterministically.
There is no runtime converter, model-envelope relabeling or new operator family.

The target is an actual dynamic-batch float32 MLP inference export through
PyTorch 2.2.2's `torch.onnx.export`, source commit
`39901f229520a5256505ec24782f716ee7ddc843`. Its original module/explicit weights
are in [the optional generator](../tools/check_profiles.py); the exporter emits
IR 8 and opset 17 directly. The raw buffer is stored unchanged at
[testdata/profiles/exports/pytorch_mlp.onnx](../testdata/profiles/exports/pytorch_mlp.onnx),
SHA-256 **`34d16bc9604e8d1166a27255750d91d5caf20f29833cf017d1f7d55c7c80c9a1`**.
Named float32 `features`/`probabilities` use `[batch,2]`; batches 1 and 3 match
PyTorch eager inference, ONNX's evaluator and ORT. The example runs it with
`-graph exported`, without installing any reference tools.

This single original module is the audited demand/evidence; no external corpus
or general exporter compatibility is claimed. The exporter is an optional BSD-3-
Clause development tool; no exporter implementation/model/weights are copied.
The repository still has no LICENSE, and original fixtures have no separately
declared license. No license was invented. The [fixture guide](../testdata/profiles/README.md)
records source, tool/support-package versions, license distinctions and commands.

## Implementation and version decisions

[capabilities.json](../capabilities.json) has 50 implemented records for the same
25 operators: exact disjoint imports [13,13] and [17,17].
[The stored audit](../testdata/profiles/schema_audit.json) compares all 25 actual
applicable ONNX 1.16.2 schemas, including arities, attributes/defaults, type
constraints, standard-function flags and documentation hashes. Nine selections
change: Identity-16, Add/Sub/Mul/Div-14, Relu-14, Reshape-14, Shape-15 and
BatchNormalization-15. The other sixteen retain the prior schema. New type and
standard-function forms do not expand dense float32/int64 or inference support.

- [Registry](../internal/operators/registry.go) permits nonoverlapping records
  for the same schema, selects the newest applicable schema first and then
  requires an explicitly tested interval for that schema. It never falls back
  to older semantics when a newer selected schema lacks coverage. Gaps remain
  gaps in `Capabilities()` and the [Go-generated table](getting-started.md#implemented-capability-table),
  whose new tested-import column distinguishes all records.
- [Shared envelope policy](../internal/profile/profile.go) is used by graph
  validation and compilation, with manifest agreement checked in Go. Metadata,
  compiled Steps and node diagnostics retain the actual import instead of 13.
- [Shape planning](../internal/operators/shape_planner.go),
  [shape rules](../internal/operators/shape_rules.go) and
  [execution](../shape_call.go) resolve version-specific flags/defaults.
  Reshape-13 rejects allowzero even when 0. Reshape-14 defaults it to 0 for
  dimension copying; 1 means literal zero and forbids zero with -1. Shape-13
  rejects start/end; Shape-15 implements negative indexing, rank clamping and
  empty slices for start >= end, including int64 boundary attributes. Static
  and value-dependent failures remain at Load and Run respectively.
- [CNN config](../internal/operators/cnn_config.go) accepts BN-15 training_mode
  only when 0, with exactly one output and five float32 inputs. BN-9 rejects the
  new attribute even when 0. Stored-statistics inference is unchanged and never
  updates statistics. Extra empty BN training slots still fail. MaxPool's empty
  optional indices slot remains positional; named indices remain unsupported.
- [Step](../internal/operators/step.go) carries fixed version flags. The same
  256-byte scalar metadata bound remains sufficient (224 bytes on arm64 under
  the logical payload contract); its focused regression test checks the actual
  field tree against the shared compiler/snapshot bound. Shape staging still
  charges the full rank snapshot even when slicing to a shorter output.

Public runtime APIs, ownership, backend behavior, limits and scheduling are
unchanged. Go 1.27.2 / language 1.26.1, neuralnetwork v0.7.0 at
`6048452c57a0a293db4591cd2e2febd7d6b743a3`, protobuf v1.36.12 and the ONNX source
pin `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557` remain unchanged. The adjacent
backend checkout was not edited. No NN-13/RT-16 optimization was implemented.

## Fixtures, checks and actual results

The [new provenance](../testdata/profiles/provenance.json) contains **67**
executed offline cases: equivalent isolated graphs for all 25 operators under
both profiles, exact int64 extrema/16777217, empty/special identities, versioned
zero/slice behavior, BN stored statistics and the two-batch actual export.
ONNX 1.16.2 agrees on **66**; the BN-9 evaluator's momentum application remains
one executed difference. Its BN-15 evaluator agrees. ORT 1.19.2 agrees on **all
67**. Actual outputs retain float32 bits/int64 payloads and shape/type metadata;
independent expectations and reference observations remain distinct. Normal Go
tests invoke no Python and check every new artifact hash, applicable schema,
named output and retained output lifetime. All inherited fixture bytes/hashes
remain exactly identical to the base.

[Version-profile tests](../version_profile_test.go) also cover old-attribute
rejections, untested IR/import gaps, extended dtypes/sequence values/functions/
training rejection, BN training/extra slots, MaxPool optional slots, dynamic
allowzero errors/recovery/changing batches, actual-import diagnostics, canceled
runs, live/workspace limits, preserved results after closure and independent
concurrent exporter sessions. Conformance validates exact selected schemas;
old fixture counts and explicit exception classifications remain unchanged.

Local checks used **Go 1.27.2, Darwin/arm64**, cached modules and
`GOCACHE=/private/tmp/onnx-runtime-gocache`. Passed fmt, tidy -diff, mod verify,
build, vet, uncached full tests in normal, `CGO_ENABLED=0`, cgo-disabled purego
and race modes, and the Go-only generated-table check. Wire regeneration with
protoc 33.2, all seven inherited no-write reference generators and the new
profile generator reproduced their artifacts. The pinned archive hash remains
`d32b94c9c6feda1403d340813b8926d5825082f497e1a65285b50faa31aad3c5`.
All four native fuzz smokes passed at `GOMAXPROCS=2`, `-parallel=2`, 3 seconds:
protobuf **60,949**, tensor **79,053**, graph **28,088**, planning **114,810**
executions. Fuzz counts are observations within bounded targets, not broader
compatibility claims. Go style/type-method colocation, local Markdown links and
`git diff --check` pass.

A fresh clone at `5840f3ed99e43d04bbfef36a890d7e44862c5c30` used a new
HOME/build cache, PATH containing only the selected Go toolchain, cached module
sources, `GOTOOLCHAIN=local`, `GOWORK=off`, `GOPROXY=off`, `GOSUMDB=off` and
`CGO_ENABLED=0`. All **28** commands passed: MLP/branch/shape/CNN/exported MLP
at batches 1/3 and external MLP at batches 1/3, each in default and purego modes,
plus the table check, example binary build and both full uncached Go test modes.
The clone remained clean. No Python or native inference runtime was on PATH;
initial tool/module provisioning still requires installation/network or caches.
The table generator also rejected a deliberately changed import row and missing
end marker; `-write` restored the guide's exact original bytes.

Hosted Linux/amd64 checks cover the final dedicated PR head, including the
support-table check, normal/cgo-disabled/purego/race tests and bounded fuzzing.
Their actual result and exact final head are verified on the PR before handoff.

## Remaining boundaries and next task

Dense value/rank restrictions, logical rather than RSS budgets, synchronous
backend cancellation limits, external-data resolver rules and all inherited
numerical exceptions remain. No additional IR, dtype, operator, external
converter, provider framework or optimization is implied by the profile.

RT-17's acceptance is complete. RT-16 remains unchecked until NN-13 publishes
its required results and its separate acceptance work resumes. The next
independent roadmap task is **RT-19**, in a separate sequential task/branch/PR.
No RT-19 work, backend change, merge, tag or publication was started here.
