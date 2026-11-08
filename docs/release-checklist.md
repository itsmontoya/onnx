# M1–M4 release checklist

RT-18 prepares documentation and a reviewable checklist for the implemented
subset. Publishing a tag, merging the stacked PRs and deciding the project's
license are separate maintainer actions. No tag is published by this task.
The [RT-18 handoff](handoff-rt18.md) records exact revisions and validation.

## Scope and known gaps

- [x] Document IR **exactly 8**, one **empty-domain opset-13** import, CPU serial
  scheduling and dense **float32/int64**. Operator schema versions remain
  distinct from the import version.
- [x] Generate the operator table from `Capabilities()`, whose embedded
  `capabilities.json` records also configure validation. Check without Python:
  `CGO_ENABLED=0 go run ./tools/capabilitiesgen`; regenerate with `-write`.
- [x] Document named I/O, default overrides, independent copied outputs,
  constructor/accessor ownership, session contention, lifecycle and recovery.
- [x] Document all finite limits, logical payload accounting rather than RSS,
  synchronous-kernel cancellation and cooperating caller I/O requirements.
- [x] Document explicit external resolution, exact ranges, checksum rejection,
  reader/resolver ownership, confined directory paths and js/plan9 restrictions.
- [x] Retain actual reference differences, expected rejections, provenance/hashes
  and independent expectations. No skipped test counts as support.
- [x] State gaps: other IR/opsets/domains/dtypes, sparse/sequence/map/optional
  values, local functions, training, subgraphs/control flow, tensor segments,
  unknown input ranks and dynamic nonleading input dimensions are unsupported.
  MaxPool indices, later-schema attributes and external checksum metadata are
  unsupported. See the [generated operator restrictions](getting-started.md#implemented-capability-table).

All nodes execute even when requesting fewer outputs. The executor retains all
intermediates for a run, uses conservative copying-backend charges and offers no
cross-model memory governor, borrowed mutable tensors, optimizer, GPU, native
ONNX Runtime bridge or broad exporter/corpus compatibility guarantee. Those
boundaries are part of this release profile, rather than skipped acceptance
checks. Empty and nonfinite numerical edge policies are explicit in the
[runtime contract](runtime-contract.md#numerical-and-empty-value-decisions).

## Go verification

The RT-18 checks use **Go 1.27.2**, module language baseline **Go 1.26.1**,
**neuralnetwork v0.7.0** at
`6048452c57a0a293db4591cd2e2febd7d6b743a3` and **protobuf v1.36.12**.
Versions/checksums remain in go.mod/go.sum; no dependency changed. Local checks
run on **Darwin/arm64**. The GitHub **Linux/amd64** workflow runs the full Go
suite including bounded fuzzing and the new generated-table check. Hosted
results and exact head revision are recorded on the stacked PR before handoff.
Neither platform result establishes support on every Go platform. Directory
symlink tests explicitly skip on hosts unable to create symlinks; those skips
are not confinement evidence for such a host.

Run from the repository root after `go mod download`:

```sh
go mod verify
go mod tidy -diff
go fmt ./...
CGO_ENABLED=0 go run ./tools/capabilitiesgen
go build ./...
go vet ./...
go test -count=1 ./...
CGO_ENABLED=0 go test -count=1 ./...
CGO_ENABLED=0 go test -tags=purego -count=1 ./...
go test -race -count=1 ./...
git diff --check
```

- [x] Formatting, module verification/tidiness, build and vet pass.
- [x] Uncached normal, cgo-disabled, cgo-disabled purego and race tests pass.
- [x] The support table reproduces exactly and rejects deliberate guide drift;
  `-write` restores canonical content.
- [x] The MLP and CNN examples run at batches 1 and 3 in a fresh checkout with
  `CGO_ENABLED=0`; the external MLP also runs. Go-only PATH, cleared environment,
  offline cached modules and `GOTOOLCHAIN=local` demonstrate no Python/protoc/
  compiler/native inference service invocation. Both default and purego paths
  pass. This validates runtime use after initial Go dependency provisioning.
- [x] The guide's public API snippets compile and its failure probe reproduces
  unsupported IR, absent external resolver, byte-budget and missing-input
  errors using `errors.Is`.
- [x] The four bounded fuzz targets pass. See
  [the RT-15 target bounds and commands](development.md#resource-hardening-and-fuzzing).
- [x] Regeneration reproduces checked-in bindings/models/typed values/provenance;
  relative documentation links and `git diff --check` pass.

To repeat the Go-only environment check after cloning the documented branch
and downloading modules, capture these paths before clearing the environment:

```sh
runtime_path="$(go env GOROOT)/bin"
module_cache="$(go env GOMODCACHE)"
clean_home=$(mktemp -d)
env -i PATH="$runtime_path" HOME="$clean_home" GOCACHE="$clean_home/cache" \
    GOMODCACHE="$module_cache" GOTOOLCHAIN=local GOWORK=off GOPROXY=off GOSUMDB=off \
    CGO_ENABLED=0 "$runtime_path/go" run ./examples/inference -graph cnn -batch 3
rm -r "$clean_home"
```

Repeat the env command for `-graph mlp -batch 1`, both graphs at the other batch,
and `./examples/external -batch 3`; add `-tags=purego` immediately after `run`
for the portable path. This requires an already populated module cache and an
installed selected Go toolchain; missing modules are provisioning failures,
rather than evidence that Python/native libraries are needed.

The race detector is a development check requiring a supported host/cgo compiler;
that requirement is separate from cgo-disabled runtime use. CI's fuzz smoke
runs are robustness checks. A one-iteration benchmark smoke is not a performance
measurement. No credible repeatable benchmark measurements have been recorded
for this task, so no latency/throughput or improvement figures are published.
[The NN-13 handoff](handoff-nn13.md#reproduce-and-establish-a-baseline) gives
reproducible batch-1/batch-3 Load/First/Warm commands and measurement boundaries;
RT-16 consumes measured results later.

## Reference evidence

Reference generation is optional development work, separate from clean Go use.
Pinned tools are **Python 3.9.6, ONNX 1.16.2** at
`3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, **NumPy 1.26.4**, **Python protobuf
4.25.3** and, for secondary comparisons, **ORT 1.19.2** CPUExecutionProvider.
Original graphs use ONNX helper/numpy_helper, explicit IR/imports and the full
checker. No third-party exporter has been validated. Fixtures contain SHA-256
for each model/named TensorProto value plus actual reference returns/errors.

| Set and provenance | Actual evidence, without adding overlapping sets together |
| --- | --- |
| [Profile](../testdata/profile/provenance.json) | Original Identity model and selected schema audit establish the exact initial envelope; decoding alone is not conformance |
| [M1](../testdata/m1/provenance.json) | 80 evaluator agreements and one explicit empty-axis Softmax extension; Relu NaN propagation is repaired by the runtime |
| [Conformance](../testdata/conformance/README.md) | 96 classified cases: 87 agreements, one empty-Softmax extension, four executed independent differences and four required unsupported-model rejections |
| [M2](../testdata/m2/README.md) | 66 executed cases: 60 evaluator agreements, two empty-prefix Flatten evaluator rejections and four empty-Gather shape differences |
| [Inference](../testdata/inference/README.md) | Six agreements: MLP, branch and shape graph at batches 1 and 3, with independent NumPy cross-checks |
| [M3](../testdata/m3/README.md) | All 87 execute against independent expectations; ONNX evaluator: 49 agreements, 26 differences, 12 rejections; ORT: 66 agreements, six differences, 15 rejections. Composed CNN batches 1/3 agree with ORT and differ from the evaluator |
| [External](../testdata/external/README.md) | Externalized original MLP is checker/loader-valid and matches inherited batch-1/3 outputs; explicit ranges and shared-file hashes are recorded |
| [Sessions](../testdata/sessions/README.md) | Original helper-authored Identity/Constant/default/passthrough graphs have independently specified values; not advertised as evaluator-generated goldens |

Gather empty indices retain the schema-derived full prefix/indices/suffix shape
where the pinned evaluator returns `[0]`. Pooling records SAME/dilation, ceil
trimming, padding/empty/nonfinite disagreements with either oracle. Stored-
statistics BatchNormalization differs from the evaluator's momentum application;
zero variance/epsilon also differs from ORT. Those are actual executed differences,
rather than hidden failures. Read the linked fixture records for per-case
observations and comparison policies. Ordinary Go tests verify hashes, exact
integer/type/shape results and stated float tolerances offline.

The pinned upstream backend archive contains **1,422** models and **none** with
exactly IR 8/opset 13. Three copied upstream models retain their original IR-7
bytes and must fail; the opset-18 reduction case must fail too. No version
relabeling, skipped upstream case or backend-only kernel test establishes runtime
compatibility. Fixture sets overlap; their counts are not a combined model
coverage or industry conformance percentage.

Optional reproduction commands are in [development](development.md#regeneration)
and the fixture READMEs. Run all no-write tools with the pinned environment;
`tools/check_conformance.py` additionally requires the pinned upstream archive
and checks its SHA-256. Expected reference errors/differences remain successful
reproduction of those observations, never new agreement claims.

## Maintainer release actions

- [ ] Choose and add a project LICENSE for runtime source and explicitly decide
  licensing for original fixtures. Retain existing provenance history and then
  update license declarations deliberately. RT-18 does not invent this policy.
- [ ] Review [third-party notices](../third_party/README.md) and include the
  identified ONNX Apache-2.0, backend/SIMD MIT, protobuf/x/sys/Go BSD texts and
  applicable source headers in the actual distribution.
- [ ] Review and merge the stacked task PRs through the normal maintainer process.
- [ ] Select a release commit after review, rerun checks for that revision, review
  release notes against the documented subset and publish a tag separately.

RT-18's documentation work is complete when its checked technical items pass;
the unchecked maintainer actions remain visible before any public release.
NN-13/RT-16 optimization, RT-17 profiles and RT-19 audited-corpus expansion are
later tasks and do not require changing this documented M1–M4 profile.
