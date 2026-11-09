# RT-16 dependency assessment

**Status on 2026-11-08: blocked on unpublished NN-13 results.** RT-16 remains
unchecked. This is a dependency report, not an optimization or an acceptance
report. RT-12, RT-13 and RT-15 are complete; NN-13 is the remaining prerequisite.

The assessed runtime revision is RT-18
[`68df79fe28e7e1c1b55bba0963de7565059f02fd`](https://github.com/itsmontoya/onnx/tree/68df79fe28e7e1c1b55bba0963de7565059f02fd),
on `rt-18-usage-compatibility` ([open PR #16](https://github.com/itsmontoya/onnx/pull/16)).
This dedicated `rt-16-storage-performance` branch is stacked on that branch.
Runtime source, fixtures, APIs, limits, module pins and ownership are unchanged.

## Live dependency evidence

Authenticated GitHub reads on 2026-11-08 established the following:

- The backend remote is `https://github.com/itsmontoya/neuralnetwork`.
  Its latest [release](https://github.com/itsmontoya/neuralnetwork/releases/tag/v0.7.0),
  newest tag `v0.7.0` and [main revision](https://github.com/itsmontoya/neuralnetwork/commit/6048452c57a0a293db4591cd2e2febd7d6b743a3)
  all resolve to `6048452c57a0a293db4591cd2e2febd7d6b743a3`, the version
  already pinned in [go.mod](../go.mod).
- [PR #114](https://github.com/itsmontoya/neuralnetwork/pull/114) merged on
  2026-11-06 at that main revision. Its head was
  `af93bbb0954c0c61e4b38902f939a82ce4845068`; its only changes are the NN-13
  handoff and a tensor-design link. It has no discussion comments. No newer PR
  or open backend PR was present at assessment time.
- The complete public branch inventory contains only `main`, at
  `6048452c57a0a293db4591cd2e2febd7d6b743a3`. Thus no public branch or
  open/recent PR supplies an unmerged NN-13 implementation and measurements
  to evaluate as a source revision or Go pseudo-version.
- The [tracked NN-13 handoff at the exact released revision](https://github.com/itsmontoya/neuralnetwork/blob/6048452c57a0a293db4591cd2e2febd7d6b743a3/docs/handoffs/NN-13.md)
  still records NN-13 as unimplemented, with no runtime measurements or
  bridge/workspace optimization. Main contains that same handoff.

The backend's original report described missing RT-12 inputs on 2026-11-06.
**Those inputs are now available.** The remotely accessible implementation is
[`3b2895ab233b28c0c7fc731e5927e82c99416fcf`](https://github.com/itsmontoya/onnx/tree/3b2895ab233b28c0c7fc731e5927e82c99416fcf),
with final handoff
[`007b8abe53b63f35ae4d09abaed93015b7256b4d`](https://github.com/itsmontoya/onnx/tree/007b8abe53b63f35ae4d09abaed93015b7256b4d)
and [PR #12](https://github.com/itsmontoya/onnx/pull/12).
The [runtime handoff](handoff-nn13.md) supplies working MLP/shape-heavy graphs,
named inputs and expected outputs, batches 1 and 3, hashes and executable
Load/First/Warm baseline commands. NN-13 can resume from those inputs without
waiting for RT-16. Its missing published results remain RT-16's blocker.

Recheck the live evidence before resuming; these are read-only commands:

```sh
gh api repos/itsmontoya/neuralnetwork/releases --paginate
gh api repos/itsmontoya/neuralnetwork/tags --paginate
gh api repos/itsmontoya/neuralnetwork/commits/main
gh api 'repos/itsmontoya/neuralnetwork/branches?per_page=100' --paginate
gh pr view 114 -R itsmontoya/neuralnetwork --json state,mergedAt,headRefOid,body,comments
gh api repos/itsmontoya/neuralnetwork/pulls/114/files --paginate
gh api 'repos/itsmontoya/neuralnetwork/pulls?state=open&per_page=100' --paginate
gh api 'repos/itsmontoya/neuralnetwork/pulls?state=all&sort=created&direction=desc&per_page=1'
gh api 'repos/itsmontoya/neuralnetwork/contents/docs/handoffs/NN-13.md?ref=main' \
    -H 'Accept: application/vnd.github.raw+json'
```

## Verified baseline access

Both existing benchmark smoke commands passed at the assessed runtime revision:

```sh
GOCACHE=/private/tmp/onnx-runtime-gocache GOMAXPROCS=1 CGO_ENABLED=0 \
    go test -run '^$' -bench '^BenchmarkInference$' -benchtime=1x -benchmem -count=1 .
GOCACHE=/private/tmp/onnx-runtime-gocache GOMAXPROCS=1 CGO_ENABLED=0 \
    go test -tags=purego -run '^$' -bench '^BenchmarkInference$' -benchtime=1x -benchmem -count=1 .
```

Actual environment: Go **1.27.2**, Darwin/arm64, **Apple M3**, `GOMAXPROCS=1`,
`CGO_ENABLED=0`, empty GOFLAGS, default tags followed by `purego`. Both runs
executed all twelve MLP/shape Load/First/Warm subbenchmarks at batches **1 and 3**.
These single-iteration smokes establish executable baseline access only; their
timings and allocations are not a repeated performance report or a comparison.
No Python/reference evaluator ran in these commands; no reference thread
setting applies. No CNN benchmark or parsing-only measurement was added.

Rechecked SHA-256 values match [the existing provenance](../testdata/inference/provenance.json):

| Artifact | SHA-256 |
| --- | --- |
| MLP model | `04df066d0a12b0e6ddf2ba89e9e452768939a745ddd574cfb39780d0e14c06f9` |
| Shape-heavy model | `64800db535d7fd83d30f98d7c30ccf402c00e444a082a9d0865c812ed79b2cb6` |
| Inference provenance | `547933bf4e3be8e5e57e537820a1529822fcf3ddda17d953a83e98eca887ff2c` |

The benchmark does not instrument copied bytes or peak live tensor/workspace
bytes; those metrics are unmeasured, not zero. The runtime retains all value
table entries until Run ends and copies requested outputs. The unchanged
[contract](runtime-contract.md#limits-and-cancellation) accounts conservatively
for aliases, resident constants, defaults, intermediates, destinations and
copying-backend workspace. No liveness reuse, packing, folding or production
optimization was partially implemented.

## Resume conditions and task ordering

1. NN-13 publishes its reproducible MLP/shape baseline and before/after report,
   including actual allocations, copied bytes, latency and peak live storage,
   and its compatible measured optimization. Any consumed backend change must
   have a publicly fetchable exact revision and ownership/workspace contract;
   a verified public implementation/report PR and Go pseudo-version may
   qualify without a newer tag. Do not use a local replacement or unpublished
   adjacent source.
2. Verify its retained-output, immutable-weight, changing-batch, independent
   session, cancellation/failure and normal/purego/cgo-disabled/race evidence.
   Audit this runtime's scratch reservations against any new allocation paths.
3. Resume RT-16's full scope: repeated MLP/CNN/shape load, parsing, first/warm
   execution measurements with complete metadata; conservative consumer-count
   lifetime reuse; and the roadmap's branch/join, subset, alias, default/override,
   batch-change, cancellation, output-lifetime and concurrent-session checks.
   Retain optimization only with measured benefit or a documented memory
   objective. No RT-16 acceptance criterion is satisfied by this blocker report.

RT-17 and RT-19 do not depend on NN-13 or RT-16. After this focused blocker PR,
they may proceed in that order in separate sequential tasks. M1–M4 release
readiness remains as recorded in the [release checklist](release-checklist.md).
No backend task, merge, tag, license decision or broader profile change is part
of this assessment.

## Validation

The two benchmark smoke commands above passed locally, as did all 68
changed-document local links/Markdown anchors, `git diff --check` and
`CGO_ENABLED=0 go run ./tools/capabilitiesgen`. The production/test/module/fixture
diff against RT-18 is empty. The full Go workflow must pass on the final
stacked PR head; its hosted results are
recorded on that PR. This docs-only assessment does not replace RT-16's
unmet numerical, lifetime, concurrency or measurement acceptance checks.
