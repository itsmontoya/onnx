# RT-16 measurements and storage reuse (2026-11-09)

The measured objective is bounded numeric storage, with one call-local reusable
destination. CNN runs allocate four fewer objects and less numeric payload.
MLP/shape allocation counts and payload copies are unchanged; all runs pay a
32-byte larger evaluation header. Warm latency medians are higher in these
samples, so this is a measured memory tradeoff, without a speedup claim.

## Exact sources and environment

- Working merged baseline: [`4d5a000c20ffae3e053190a4cd04322529851800`](https://github.com/itsmontoya/onnx/commit/4d5a000c20ffae3e053190a4cd04322529851800).
- Before: [`ab619af3501a6705820ee020aadf08160d2fb169`](https://github.com/itsmontoya/onnx/commit/ab619af3501a6705820ee020aadf08160d2fb169), adding the dependency pin, CNN and explicit Parse benchmark phases before changing runtime ownership.
- After: [`fde7144bff7cb8b15b4e6f9a5e2a6c5a2e97abbe`](https://github.com/itsmontoya/onnx/commit/fde7144bff7cb8b15b4e6f9a5e2a6c5a2e97abbe), implementation plus corrected embedded backend provenance. Later report/tool/test additions do not change production execution.
- Both sides use published backend `v0.7.1-0.20261109154142-a64fa687e269`, source [`a64fa687e269c3a1bf3ac345434b46c8ed4a59fc`](https://github.com/itsmontoya/neuralnetwork/commit/a64fa687e269c3a1bf3ac345434b46c8ed4a59fc), without local replacements. [NN-13's report](https://github.com/itsmontoya/neuralnetwork/blob/458b64d18a2e974a1bd51313d040060a85f1f70a/docs/performance/nn13/README.md) establishes its ownership contract and separate backend gains; those gains are already present in this before baseline.
- Go `go1.27.2 darwin/arm64`, module language baseline Go 1.26.1; macOS 26.5.2 (25F84), Apple M3.
- Timings: `GOMAXPROCS=1`, `CGO_ENABLED=0`, default and `purego`; ten 100 ms samples per graph/phase. No other benchmark/test workload ran concurrently with performance samples. Nodes and CPU kernels execute serially.
- [fixtures.json](fixtures.json) records model/input/expected-output paths and SHA-256 hashes, provenance and frozen reference versions for MLP/CNN/shape batches 1 and 3. Fixture loaders verify hashes and compare outputs. Reference runtimes do not execute during these Go measurements, so reference thread settings are inapplicable.

## Storage and ownership decision

The immutable plan counts every input occurrence and groups Identity aliases.
It computes each group's final consumer position for serial execution. Source
inputs, default/initializer clones, Constant nodes and every declared graph
output group stay pinned, including unrequested declared outputs. A completed
node retires other groups only after output validation and the post-node context
check, clearing every alias slot. One retired private tensor is cached; replacing
it releases the previous candidate. A destination can borrow it only with an
exact concrete shape and type match and while all current inputs remain live.
An unusable candidate is dropped before a fresh allocation; scratch reservations
also trim it before rejecting a limit. Remaining cached storage is dropped before
output publication. Public outputs always clone and survive later runs/Close.

No pool survives Run, so changing batches, failure and cancellation cannot retain
hidden session storage. There is no global workspace/cache, parallel scheduler,
constant folding or weight packing. The measured storage benefit does not justify
those additional mechanisms. LayerNormalization's internal temporaries remain
separately reserved; its returned values obey the same conservative outer rules.

Successful reservations include cached tensor bytes until actually dropped or
transferred to a destination. Identity aliases now share a reservation, so some
formerly limited runs succeed. Distinct caller input names may still be counted
conservatively even if callers provide the same tensor. MaxLoadBytes now also
reserves exact-capacity lifetime metadata, construction work and pinned flags:
at most 41 bytes per value plus 8 per step, excluding Go headers/objects. Tight
old load budgets may therefore require more room. CPU products reserve NN-13's
one output-sized staging buffer; the Metal build retains its larger copying
fallback reservations. These are observable budget changes with no public API
or numerical-semantic change.

## Latency and Go allocations

`Parse` is bounded protobuf parsing only. `Load` includes parsing, validation,
compilation and Close. `First` includes NewSession/Run/Close on a loaded model;
`Warm` reuses a warmed session and includes owned output clones. Fixture loading
and comparisons are outside timing. Each latency cell is median [min, max]
ns/op. Allocation cells give medians and ranges when samples varied. Raw logs
are linked for both [default before](before-default.txt)/[after](after-default.txt)
and [purego before](before-purego.txt)/[after](after-purego.txt).

### default

| Case / phase | Before ns/op | After ns/op | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| `mlp_batch_1/Parse` | 3124 [3047, 3906] | 3164 [3071, 3313] | 4736 [4736, 4740] → 4736 [4736, 4739] | 119 → 119 |
| `mlp_batch_1/Load` | 186909.5 [182054, 214567] | 185474 [184770, 187668] | 156513 [156512, 156555] → 157353 [157352, 157400] | 1182 → 1191 |
| `mlp_batch_1/First` | 3588 [3530, 4879] | 3699.5 [3666, 3751] | 4808 → 4840 | 89 → 89 |
| `mlp_batch_1/Warm` | 3544 [3523, 3582] | 3718 [3708, 3724] | 4784 → 4816 | 88 → 88 |
| `mlp_batch_3/Parse` | 3223.5 [3205, 3312] | 3288 [3277, 3323] | 4736 → 4736 | 119 → 119 |
| `mlp_batch_3/Load` | 186500.5 [185660, 187904] | 190663 [190330, 191176] | 156513 [156512, 156527] → 157352 [157352, 157367] | 1182 → 1191 |
| `mlp_batch_3/First` | 3774.5 [3740, 3815] | 3919.5 [3911, 3959] | 5152 → 5184 | 89 → 89 |
| `mlp_batch_3/Warm` | 3775.5 [3765, 3797] | 3905.5 [3815, 3967] | 5128 → 5160 | 88 → 88 |
| `shape_batch_1/Parse` | 5761.5 [5601, 5963] | 5764 [5745, 5829] | 8000 [8000, 8001] → 8000 [8000, 8001] | 208 → 208 |
| `shape_batch_1/Load` | 218796 [213033, 222451] | 217034 [214566, 218526] | 185651 [185651, 185670] → 186979.5 [186979, 186996] | 1508 → 1518 |
| `shape_batch_1/First` | 6278 [6270, 6312] | 6610 [6576, 7691] | 10920 → 10952 | 139 → 139 |
| `shape_batch_1/Warm` | 6308 [6193, 6360] | 6631.5 [6593, 6743] | 10896 → 10928 | 138 → 138 |
| `shape_batch_3/Parse` | 5796 [5613, 5825] | 5819 [5774, 5872] | 8000 → 8000 | 208 → 208 |
| `shape_batch_3/Load` | 217766 [213495, 219840] | 217319 [215882, 218953] | 185651 [185651, 185667] → 186979 [186979, 186995] | 1508 → 1518 |
| `shape_batch_3/First` | 6435 [6416, 6453] | 6717 [6689, 6816] | 11312 → 11344 | 139 → 139 |
| `shape_batch_3/Warm` | 6398.5 [6390, 6435] | 6784.5 [6706, 6808] | 11288 → 11320 | 138 → 138 |
| `cnn_batch_1/Parse` | 6343.5 [6304, 6500] | 6357.5 [6337, 6390] | 9512 → 9512 | 219 → 219 |
| `cnn_batch_1/Load` | 221979.5 [219148, 224831] | 225145.5 [224687, 225374] | 188611 [188611, 188628] → 189907 [189907, 189925] | 1518 → 1528 |
| `cnn_batch_1/First` | 6669.5 [6645, 6694] | 6798 [6746, 7293] | 9312 → 9088 | 131 → 127 |
| `cnn_batch_1/Warm` | 6632 [6611, 6650] | 6862 [6844, 7005] | 9288 → 9064 | 130 → 126 |
| `cnn_batch_3/Parse` | 6380 [6306, 6424] | 6357.5 [6310, 6384] | 9512 → 9512 | 219 → 219 |
| `cnn_batch_3/Load` | 224059.5 [221393, 224588] | 223790 [222239, 226404] | 188611 [188611, 188628] → 189907 [189907, 189924] | 1518 → 1528 |
| `cnn_batch_3/First` | 8105.5 [8075, 8147] | 8181.5 [8148, 8272] | 11904 → 11424 | 131 → 127 |
| `cnn_batch_3/Warm` | 8037 [7924, 8154] | 8166.5 [8157, 8245] | 11880 → 11400 | 130 → 126 |

### purego

| Case / phase | Before ns/op | After ns/op | B/op before → after | allocs/op before → after |
| --- | ---: | ---: | ---: | ---: |
| `mlp_batch_1/Parse` | 3301.5 [3257, 4031] | 3260.5 [3248, 3987] | 4736 [4736, 4740] → 4736 [4736, 4740] | 119 → 119 |
| `mlp_batch_1/Load` | 192022.5 [190364, 220743] | 191758.5 [191216, 192007] | 156513 [156512, 156559] → 157353 [157352, 157399] | 1182 → 1191 |
| `mlp_batch_1/First` | 3604 [3592, 3654] | 3691 [3680, 4252] | 4808 → 4840 | 89 → 89 |
| `mlp_batch_1/Warm` | 3568.5 [3545, 3586] | 3677 [3666, 3734] | 4784 → 4816 | 88 → 88 |
| `mlp_batch_3/Parse` | 3280 [3270, 3441] | 3254.5 [3235, 3279] | 4736 → 4736 | 119 → 119 |
| `mlp_batch_3/Load` | 192654 [190050, 198248] | 192376 [191876, 193739] | 156513 [156512, 156528] → 157353 [157352, 157367] | 1182 → 1191 |
| `mlp_batch_3/First` | 3740 [3729, 3750] | 3829.5 [3821, 3855] | 5152 → 5184 | 89 → 89 |
| `mlp_batch_3/Warm` | 3720 [3716, 3752] | 3806 [3797, 3837] | 5128 → 5160 | 88 → 88 |
| `shape_batch_1/Parse` | 5680 [5637, 5701] | 5646.5 [5600, 5659] | 8000 [8000, 8001] → 8000 [8000, 8001] | 208 → 208 |
| `shape_batch_1/Load` | 217139 [216520, 219674] | 219399.5 [217160, 220692] | 185651.5 [185651, 185668] → 186979 [186979, 186997] | 1508 → 1518 |
| `shape_batch_1/First` | 6200.5 [6186, 6392] | 6551.5 [6500, 6627] | 10920 → 10952 | 139 → 139 |
| `shape_batch_1/Warm` | 6166 [6140, 6296] | 6492.5 [6474, 6577] | 10896 → 10928 | 138 → 138 |
| `shape_batch_3/Parse` | 5631.5 [5624, 5739] | 5698.5 [5617, 5936] | 8000 → 8000 | 208 → 208 |
| `shape_batch_3/Load` | 215567.5 [215042, 215970] | 216824 [216308, 220448] | 185651 [185651, 185668] → 186979 [186979, 186997] | 1508 → 1518 |
| `shape_batch_3/First` | 6357.5 [6280, 6409] | 6708.5 [6578, 6759] | 11312 → 11344 | 139 → 139 |
| `shape_batch_3/Warm` | 6311 [6243, 6328] | 6631 [6556, 6665] | 11288 → 11320 | 138 → 138 |
| `cnn_batch_1/Parse` | 6297 [6283, 6776] | 6310 [6200, 6336] | 9512 → 9512 | 219 → 219 |
| `cnn_batch_1/Load` | 221873 [219284, 223426] | 223314.5 [222829, 224338] | 188611.5 [188611, 188628] → 189907 [189907, 189925] | 1518 → 1528 |
| `cnn_batch_1/First` | 6619 [6587, 6639] | 6700 [6680, 6823] | 9312 → 9088 | 131 → 127 |
| `cnn_batch_1/Warm` | 6584 [6546, 6684] | 6775 [6718, 6840] | 9288 → 9064 | 130 → 126 |
| `cnn_batch_3/Parse` | 6223.5 [6213, 6310] | 6285.5 [6273, 6393] | 9512 → 9512 | 219 → 219 |
| `cnn_batch_3/Load` | 223211 [220886, 224567] | 222472.5 [221230, 242454] | 188611 [188611, 188628] → 189907 [189907, 189924] | 1518 → 1528 |
| `cnn_batch_3/First` | 7925 [7912, 7969] | 8096 [8062, 8328] | 11904 → 11424 | 131 → 127 |
| `cnn_batch_3/Warm` | 7912.5 [7892, 8063] | 8138 [8043, 8186] | 11880 → 11400 | 130 → 126 |

CNN First/Warm allocation counts fall by four; Warm B/op falls 9288→9064 at
batch 1 and 11880→11400 at batch 3 (224/480 bytes). MLP/shape keep their run
allocation counts with +32 B/run bookkeeping. Loading adds lifetime metadata;
Parse is unaffected by the ownership algorithm.

Default Warm latency medians rise 1.6–6.0% across these six cases, including
5.1/6.0% for shape batches 1/3. Purego Warm medians rise 2.3–5.3%. These short
single-host samples are not a controlled cross-machine result or proof of a
universal slowdown, but they do not establish a latency improvement. Retention
is justified by the explicit CNN allocation/payload/peak objective below. No
universal zero-allocation or speedup target is imposed.

## Copies, numeric payload and reservation high-water

[Before](before-payload.json)/[after](after-payload.json) are separate serial
observations, reproduced exactly for all 48 graph/phase records. They use the
published NN-13 observer from backend tooling revision
`458b64d18a2e974a1bd51313d040060a85f1f70a`, extended by this task's isolated
measurement harness and explicit reservation/reuse hooks.

- Copied bytes count executed numeric payload `copy` lengths and scalar transpose materialization assignments. Arithmetic writes, shape/stride metadata and wire decoding are excluded. Copies do not decrease in these fixtures; the optimization reuses destination storage.
- Newly allocated numeric payload counts actual registered float32/int64 backing capacity, excluding preexisting roots. Aliases count once. This is separate from Go benchmark B/op and allocator rounding.
- Function-scoped peak payload includes constants, caller inputs, runtime destinations, accessor/Matrix/product temporaries and one retained previous Warm output set. Nested function returns/receiver roots preserve ownership. Scope exit releases unreturned locals. It is a conservative logical lifetime upper bound, potentially retaining locals beyond compiler last use; metadata, wire/protobuf objects, allocator overhead, observer memory, exact GC liveness and RSS are excluded.
- Runtime reservation peak records the actual successful `runBudget.reserve` high-water (live + workspace), including constants, bound sources, destinations, cached storage and current conservative scratch. It observes explicit release/reuse and excludes outputs retained from earlier runs, as the public budget specifies. Workspace peak is its scratch component. These bounds can exceed actual temporary storage. The before baseline still used old copying-product scratch charges despite consuming NN-13, so decreases caused by the audit alone are accounting corrections.
- The function-scoped observer does not prove intra-Run last-use reclamation. Shape's lower reservation peak must not be presented as a measured reduction in its function-scoped payload bound. CNN also has real allocation/payload and observer-peak reductions.
- Parse's numeric counters are zero for these raw-data fixtures: wire/protobuf memory is outside this observer, not zero parser memory/copying. Load's numeric conversions are counted. Generated protobuf typed-numeric allocation paths are not exercised or claimed covered by these fixtures.

| Case / phase | Copied bytes | New numeric payload | Function-scoped peak payload | Reservation peak | Workspace peak | Destinations reused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cnn_batch_1/First` | 1036 → 1036 | 1464 → 1336 | 1192 → 1064 | 1576 → 1448 | 768 → 768 | 0 → 1 |
| `cnn_batch_1/Warm` | 1036 → 1036 | 1464 → 1336 | 1236 → 1108 | 1576 → 1448 | 768 → 768 | 0 → 1 |
| `cnn_batch_3/First` | 2748 → 2748 | 4032 → 3648 | 2856 → 2472 | 4008 → 3624 | 2304 → 2304 | 0 → 1 |
| `cnn_batch_3/Warm` | 2748 → 2748 | 4032 → 3648 | 2988 → 2604 | 4008 → 3624 | 2304 → 2304 | 0 → 1 |
| `mlp_batch_1/First` | 168 → 168 | 208 → 208 | 204 → 204 | 240 → 240 | 72 → 72 | 0 → 0 |
| `mlp_batch_1/Warm` | 168 → 168 | 208 → 208 | 212 → 212 | 240 → 240 | 72 → 72 | 0 → 0 |
| `mlp_batch_3/First` | 368 → 368 | 488 → 488 | 340 → 340 | 448 → 448 | 216 → 216 | 0 → 0 |
| `mlp_batch_3/Warm` | 368 → 368 | 488 → 488 | 364 → 364 | 448 → 448 | 216 → 216 | 0 → 0 |
| `shape_batch_1/First` | 312 → 312 | 336 → 336 | 328 → 328 | 328 → 216 | 24 → 24 | 0 → 0 |
| `shape_batch_1/Warm` | 312 → 312 | 336 → 336 | 392 → 392 | 328 → 216 | 24 → 24 | 0 → 0 |
| `shape_batch_3/First` | 648 → 648 | 672 → 672 | 712 → 712 | 712 → 456 | 24 → 24 | 0 → 0 |
| `shape_batch_3/Warm` | 648 → 648 | 672 → 672 | 872 → 872 | 712 → 456 | 24 → 24 | 0 → 0 |

All byte cells are before→after. CNN reuses one destination per Run, saving
128/384 newly allocated numeric bytes at batches 1/3. Its function-scoped peak
falls by the same amount, 1236→1108 /2988→2604 for Warm. Shape's reservation
peak falls 328→216 /712→456, with unchanged function-scoped peaks. MLP has
no destination reuse in this graph and unchanged payload/copy/high-water values.
Load/Parse observations, including their distinct excluded-memory scope, remain
in the JSON; no reservation budget exists during parsing/loading.

## Allocation profile cross-check

Rate-1 `alloc_space` profiles separately cover `cnn_batch_3/Warm` for one second.
Published [before](before-alloc.txt)/[after](after-alloc.txt) summaries and
[iteration counts before](before-profile.txt)/[after](after-profile.txt) must be
normalized because iteration counts differ (8107 versus 8224). Approximate
recurring allocations per measured iteration are:

| Site | Before B/run | After B/run |
| --- | ---: | ---: |
| tensor.newTensor | 4160 | 3648 |
| evaluation + value table | 240 | 272 |
| Net | | 480 fewer |

The 512-byte destination/object reduction minus 32-byte evaluation overhead
matches unprofiled B/op. Numeric payload drops by 384 bytes; allocator rounding,
shape and Tensor/identity objects explain the distinction. Profiling overhead
makes these profiles unsuitable as latency samples. The profile's setup totals
are not claimed as per-run live storage.

## Reproduce

Use isolated clean runtime checkouts for the exact before/after commits above.
Both already require the published backend pseudo-version, without replacements.
Run sequentially on each source and save separate logs:

```sh
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkInference$' -benchtime=100ms -benchmem -count=10 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -tags=purego -run '^$' -bench '^BenchmarkInference$' -benchtime=100ms -benchmem -count=10 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkInference/cnn_batch_3/Warm$' -benchtime=1s -memprofile=/tmp/rt16.pprof -memprofilerate=1 -o /tmp/rt16.test .
go tool pprof -top -nodecount=30 -alloc_space /tmp/rt16.pprof
```

For payload/high-water observations, use this task's tooling with both Git
repositories available, including the exact backend tooling commit:

```sh
python3 tools/rt16/measure.py --runtime . --runtime-revision ab619af3501a6705820ee020aadf08160d2fb169 --backend /path/to/neuralnetwork --output /tmp/before-payload.json
python3 tools/rt16/measure.py --runtime . --runtime-revision fde7144bff7cb8b15b4e6f9a5e2a6c5a2e97abbe --backend /path/to/neuralnetwork --output /tmp/after-payload.json
```

The Python tool resolves commits to exact hashes, archives pinned source into a
temporary directory, applies the published Go AST observer only there, adds the
explicit budget/reuse hooks, and uses a temporary Go workspace. It never patches
production checkouts or module-cache files. The observation-only overlay is
separate from the unmodified modules used for timing and correctness checks.
Python 3/Go/Git/tar are development requirements; ONNX/ORT Python is not needed.

## Validation and limits

The focused suite verifies duplicate consumers, alias-group retirement, all
alias slots cleared, graph-output pinning, matching/different dtype and empty
shapes, candidate replacement, trim-before-destination/scratch limits, defaults
and overrides, branch/join numerics, changing/empty batches, retained output
independence, every cancellation boundary, late-failure recovery and independent
concurrent sessions. A zero-contraction MatMul overwrites a nonzero reused
destination with positive zeros. Existing conformance, normalization, external
weights, resource and numerical-edge tests remain applicable.

Full Go verification, bounded fuzzing, relevant examples, local links and hosted
CI are recorded in [the RT-16 handoff](../../handoff-rt16.md). No new operator,
version profile, public API, release tag or optimization PR merge is included.
The cache is deliberately call-local, so there is no promise of zero allocation
or reuse across separate Runs, and no hard process-memory bound.
