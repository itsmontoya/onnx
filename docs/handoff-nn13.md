# RT-12 runtime inputs for NN-13

RT-12 implements a working M2 runtime against the unchanged copying backend
`github.com/itsmontoya/neuralnetwork v0.7.0`, published commit
[`6048452c57a0a293db4591cd2e2febd7d6b743a3`](https://github.com/itsmontoya/neuralnetwork/tree/6048452c57a0a293db4591cd2e2febd7d6b743a3).
The [runtime branch](https://github.com/itsmontoya/onnx/tree/rt-12-shape-operators)
contains its source, fixtures, example and baseline benchmarks. The implementation
revision is the commit introducing this handoff (`git log --reverse --format=%H
-- docs/handoff-nn13.md | head -1`); the completed RT-12 roadmap handoff records
that full revision. No local replace or unpublished backend is used.

This supplies NN-13's required runtime inputs. It does not implement NN-13 or
RT-16, alter backend ownership, or claim optimization results. Follow the
[published NN-13 measurement plan](https://github.com/itsmontoya/neuralnetwork/blob/v0.7.0/docs/handoffs/NN-13.md)
for measured copies, peak live memory and before/after comparisons. Baseline
benchmarks below measure latency and Go allocations only; absent copied-byte
and peak-memory instrumentation is not reported as zero.

## Exact fixtures and expected values

[Inference provenance](../testdata/inference/provenance.json) records full
SHA-256 for each model and every named input/output TensorProto. It pins ONNX
1.16.2 at `3bf92c03a9f27eba3bda1e5b9e63ea20ec213557`, Python 3.9.6,
NumPy 1.26.4 and protobuf 4.25.3. The [generator](../tools/check_inference.py)
checks exact versions/full checker acceptance and independently compares actual
ReferenceEvaluator returns with NumPy formulas. Normal Go execution is Python-free.

| Model | SHA-256 | Named I/O |
| --- | --- | --- |
| [MLP](../testdata/inference/models/mlp.onnx) | `04df066d0a12b0e6ddf2ba89e9e452768939a745ddd574cfb39780d0e14c06f9` | `features:float32[batch,2]` → `probabilities:float32[batch,2]` |
| [Shape-heavy](../testdata/inference/models/shape.onnx) | `64800db535d7fd83d30f98d7c30ccf402c00e444a082a9d0865c812ed79b2cb6` | `features:float32[batch,2,3]` → `reshaped:float32[batch,6]`, `target:int64[2]`, `flat:float32[1,6*batch]` |

The provenance SHA-256 is
`547933bf4e3be8e5e57e537820a1529822fcf3ddda17d953a83e98eca887ff2c`.
Its `mlp_batch_1`, `mlp_batch_3`, `shape_batch_1` and `shape_batch_3` cases
point to the named `input_features.pb` and `output_<name>.pb` files under
`testdata/inference/values/<case>/`, with a hash for each. The original RT-11
MLP/branch model/value bytes are unchanged.

MLP inputs are the first batch rows of `[[1,2],[-1,0.5],[0,-2]]`; actual
probability goldens are stored without display rounding. Shape inputs are
row-major 1 through `6*batch`. `reshaped` preserves row order, `target` is
exactly `[batch,-1]`, and `flat` is the row-major flattening of the transposed
`[batch,6]` value. At batch 3 it is
`[1,7,13,2,8,14,3,9,15,4,10,16,5,11,17,6,12,18]` with shape `[1,18]`.
The graph composes Shape → Gather → Unsqueeze → Concat → Reshape → Unsqueeze →
Transpose → Squeeze → Flatten without inferred intermediate value_info.

No repository LICENSE is present. These are original artifacts with no
separately declared license; origin is not a license claim.

## Reproduce and establish a baseline

From the repository root, with Go 1.27.2 installed and dependencies downloaded:

```sh
go mod download
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 3
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 1
CGO_ENABLED=0 go run ./examples/inference -graph shape -batch 3
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
go test -race ./...
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkInference$' -benchmem -count=10 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -tags=purego -run '^$' -bench '^BenchmarkInference$' -benchmem -count=10 .
```

[BenchmarkInference](../inference_benchmark_test.go) uses these exact hashed
MLP/shape fixtures at batches 1 and 3. `Load` times LoadBytes plus model Close;
`First` times a new session, its first Run and Close with an already-loaded
model; `Warm` reuses a warmed session and includes independent output copies.
Fixture decoding/verification stays outside timed loops. Record Go version,
OS/CPU, tags, cgo and GOMAXPROCS with repeated samples. A one-iteration smoke
run passed on Go 1.27.2/Darwin arm64/Apple M3; it is not a performance report.

Optional development regeneration uses the pinned environment:

```sh
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_inference.py
/private/tmp/neuralnetwork-onnx-reference/bin/python tools/check_m2.py
```

[The fixture guide](../testdata/m2/README.md) details exact integer/float-bit
isolated coverage and the NN-07/08 reference exceptions. Integration tests
reuse one session for batches 1 and 3, compare retained outputs after later
runs/Close, reverse requested output order, and run separate sessions
concurrently. Dynamic invalid axes/shapes return contextual errors and recover.

## Allocation and ownership baseline

All backend tensors remain owned and contiguous. Transforms, Gather and Concat
write directly into independent destinations with no extra numerical kernel
scratch. Shape reserves an int64 input-shape snapshot as staging before
FromInt64. Reshape and supplied squeeze/unsqueeze axes reserve their copying
int64 accessor snapshots before access, then reserve the destination while
staging remains charged. The runtime retains intermediates until Run ends and
copies requested outputs before publication. All numerical adapters retain
RT-09's audited copying reservations. See the [runtime contract](runtime-contract.md#limits-and-cancellation).

No public borrowed views, mutable initializer exposure, hidden global scratch,
constant folding, liveness reuse or packed weights are added. Kernels remain
synchronous and cancellation is observed around kernels, never promised inside
them. NN-13 should measure this baseline before selecting a justified bridge or
workspace change; RT-16 consumes those results later.
