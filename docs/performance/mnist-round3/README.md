# MNIST profiling, third pass

The next measured win is in `neuralnetwork/tensor/conv2d.go`. An isolated
prototype reduces end-to-end MNIST inference from **433.08 to 318.64 µs**,
**26.4% lower latency / 1.36× faster**, with unchanged allocations. The second
convolution supplies most of the improvement.

These are investigation artifacts. The ONNX implementation and dependency
files are unchanged. The [prototype patch](neuralnetwork-experiment.patch)
needs production review and broader performance coverage in the backend
repository. The [agent prompt](AGENT_PROMPT.md) describes that follow-up.

## Environment and method

Measured on 2026-11-09, Apple M3, macOS arm64, Go 1.27.2,
`GOMAXPROCS=1`, `CGO_ENABLED=0`, `GOARM64=v8.0`. ONNX revision:
`b3e2cc8078ece856bfd8296e07f8b383458bfb63`.
The baseline is published `github.com/itsmontoya/neuralnetwork v0.9.0`,
checksum `h1:4WNDbwFxz1AVZCVDk60KqBDRtzqFzAgNa+euNttE+2M=`.
The experiment uses a writable copy of that release in a temporary Go
workspace. The sibling backend checkout was not modified.

Three precompiled binaries ran sequentially, in rotating order, seven
repetitions each, with 700 ms per sample. Only the experimental backend differs
in inference behavior. The experimental binaries additionally contain the
separate diagnostic benchmark retained in [probe_test.go.txt](probe_test.go.txt).
Model loading, session creation and warmup are outside timing; each inference
returns owned outputs. No other tests or benchmarks ran concurrently with the
final rotating measurements. CPU and allocation profiles were collected in
separate runs; their slower benchmark timings are not used for speedup claims.

## Where time goes

The [baseline CPU profile](cpu.txt) includes only stacks containing
`Session.Run`, with percentages relative to those stacks:

| Path | Cumulative share of sampled Run stacks |
| --- | ---: |
| `Conv2DInto` | 79.40% |
| Scalar `conv2DValue` | 46.28% |
| Interior `conv2DFourValues` | 28.53% |
| `AddInto` | 6.02% |
| `MaxPool2DInto` | 4.91% |
| ONNX `patchRelu` | 2.69% |

Rows overlap: the scalar and four-column helpers are included in convolution.
The [unfiltered profile](cpu-full.txt) contains many `runtime.kevent` samples
and background scavenging work. Its blocking/runtime stacks are outside this
denominator; these percentages are not percentages of wall-clock latency.
The isolated kernel measurements below independently confirm the convolution
bottleneck. [Line attribution](cpu-lines.txt) and
[generated scalar assembly](scalar-objdump.txt) show that the hot scalar loop
already uses scalar fused multiply-add, while retaining two payload bounds
checks and one dependent accumulator per inner iteration.

Both convolutions use stride 1, dilation 1, groups 1 and padding 2 on all sides:

| Node | Input NCHW | Weights | Output NCHW | Columns using scalar path per row |
| --- | --- | --- | --- | ---: |
| 0 | `[1,1,28,28]` | `[8,1,5,5]` | `[1,8,28,28]` | 4/28 |
| 4 | `[1,8,14,14]` | `[16,8,5,5]` | `[1,16,14,14]` | 6/14 |

The second layer runs four-column blocks at output columns 2–5 and 6–9.
Columns 0–1 and 12–13 are padding boundaries; columns 10–11 are interior
remainders. All six use the scalar helper. This layer has eight input channels
per filter, making those scalar columns particularly expensive. Vertical
padding does not by itself disable the existing four-column helper.

## Prototype and measurements

The first experiment computes four output channels together, sharing input
samples, clipping and geometry across four independent accumulators. The
second combines that approach at boundaries and remainders with v0.9.0's
four-spatial-column helper in the interior. Channel blocks stay inside their
group. Each output retains input-channel, kernel-row and kernel-column
summation order, and bias is added last. There is no packing, assembly,
parallel execution, scratch payload, or new API.

| Variant | MNIST median µs/op [min, max] | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Published v0.9.0 | 433.08 [424.71, 443.10] | 179440 | 174 |
| Four-channel blocks throughout | 346.59 [334.26, 353.44] | 179440 | 174 |
| Channel blocks at edges/remainders; spatial blocks inside | 318.64 [311.87, 325.65] | 179440 | 174 |

Raw samples: [baseline](baseline-bench.txt), [channels](channels-bench.txt),
[combined approach](hybrid-bench.txt). All seven combined samples are faster
than all seven baseline samples. The [post-change profile](hybrid-cpu.txt)
attributes 71.83% of sampled Run stacks to convolution; the old scalar helper
no longer appears among its substantial costs.

Five alternating measurements of each variant's six small fixture warm
benchmarks show median changes from -1.02% to +0.44%, with identical
allocations. Those short measurements show no material regression on these
fixtures; they do not cover broader convolution dispatch choices. Raw samples:
[baseline](small-baseline-bench.txt), [prototype](small-hybrid-bench.txt).

Isolated node benchmarks reuse real intermediate tensors from the same MNIST
input as `BenchmarkMNIST`. They include the existing kernel validation and
ONNX activation correction, but exclude session dispatch and tensor allocation
outside the call. They are diagnostic measurements rather than an additive
end-to-end latency decomposition. Baseline medians use five 400 ms samples;
experimental medians use three 400 ms samples.

| Node/kernel | Baseline µs/op | Prototype µs/op |
| --- | ---: | ---: |
| First Conv | 59.48 | 48.37 |
| Second Conv | 284.52 | 181.02 |
| First Add | 17.69 | 17.64 |
| Second Add | 8.54 | 8.53 |
| First ReLU including ONNX correction | 10.87 | 10.88 |
| Second ReLU including ONNX correction | 5.76 | 5.87 |
| First MaxPool | 15.89 | 15.84 |
| Second MaxPool | 4.74 | 4.76 |
| MatMul | 1.72 | 1.69 |

The second convolution improves about 36%. Raw node logs:
[baseline](nodes-bench.txt), [prototype](hybrid-nodes-bench.txt).

## Allocation and follow-up priorities

The [allocation profile](alloc-space.txt), collected separately with
`-test.memprofilerate=1`, attributes 71.18% of allocated bytes to tensor
construction and 22.74% to `Float32Values`. Both ReLU input snapshots together
allocate 40,832 bytes per inference. Constant clones and intermediate tensors
remain visible. Removing those allocations requires ownership, lifetime,
NaN/cancellation, and runtime-budget analysis in ONNX; it does not address the
largest measured computation cost.

After convolution, channel-bias broadcast is the next backend candidate:
the two large Add nodes cost about 26 µs together. The current traversal handles
general broadcasting for shapes `[1,8,28,28] + [8,1,1]` and
`[1,16,14,14] + [16,1,1]`. A checked contiguous channel-bias path could be useful,
but has not been prototyped here. MatMul is too small on this model to be the
next priority. Specialized pooling and ONNX's ReLU snapshot are smaller targets.

## Correctness and limitations

The full backend suite and its `purego` tensor suite pass with the combined
prototype. The full ONNX suite passes against that workspace in both default
and `purego` builds. The patch adds 400 scalar-reference cases spanning group
boundaries, filter counts 4/5/7/8, remainder channels, batches, asymmetric
padding, strides, dilation, empty dimensions and nonfinite values. Existing
backend tests additionally exercise bias-free calls, aliases, invalid
destinations, immutability and concurrency.

End-to-end outputs are bit-identical to published v0.9.0 for 131 inputs:
the upstream vector, zeros, ones, 64 seeded uniform `[0,1]` tensors and 64
seeded uniform `[-10,10]` tensors. These random probes use Python's
`random.Random(20261109)` and differ from the comparison driver's NumPy cases.
The upstream result also passes the archived Model Zoo expected output at the
existing `atol=1e-4`, `rtol=1e-5` tolerances, with maximum absolute error
`3.0517578125e-5` and top-1 class 3. See
[validation summary](validation-summary.json) and [results.json](results.json).

This experiment did not rerun ORT: the optional Python ONNX/ORT environment
was unavailable. Applying the measured 0.736 latency factor to the earlier
10.15× gap suggests roughly **7.5×**, but that is an estimate, not a measured
new ORT comparison. A fresh comparison is required after backend integration.

The prototype is not a complete dispatch policy: for groups with four or more
filters, leftover filters currently use the scalar helper. Production work
should retain the existing spatial optimization for those filters and measure
wide, narrow, grouped, depthwise and dilated workloads before choosing dispatch
thresholds. Cross-architecture performance and assembly/SIMD were not tested.

## Reproduce

Prepare the cached MNIST fixture using the existing
[comparison instructions](../../../tools/compare_runtime/README.md). With a
clean checkout of the revision above, create an isolated backend copy:

```sh
repo_root=$PWD
profile_dir=$(mktemp -d /tmp/onnx-profile-round3.XXXXXX)
backend_module=$(GOWORK=off go list -m -f '{{.Dir}}' github.com/itsmontoya/neuralnetwork)
cp -R "$backend_module" "$profile_dir/neuralnetwork"
chmod -R u+w "$profile_dir/neuralnetwork"
cp docs/performance/mnist-round3/probe_test.go.txt round3_probe_test.go
GOWORK=off CGO_ENABLED=0 go test -c -o "$profile_dir/baseline.test" .
git -C "$profile_dir/neuralnetwork" apply "$repo_root/docs/performance/mnist-round3/neuralnetwork-experiment.patch"
(cd "$profile_dir" && go work init "$repo_root" ./neuralnetwork)
GOWORK="$profile_dir/go.work" CGO_ENABLED=0 go test -c -o "$profile_dir/hybrid.test" .
rm round3_probe_test.go
```

Alternate these binaries, seven repetitions each, with no profiling enabled:

```sh
GOMAXPROCS=1 "$profile_dir/baseline.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=700ms -test.benchmem
GOMAXPROCS=1 "$profile_dir/hybrid.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=700ms -test.benchmem
```

For node measurements use `-test.bench '^BenchmarkMNISTNodes$'`. Collect CPU
and allocation profiles in separate runs, using `-test.cpuprofile=...` or
`-test.memprofile=... -test.memprofilerate=1`. The retained
[probe](probe_test.go.txt) supplies kernel shapes in its benchmark logs.
Raw binaries, workspaces and `.pprof` files from this investigation are retained
locally in `/private/tmp/onnx-profile-round3/`.
