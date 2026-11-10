# MNIST profiling, second pass

Measured on 2026-11-09, Apple M3, macOS 26.5.2 (25F84), Go 1.27.2,
`GOMAXPROCS=1`, `CGO_ENABLED=0`. Two tensor backend optimizations reduce MNIST
inference latency by about 39% with published `neuralnetwork v0.9.0`.
The earlier prototype against v0.8.0, including its
[patch](neuralnetwork.patch), remains below as historical evidence.

## Published v0.9.0 integration

This checkout now requires published `neuralnetwork v0.9.0`, commit
`d0e5c475b750fea5c453f2ab0437f6c3e708448f`, checksum
`h1:4WNDbwFxz1AVZCVDk60KqBDRtzqFzAgNa+euNttE+2M=`.
There is no workspace or local replacement. The
[v0.9.0 build information](v0.9.0-build.txt) records the published module and
checksum. Transitive dependency versions and the backend license are unchanged.
No ONNX execution code changes; the capability manifest's backend provenance
now identifies the selected release.

Five alternating repetitions of precompiled v0.8.0 and v0.9.0 binaries used
500 ms per MNIST sample and 200 ms per small fixture warm sample. The baseline
was compiled from ONNX commit `8823861325e3218a8b6dea78332d38c6d8ed2df5`
before upgrading the dependency. The new binary uses that source plus this
integration's dependency and provenance changes. Both builds use `GOWORK=off`
and `CGO_ENABLED=0`; measured runs use `GOMAXPROCS=1`. No correctness tests or
other benchmarks ran concurrently with these samples.

| Published backend | MNIST median µs/op [min, max] | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| v0.8.0 | 734.70 [726.60, 768.16] | 179504 | 176 |
| v0.9.0 | 451.01 [448.12, 462.23] | 179440 | 174 |

The median improves 1.63×, a 38.6% latency reduction. Raw logs:
[v0.8.0](v0.8.0-bench.txt), [v0.9.0](v0.9.0-bench.txt).
Small-fixture median changes are within 1%, with identical allocations.

The checked-in ORT comparison driver ran unchanged with `GOWORK=off`, empty
`GOFLAGS`, the same 19 cases, ten warmups and 100 measured calls per case,
one thread per runtime, and disabled ORT graph optimizations:

| Runtime | Median µs | p95 µs |
| --- | ---: | ---: |
| Go with neuralnetwork v0.9.0 | 452.90 | 548.02 |
| ONNX Runtime 1.19.2 | 44.62 | 46.25 |

The median gap is **10.15×**, close to the earlier prototype's 9.86× result.
All 19 numerical and top-1 comparisons pass; the maximum absolute error remains
`1.9073486328125e-5`. All Go output bits match the earlier v0.8.0 results.
Original-versus-converted ORT and upstream expected-output checks also pass.
This remains a single-model, single-host measurement.

[v0.9.0-comparison-summary.json](v0.9.0-comparison-summary.json) retains every
check, benchmark sample, selected backend identity and module-file hashes.
The full report and per-call latencies are in
`.cache/compare-runtime/v0.9.0-report.json`. The full ONNX suite passes in
normal, cgo-disabled, `purego` and race configurations. Build, vet, module
verification/tidiness, formatting, generated capability-table verification,
all four bounded fuzz targets and comparison-tool policy tests pass.
The included backend license matches v0.9.0 byte for byte.

To reproduce the current release measurements from this checkout:

```sh
GOWORK=off GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkMNIST$' -benchtime=500ms -benchmem -count=5 .
GOWORK=off .cache/compare-runtime/venv/bin/python tools/compare_runtime.py --report .cache/compare-runtime/v0.9.0-report.json
```

Prepare the optional Python environment as described in the
[comparison tool instructions](../../../tools/compare_runtime/README.md).

## Prototype sources and measurement

The ONNX source is commit `8823861325e3218a8b6dea78332d38c6d8ed2df5`.
The baseline uses published `github.com/itsmontoya/neuralnetwork v0.8.0`,
checksum `h1:8a3/N9yfGtgqQ+CukFbsJEUqbhLJ1KI92S7B5Iv0x+8=`.
The experiment uses a writable copy of that module in a temporary Go workspace.
No ONNX execution code, module files, model, inputs, timing policy or numerical
tolerances change. [Build information](before-build.txt) and
[experiment build information](after-build.txt) record both resolutions.
[workspace.txt](workspace.txt) records the workspace contents and selected
backend source directory; workspace modules appear as `(devel)` in build info.

[results.json](results.json) records the patch and changed-source hashes,
module-file hashes, individual benchmark samples, reference checks, input/model
hashes, timing summaries and environment. Full comparison reports, including
raw per-call latencies, remain in the ignored cache:
`.cache/compare-runtime/round2-{before,after}.json`.

Precompiled binaries ran sequentially in rotating baseline, broadcast-only,
convolution-only and combined order, five repetitions per variant. MNIST used
500 ms per repetition; the small fixture warm benchmarks used 200 ms. Setup,
model loading, session construction and warmup are outside timing. Each call
returns owned outputs. Profiling and correctness checks ran separately from
latency measurements; no other test or benchmark workload ran concurrently.

## Profile and changes

The [baseline CPU profile](before-cpu.txt) attributes 78.02% of sampled stacks
within `Session.Run` to convolution and 6.23% to broadcast arithmetic. These are
relative percentages after filtering to stacks containing `Session.Run`;
background GC, scheduler and profiler work are outside the denominator.

The patch makes two independent changes:

- Compute four adjacent interior convolution columns together when width
  dilation is one. Each output keeps the original input-channel, kernel-row
  and kernel-column summation order. Sharing weights and geometry exposes four
  independent scalar accumulators. Padding, remaining columns and width
  dilation retain the original scalar path. Bias is still added last. There is
  no weight packing, extra numerical storage, assembly or parallel execution.
- Advance broadcast coordinates and source offsets incrementally. This
  replaces per-element, per-axis division and modulo with increments and axis
  carries. Shape, type and alias validation still finish before writes. Scalar,
  empty and nonfinite behavior remains covered by the existing reference tests.

A preliminary attempt to slice contiguous convolution rows without processing
multiple outputs showed no clear gain in three paired samples and was discarded.

## Repeated Go benchmarks

| Variant | MNIST median µs/op [min, max] | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Published v0.8.0 | 716.88 [696.03, 722.29] | 179504 | 176 |
| Broadcast change | 688.59 [678.78, 692.43] | 179440 | 174 |
| Convolution change | 468.07 [460.53, 471.53] | 179504 | 176 |
| Both changes | 443.12 [439.20, 444.32] | 179440 | 174 |

The combined median is 1.62× faster, a 38.2% latency reduction. Most of the gain
comes from convolution; the allocation reduction is only 64 bytes and two
objects per run. Raw logs: [baseline](before-bench.txt),
[broadcast](broadcast-bench.txt), [convolution](four-bench.txt),
[combined](after-bench.txt).

| Small fixture / Warm | Baseline ns/op | Combined ns/op |
| --- | ---: | ---: |
| mlp_batch_1 | 3498 | 3461 |
| mlp_batch_3 | 3596 | 3632 |
| shape_batch_1 | 6435 | 6399 |
| shape_batch_3 | 6510 | 6511 |
| cnn_batch_1 | 6349 | 6293 |
| cnn_batch_3 | 7464 | 7466 |

Small-fixture allocations are unchanged. Their convolution interiors are too
narrow to use the four-column path. These short samples show little movement;
they do not establish a general speedup across models or machines.

## ONNX Runtime comparison

Both variants use the existing 19 cases, ten warmups and 100 measured calls per
case, single-threaded CPU settings and disabled ORT graph optimizations.
Python is 3.9.6, ONNX 1.16.2, ORT 1.19.2 and NumPy 1.26.4.

| Variant | Go median / p95 µs | ORT median / p95 µs | Go / ORT median |
| --- | ---: | ---: | ---: |
| Published v0.8.0 | 709.81 / 770.72 | 44.12 / 44.88 | 16.09× |
| With proposed backend patch | 440.83 / 504.73 | 44.71 / 45.79 | 9.86× |

All 19 Go outputs are bit-identical between variants. All Go-versus-ORT,
original-versus-converted ORT, upstream expected-output and top-1 checks pass.
The maximum Go-versus-ORT absolute error remains `1.9073486328125e-5`.
The comparison shows a 37.9% Go latency reduction on this model. ORT timing also
varies; the ratio alone is insufficient evidence of a Go improvement.

The baseline runs the checked-in Python driver unchanged. The experiment uses
a temporary copy with `ROOT` fixed to this checkout and its explicit
`GOWORK="off"` changed to the experimental workspace path. Cases, reference
settings, transport, runner and timing logic are identical. Driver hashes are
retained in `results.json`; the checked-in driver is unchanged.

## Validation and remaining targets

The full ONNX suite passes in default and `purego` builds with the proposed
backend, and `go vet ./...` passes. The full backend suite passes; its tensor
suite also passes with `purego`, and `go vet ./tensor` passes.

Fifteen new convolution cases compare against the retained scalar computation,
checking exact finite/infinite/signed-zero bits and matching NaNs. Cases cover
four-column blocks, tails, batches, groups, strides, height dilation, width
dilation fallback, padding, pointwise kernels, nonfinite values, empty spatial
dimensions and sums that expose changes in float32 accumulation order.
Existing tests include 46
convolution reference cases, 40 arithmetic reference cases, destination alias
and argument checks, input immutability and concurrent execution.

The [combined CPU profile](after-cpu.txt) still attributes 78.26% of samples
within Run to convolution. Scalar padding/tail work alone accounts for 47.28%,
making boundary handling a useful next target. The
[baseline](before-alloc.txt) and [combined allocation profiles](after-alloc.txt)
still identify tensor creation, constant clones and ReLU input snapshots as
allocation sources. The profiles measured 8,592 and 13,250 calls respectively;
their raw allocation totals must not be compared as per-call bytes.
Storage ownership changes would need separate budget and lifetime work.

## Reproduce

For the historical prototype, start from an isolated checkout of ONNX baseline
commit `8823861325e3218a8b6dea78332d38c6d8ed2df5`, which pins v0.8.0.
Copy this report's patch into that checkout and prepare the optional MNIST cache
and Python environment using the
[comparison tool instructions](../../../tools/compare_runtime/README.md).
From that checkout's root, create an isolated backend experiment:

```sh
repo_root=$PWD
profile_dir=$(mktemp -d /tmp/onnx-round2.XXXXXX)
backend_module=$(GOWORK=off go list -m -f '{{.Dir}}' github.com/itsmontoya/neuralnetwork)
cp -R "$backend_module" "$profile_dir/neuralnetwork"
chmod -R u+w "$profile_dir/neuralnetwork"
git -C "$profile_dir/neuralnetwork" apply "$repo_root/docs/performance/mnist-round2/neuralnetwork.patch"
(cd "$profile_dir" && go work init "$repo_root" ./neuralnetwork)
GOWORK=off GOMAXPROCS=1 CGO_ENABLED=0 go test -c -o "$profile_dir/before.test" .
GOWORK="$profile_dir/go.work" GOMAXPROCS=1 CGO_ENABLED=0 go test -c -o "$profile_dir/after.test" .
```

Alternate the binaries for five repetitions, with profiling disabled:

```sh
GOMAXPROCS=1 "$profile_dir/before.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=500ms -test.benchmem
GOMAXPROCS=1 "$profile_dir/after.test" -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=500ms -test.benchmem
```

For the attribution variants, restore `tensor/conv2d.go` from v0.8.0 for
broadcast-only, or restore `tensor/arithmetic.go` for convolution-only, before
building their binaries. For small fixtures, use
`-test.bench '^BenchmarkInference$/.*/Warm$' -test.benchtime=200ms`.

Collect profiles separately, following the first pass's
[profiling commands](../mnist/README.md#reproduce-profiling), with
`GOWORK="$profile_dir/go.work"` for the experiment. To compare with ORT, run the
standard driver for the baseline, then use a temporary copy as described above
for the workspace variant; an environment-only `GOWORK` override is insufficient
because the driver explicitly sets it to `off`.

The published v0.9.0 measurements above complete the dependency integration and
repeat the standard comparison without a workspace replacement.
