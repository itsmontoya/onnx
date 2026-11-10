# MNIST profiling and first optimizations

The [second profiling pass](../mnist-round2/README.md) records the subsequent
published v0.9.0 integration, about 39% lower MNIST latency and a 10.15× ORT
median gap. The v0.8.0 measurements below remain historical evidence.

Measured on 2026-11-09, Apple M3, macOS arm64, Go 1.27.2,
`GOMAXPROCS=1`, `CGO_ENABLED=0`. The baseline is ONNX commit
`ded47707065787c819849e7c33fae592f2d878d6`, with the new benchmark added.
The initial profiling used
`github.com/itsmontoya/neuralnetwork v0.7.1-0.20261109154142-a64fa687e269`.

## Published v0.8.0 follow-up

The first optimization pass upgraded to published `neuralnetwork v0.8.0`, commit
`743e049ca9e4877de3c33c0969b17119b6f1491d`, which includes the convolution and
direct ReLU changes below. The standard comparison driver ran unchanged, with
`GOWORK=off` and no local replacement. The ONNX ReLU snapshot optimization is
also included. The [binary build information](v0.8.0-build.txt) records the
actual module version and checksum.

The same 19 cases, ten warmups and 100 measured calls per case, single-threaded
CPU settings and disabled ORT graph optimizations produced:

| Runtime | Median µs | p95 µs |
| --- | ---: | ---: |
| Go with neuralnetwork v0.8.0 | 689.58 | 772.90 |
| ONNX Runtime 1.19.2 | 41.54 | 44.21 |

The median gap is **16.60×**, compared with the original roughly 28× result.
Go's median is about 1.79× faster than the initial 1231.27 µs comparison in this
report. All 19 numerical and top-1 checks pass, with maximum absolute error
`1.9073486328125e-5`. Original-versus-converted ORT and upstream expected-output
checks also pass.

An additional five alternating repetitions of the original baseline binary and
the v0.8.0 binary, each 500 ms with profiling disabled, measured:

| Variant | MNIST median µs/op [min, max] | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Original baseline | 1202.08 [1174.65, 1211.59] | 384112 | 196 |
| ONNX + neuralnetwork v0.8.0 | 701.24 [688.73, 869.08] | 179504 | 176 |

The paired Go benchmark shows about 1.71× faster execution and 53.3% fewer
allocated bytes. Raw logs: [baseline](v0.8.0-baseline-bench.txt) and
[v0.8.0](v0.8.0-bench.txt). The
[comparison summary](v0.8.0-comparison-summary.json) retains all checks,
versions, backend identity, module-file hashes and timing summaries. The full
report and raw per-call latencies are in
`.cache/compare-runtime/v0.8.0-report.json`.

The full ONNX suite passes with v0.8.0 in default and `purego` builds.
`go vet ./...`, `go mod verify` and `go mod tidy -diff` pass. The included backend
license still matches the published module byte for byte. The earlier temporary
patch measurements below are retained as historical evidence.

## Initial profiling

Convolution is the main latency bottleneck: **84.4% of CPU samples within
Session.Run** in the [baseline profile](before-cpu.txt). The scalar inner loop
recomputes input/weight offsets and checks horizontal padding for each multiply.
The [allocation profile](before-alloc.txt) also identifies ReLU's matrix adapters
and ONNX's NaN correction copies as substantial allocation sources.

The CPU summaries filter to stacks containing `Session.Run` and use relative
percentages. Background GC, scheduler and profiler stacks are outside that
denominator. These percentages describe sampled stacks, not total wall time.

## Changes

The ONNX change materializes ReLU's output snapshot and corrected tensor only
when an input NaN requires correction. Other inputs return the kernel's existing
destination. NaN propagation, negative-zero handling, cancellation checks,
output ownership and the conservative six-times-input workspace charge remain
covered by the existing contract. Regression tests check destination reuse,
multiple NaNs, infinities, negative zero, empty tensors and input immutability.

The proposed [neuralnetwork patch](neuralnetwork.patch) has two changes:

- Hoist convolution plane/row offsets and clip valid kernel columns once per
  output. Advance offsets in the inner loop. Keep the original float32
  accumulation order and skip padded products, including nonfinite weights.
- Write ReLU directly into its checked destination, removing three numerical
  matrix adapter buffers. Preserve the backend's NaN/negative-zero-to-positive-
  zero behavior; ONNX still applies its own NaN correction when needed.

The initial backend patch was applied and tested in an isolated temporary copy
of the old pinned module. That experiment preceded the published v0.8.0 release;
the follow-up above measures the integrated release.

## Repeated Go benchmarks

Three precompiled test binaries ran sequentially in rotating baseline, ONNX,
backend order, five repetitions each, with profiling disabled. MNIST used
500 ms per repetition; the small fixtures used 200 ms. The MNIST input is the
fixed `[1,1,28,28]` tensor `float32(index % 256) / 255`; setup, model loading,
session creation and warmup are outside timing. Each timed call owns its outputs.

| Variant | MNIST median µs/op [min, max] | B/op | allocs/op |
| --- | ---: | ---: | ---: |
| Baseline | 1227.89 [1212.35, 1343.11] | 384112 | 196 |
| ONNX ReLU change | 1226.12 [1198.28, 1234.71] | 302192 | 186 |
| ONNX + proposed backend patch | 703.90 [697.84, 747.25] | 179504 | 176 |

The ONNX change saves 81,920 B/run (21.3%) and ten allocations, with little
measured latency change. With the backend proposal, latency improves about
1.74× and allocated bytes fall 53.3%. Raw samples:
[baseline](before-bench.txt), [ONNX](onnx-bench.txt), [backend](backend-bench.txt).

| Small fixture / Warm | Baseline ns/op | ONNX ns/op | With backend patch ns/op |
| --- | ---: | ---: | ---: |
| mlp_batch_1 | 3705 | 3615 | 3466 |
| mlp_batch_3 | 3871 | 3827 | 3617 |
| shape_batch_1 | 6428 | 6492 | 6427 |
| shape_batch_3 | 6579 | 6587 | 6502 |
| cnn_batch_1 | 6660 | 6525 | 6373 |
| cnn_batch_3 | 8019 | 7854 | 7443 |

These short, single-host samples are directional. Shape's unchanged code and
allocations provide a useful noise check. Full B/op, allocation counts and
individual timing observations are in the logs.

## ONNX Runtime comparison

The existing comparison used all 19 cases, ten warmups and 100 measured calls
per case, with ORT graph optimizations disabled and one thread per runtime.

| Variant | Go median / p95 µs | ORT median / p95 µs | Go / ORT median |
| --- | ---: | ---: | ---: |
| Baseline | 1231.27 / 1336.98 | 45.38 / 47.50 | 27.14× |
| ONNX ReLU change | 1226.77 / 1308.75 | 44.54 / 45.71 | 27.54× |
| ONNX + proposed backend patch | 691.12 / 792.94 | 41.58 / 44.67 | 16.62× |

All variants pass 19/19 comparisons and top-1 checks, with maximum absolute
error `1.9073486328125e-5`. Original-versus-converted ORT and the bundled
upstream output checks also pass. The [comparison summary](comparison-summary.json)
records each check, model hash, versions, timing summaries and backend patch
hash. Full reports and raw per-call latencies remain in the ignored
`.cache/compare-runtime/profile-{before,onnx,backend}.json` files.

The experimental backend comparison used a temporary copy of the Python driver
with its root fixed to this checkout and its explicit `GOWORK=off` changed to
the temporary workspace path. An experiment label was added to its report.
Timing, cases, tolerances, ORT settings and the Go runner were unchanged.
The checked-in comparison driver retains its published-module behavior.

ORT also varies between runs; compare the Go latencies as well as their ratios.
These initial results projected a reduction from roughly 28× to 17× on this
network. The v0.8.0 follow-up above confirms that reduction with the published
backend; it remains a single-model comparison.

## Reproduce profiling

Prepare the optional cached MNIST model with the
[existing comparison tool](../../../tools/compare_runtime/README.md). Go
benchmarks skip it when absent and never download a model.

```sh
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkMNIST$' -benchtime=500ms -benchmem -count=5 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkInference$/.*/Warm$' -benchtime=200ms -benchmem -count=5 .
GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkMNIST$' -benchtime=3s -cpuprofile=/tmp/mnist-cpu.pprof -memprofile=/tmp/mnist-mem.pprof -o /tmp/mnist.test .
go tool pprof -top -relative_percentages -focus='onnx.\(\*Session\).Run' /tmp/mnist.test /tmp/mnist-cpu.pprof
go tool pprof -top -alloc_space /tmp/mnist.test /tmp/mnist-mem.pprof
```

To measure the proposed backend without changing this repository's module files,
run from the repository root:

```sh
repo_root=$PWD
profile_dir=$(mktemp -d /tmp/onnx-profile.XXXXXX)
GOWORK=off go mod download github.com/itsmontoya/neuralnetwork@v0.7.1-0.20261109154142-a64fa687e269
backend_module=$(GOWORK=off go list -m -f '{{.Dir}}' github.com/itsmontoya/neuralnetwork@v0.7.1-0.20261109154142-a64fa687e269)
cp -R "$backend_module" "$profile_dir/neuralnetwork"
chmod -R u+w "$profile_dir/neuralnetwork"
git -C "$profile_dir/neuralnetwork" apply "$repo_root/docs/performance/mnist/neuralnetwork.patch"
(cd "$profile_dir" && go work init "$repo_root" ./neuralnetwork)
GOWORK="$profile_dir/go.work" GOMAXPROCS=1 CGO_ENABLED=0 go test -run '^$' -bench '^BenchmarkMNIST$' -benchtime=500ms -benchmem -count=5 .
GOWORK="$profile_dir/go.work" GOMAXPROCS=1 CGO_ENABLED=0 go test ./...
(cd "$profile_dir/neuralnetwork" && GOWORK=off CGO_ENABLED=0 go test ./tensor)
```

For the original baseline, use an isolated checkout of the baseline commit,
copy in `mnist_benchmark_test.go` and the cached converted model, then build its
test binary before rotating the three binaries. The small fixtures are already
tracked. Run profile collection separately from latency benchmarks.

## Validation and next targets

The full ONNX suite passes in default and `purego` builds with both the published
backend and the proposed patch. The backend tensor suite passes in both builds,
including its 46 convolution reference cases and the patch's new padding tests.
The new tests cover dilated/strided asymmetric padding, windows outside the
input, dilation skipping a narrow input, nonfinite padding and empty widths.
The patch also passes a reverse application check against the tested source.

The [backend CPU profile](backend-cpu.txt) still attributes 80% of samples within
Run to convolution, followed by broadcasting at about 9%. Contiguous-row/SIMD
convolution and cheaper broadcast indexing are the next measured targets.
Weight snapshot reuse may reduce allocation further, but requires separate
ownership and budget design work.

[Backend allocation summary](backend-alloc.txt) and
[profile iteration count](backend-profile.txt) are also retained. The original
profile measured 2,536 calls and the backend profile 4,984; their raw allocation
totals must not be compared as per-call bytes. Use the unprofiled B/op above.
