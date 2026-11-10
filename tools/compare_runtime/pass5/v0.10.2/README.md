# neuralnetwork v0.10.2 integration — 2026-11-10

Upgrading published neuralnetwork **v0.10.1 → v0.10.2** reduces unchanged MNIST
benchmark latency from **200.790 to 166.291 µs/op**, **17.2% lower latency / 1.21×
faster**, with unchanged **179,440 B/op and 174 allocations/op**. The unchanged
19-case driver measures a **3.77× gap versus single-threaded ORT**. Numerical
outputs, APIs and supported operators remain unchanged.

## Dependency identity and scope

* Published tag: `v0.10.2`.
* Commit: `531e7f835c9e4bd2ac3122439231b6d0bace72b8`.
* Module sum: `h1:FYqGuz6rbQOkhlklqNVUOnEtOj3trsgx16Zxu//LyBY=`.
* Module-file sum: `h1:4ha7jggW7cxuYy1EIaGUvn6aLe8NBt8BLOYm4to6y+Y=`.
* No module replacements, runtime adapter changes or other dependency upgrades.
* The included MIT license remains byte-identical to the published module.

The release adds `conv2DResidualChannels → conv2DResidualSums` for eligible
ARM64 convolutions. It clips rows/columns and proves spans before assembly,
retains scalar fused reductions and skips padded products, removing repeated
inner address/bounds work. It also preserves bias operand priority. This is
separate from the existing NEON four-column interior tile. `purego` and non-ARM64
builds retain the existing fallback. Empty/single-product cases keep their
cheaper scalar path.

The [original investigation](../README.md) and
[historical backend handoff](../neuralnetwork-prompt.md) describe the v0.10.1
bottleneck; they are historical evidence rather than a new ranking of v0.10.2.

## Measurements

Apple M3, macOS arm64, Go 1.27.2, `GOARM64=v8.0`, `CGO_ENABLED=0`,
`GOMAXPROCS=1`, `GOWORK=off`. Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2,
ORT 1.19.2 and protobuf 4.25.3. Baseline and upgraded binaries were built before
measurement. Seven serial 700-ms benchmark pairs alternate execution order;
no builds, tests or profiles run concurrently with benchmark measurements.

| Original `BenchmarkMNIST` | v0.10.1 | v0.10.2 |
| --- | ---: | ---: |
| Median | 200.790 µs/op | 166.291 µs/op |
| Range across seven samples | 199.140–202.997 µs | 162.629–174.519 µs |
| Bytes/op | 179,440 | 179,440 |
| Allocations/op | 174 | 174 |

The optional prepared-node benchmark, five alternating 300-ms pairs, isolates
the relevant kernels with real intermediate tensors:

| Prepared call | v0.10.1 | v0.10.2 | Lower latency |
| --- | ---: | ---: | ---: |
| First convolution | 18.322 µs | 16.166 µs | 11.8% |
| Second convolution | 100.540 µs | 67.408 µs | 33.0% |

The combined 35.3-µs kernel reduction closely matches the 34.5-µs end-to-end
reduction, supporting attribution to the dependency's convolution change.
Before/after CPU profiles of the second convolution confirm the new residual
assembly is reached; the old `conv2DFourChannels` hotspot disappears from this
call's profile. Raw profile percentages are conditional and subject to the
profiling limitations recorded in the original investigation.

Five alternating 300-ms samples for each of the six small MLP/CNN/shape warm
fixtures show median decreases of 0.4–2.4%, with unchanged allocations. Small
variations on these fixtures do not establish a universal improvement. See
[all benchmark summaries](data/benchmark-summary.json) and raw samples in `data`.

## Unchanged ORT comparison and correctness

The pinned IR8/opset13 MNIST model, input hashes, all 19 cases, seed, tolerances
and counts remain unchanged. ORT uses the CPU provider, sequential execution,
one intra-op/inter-op thread and **disabled graph optimization**. Each case has
10 warmups and 100 measured calls. Timing includes inference and owned outputs;
loading, tensor preparation, JSON and process startup are excluded.

| Driver median / p95 | v0.10.1 comparison | v0.10.2 comparison |
| --- | ---: | ---: |
| Go | 199.10 / 283.84 µs | **161.81 / 244.64 µs** |
| ORT | 42.17 / 99.88 µs | **42.96 / 48.79 µs** |
| Median slowdown | 4.72× | **3.77×** |

Both comparisons pass all numerical, dtype, shape, name, top-1,
original/adapted-model and upstream expected-output checks. Maximum absolute
Go/ORT error remains **1.9073486328125e-5**. Every Go output bit matches the fresh
v0.10.1 baseline. These single driver runs are separate from the repeated
paired Go benchmark evidence; ORT's p95 variation is not a backend improvement.

An additional **131 archived probes** produce bit-identical output tensors
through v0.10.1, v0.10.2 and v0.10.2 `purego`. Their input request and output
reports are retained as gzip JSON under `data`. This is synthetic numerical
coverage rather than a labeled MNIST accuracy evaluation. The archived input
SHA-256 is
`7801b001c362560478a24c275e1134e79794905fcfe5619ec9e31e214b29afe8`.

## Validation and artifacts

The standard CI checks pass locally: formatting, tidy diff, generated support
table, module verification, build, vet, default tests, cgo-disabled tests,
`purego` tests, race tests and all four bounded fuzz targets. A Linux/amd64
cgo-disabled cross-build and the three Python comparison-policy tests pass.
The published backend's focused residual bit/operand-priority and payload-bound
tests also pass on this ARM64 host. Existing tests, fixtures, numerical
expectations and comparison tolerances were not weakened or changed.

[checks.json](data/checks.json) records exact commands, environment overrides and
exit statuses. `data` contains benchmark samples, before/after convolution
profiles, validation summaries, module identity, compressed full comparison
reports and archived-probe input/output reports. The parent
[integrity manifest](../sha256.json) covers the evidence. Full local binaries and
logs are at `/private/tmp/onnx-v0.10.2-integration/`; the new comparison is also
cached at `.cache/compare-runtime/v0.10.2-report.json`.

To reproduce the upgraded measurements, prepare the optional model using the
parent comparison tool, build with the flags above and run:

```sh
go test -c -o /tmp/onnx-v0.10.2.test .
GOMAXPROCS=1 /tmp/onnx-v0.10.2.test -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=700ms -test.count=7 -test.benchmem
```

Use the parent investigation commit for v0.10.1, build a separate binary and
alternate before/after order when reproducing the paired comparison. Leave the
comparison driver at its default case counts and ORT settings. Published
dependency selection changes no ONNX public interfaces, ownership, limits,
cancellation policy or operator coverage; performance gains are architecture
and workload dependent.
