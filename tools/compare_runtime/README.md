# Compare with ONNX Runtime

Run the same model and named tensors through this repository's Go runtime and
the official Python `onnxruntime` CPU package. The default is the small pretrained
[MNIST CNN from the ONNX Model Zoo](https://github.com/onnx/models/tree/4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177/validated/vision/classification/mnist).
This is an optional development tool; it adds no Go runtime dependency.

## Run

From the repository root, with Go from `go.mod` and Python **3.9–3.12**:

```sh
python3.11 -m venv .cache/compare-runtime/venv
.cache/compare-runtime/venv/bin/python -m pip install -r tools/compare_runtime/requirements.txt
.cache/compare-runtime/venv/bin/python tools/compare_runtime.py
```

The Python pins match the repository's existing reference tools. Initial setup
needs network for packages, Go modules and the 26 kB model archive. The model
download uses an immutable upstream revision and a checked SHA-256. Later runs
reuse the archive in `.cache/compare-runtime/`. You can supply an already
downloaded archive without network:

```sh
.cache/compare-runtime/venv/bin/python tools/compare_runtime.py \
  --archive /path/to/mnist-12.tar.gz --report /tmp/mnist-comparison.json
```

The command prints agreement and latency summaries and writes a JSON report to
`.cache/compare-runtime/report.json` by default. A numerical, dtype, shape, name,
conversion or top-1 mismatch exits nonzero. Unsupported models and operational
errors also fail. `--help` lists tolerances, run counts, seed and cache options.

## Model preparation and checks

The original pretrained model has **IR 7/opset 12**, outside the Go runtime's
documented profiles. The tool keeps its bytes as `mnist-original.onnx`, uses
ONNX's version converter to reach opset 13, and raises its IR header to 8. It
checks the original and converted models with the full ONNX checker and writes
the adapted model as `mnist-ir8-opset13.onnx`. This conversion is specific to
the pinned MNIST model; custom models are never automatically converted.

Every case checks both Go versus converted-model ORT and converted-model ORT
versus original-model ORT. The upstream case also checks ORT against the Model
Zoo's saved output. The source, archive, model, input and tool hashes, versions,
profiles and conversion are recorded. The original bytes remain outside Go's
supported profiles. The [model README declares MIT licensing](https://github.com/onnx/models/blob/4c46cd00fbdb7cd30b6c1c17ab54f2e1f4f7b177/validated/vision/classification/mnist/README.md#license);
model artifacts are downloaded into the ignored cache and are not vendored.

The default **19 cases** are the bundled upstream test vector, zeros, ones and
16 seeded uniform `[0,1]` tensors, all shaped `[1,1,28,28]`. The upstream vector
has values outside `[0,1]`; these are synthetic numerical probes, not a labeled
MNIST accuracy evaluation. Top-1 agreement compares the argmax of the ten
output logits. It does not establish classification accuracy.

Float32 comparisons use `abs(actual - expected) <= atol + rtol * abs(expected)`,
with defaults `atol=1e-4`, `rtol=1e-5`. Matching NaNs and matching infinity signs
are accepted; signed zero is ignored. Int64 values compare exactly. Dtypes,
shapes and output names must match. Tensor transport preserves float32 bits and
full int64 precision, including empty tensors and scalars.

## Other compatible models

Provide a local model and an NPZ file whose keys are the model's input names.
Only dense float32/int64 tensors and models supported by the Go runtime can run.
The tool does not configure an external-weight resolver.

```sh
.cache/compare-runtime/venv/bin/python - <<'PY'
import numpy as np
np.savez('/tmp/mlp-inputs.npz', features=np.array([[1, 2]], dtype=np.float32))
PY
.cache/compare-runtime/venv/bin/python tools/compare_runtime.py \
  --model testdata/inference/models/mlp.onnx --inputs /tmp/mlp-inputs.npz \
  --report /tmp/mlp-comparison.json
```

Repeat `--inputs` for more cases, including multiple named inputs or different
supported batch sizes. Custom mode compares the supplied model directly without
model conversion or classification assumptions.

## Timing and observed result

Both runtimes use one session, with `GOMAXPROCS=1` and `CGO_ENABLED=0` for Go,
and sequential CPU execution with one intra/inter-op thread for ORT.
ORT graph optimization is disabled by default; `--ort-optimizations all` enables
its optimized deployment path and records the setting. Each case has 10 warmup
runs and 100 measured runs by default. Timing covers the session inference call
and owned outputs, excluding process startup, Go compilation, model loading,
input preparation and JSON transport. Load/session creation are recorded
separately, and raw timing samples remain in the report.

On 2026-11-09, macOS arm64, Go 1.27.2, Python 3.9.6 and the pinned Python
packages, using runtime revision `985f3b311815d170308d62f48aeffc2f6bf4483c`
with these tool additions:

| Check | Observed result |
| --- | --- |
| Go versus ORT | 19/19 cases pass; max absolute error `1.9073486328125e-5` |
| Top-1 agreement | 19/19; upstream vector predicts class 3 in both |
| Original versus adapted model in ORT | Exact agreement across all 19 cases |
| ORT versus bundled expected output | Pass; max absolute error `3.814697265625e-5` |
| Go median / p95 | 1,187.81 / 1,294.62 µs |
| ORT median / p95 | 41.79 / 45.17 µs |

This single local run measured 1,900 calls per runtime, with ORT approximately
28 times faster. Runtimes run sequentially, through different language APIs;
thermal state, order and API overhead can affect results. This is a repeatable
smoke comparison on one small network, not a general performance claim.

After upgrading to published **neuralnetwork v0.8.0** and avoiding unnecessary
ONNX ReLU output snapshots, the same settings on 2026-11-09 measured:

| Runtime | Median / p95 |
| --- | ---: |
| Go | 689.58 / 772.90 µs |
| ORT | 41.54 / 44.21 µs |

All 19 cases and top-1 checks pass with the same maximum absolute error. The
median gap is **16.60×**. The [profiling report](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/performance/mnist/README.md)
records allocation benchmarks, module identity, checks and the earlier baseline.
The full new comparison is `.cache/compare-runtime/v0.8.0-report.json`.

With published **neuralnetwork v0.9.0**, the unchanged driver measured Go at
452.90 / 548.02 µs median / p95 and ORT at 44.62 / 46.25 µs on 2026-11-09,
a **10.15×** median gap. All 19 cases and top-1 checks pass, with unchanged
maximum absolute error and Go outputs bit-identical to v0.8.0. Five alternating
Go benchmark repetitions measured medians of 734.70 µs for v0.8.0 and 451.01 µs
for v0.9.0, about 39% lower latency. The
[second profiling report](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/performance/mnist-round2/README.md) records
the published module identity, raw benchmark samples and validation.
The full comparison is `.cache/compare-runtime/v0.9.0-report.json`.

With published **neuralnetwork v0.10.0**, the unchanged driver measured Go at
324.21 / 374.30 µs median / p95 and ORT at 44.21 / 46.50 µs on 2026-11-09,
a **7.33×** median gap. All 19 numerical and top-1 checks pass, with unchanged
maximum absolute error and Go outputs bit-identical to v0.9.0. Seven alternating
Go benchmark pairs measured medians of 433.70 µs for v0.9.0 and 310.14 µs for
v0.10.0, about 28.5% lower latency with unchanged allocations. The
[third profiling report](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/performance/mnist-round3/README.md) records
the published module identity, raw measurements and validation. The full
comparison is `.cache/compare-runtime/v0.10.0-report.json`.

On 2026-11-10, upgrading to published **neuralnetwork v0.10.1**, commit
`ea17b585c42264ca85a637705608c284b80464bc`, measures Go at **194.50 / 277.72 µs**
median / p95 and ORT at **41.96 / 47.50 µs**, a **4.64×** median gap. All 19
numerical and top-1 cases pass, including the original/adapted-model and upstream
expected-output checks. Maximum absolute error remains `1.9073486328125e-5`;
all Go output bits and model/input hashes match the fresh v0.10.0 baseline.

Seven alternating precompiled benchmark pairs, reversing order each repetition,
measure `BenchmarkMNIST` at **311.059 µs/op** for v0.10.0 and **200.054 µs/op**
for v0.10.1: **35.7% lower latency / 1.55× faster**, with unchanged
**179440 B/op / 174 allocations**. Baseline samples span 305.389–326.451 µs;
new samples span 196.647–202.179 µs. Each sample runs for 700 ms, using
`GOWORK=off`, `CGO_ENABLED=0`, `GOARM64=v8.0`, and `GOMAXPROCS=1`, on Apple M3
with Go 1.27.2 and the same pinned Python packages. No tests, builds or profiles
run concurrently with those measurements. The backend's new interior
convolution tile is specific to ARM64 builds without `purego`; portable builds
retain their prior path.

The six small warm fixtures show median increases of 1.1–3.1% in five alternating
200 ms samples, with unchanged allocations. Longer seven-pair 700 ms checks of
the batch-three MLP and CNN show increases of 2.48% and 2.47%, respectively
(about 0.09 µs and 0.18 µs). The cause of these small regressions was not isolated;
the large MNIST gain does not establish a performance improvement for every graph.

The additional 131 archived numerical probes produce bit-identical outputs
through v0.10.0, v0.10.1 and v0.10.1 `purego`. Default, cgo-disabled, `purego`
and race suites, bounded fuzzing and comparison-policy tests pass. The full
comparison is `.cache/compare-runtime/v0.10.1-report.json`; local raw validation
and benchmark logs are in `/private/tmp/onnx-v0.10.1-integration/`.

The optional `BenchmarkMNISTNodes` isolates prepared operators and ReLU
correction costs using real intermediate tensors. Its `04_ConvRotatingWorkingSet`
subbenchmarks rotate independent tensor sets to diagnose cache sensitivity;
they are separate experiments from the unchanged end-to-end `BenchmarkMNIST`.

The comparison-policy tests run with the same optional Python environment:

```sh
.cache/compare-runtime/venv/bin/python -m unittest discover -s tools -p 'test_compare_runtime.py'
```
