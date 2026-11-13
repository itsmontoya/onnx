# Pass 16 evidence and NeuralNetwork handoff

ONNX now uses published **NeuralNetwork v0.10.10**. Fresh integration evidence is
preserved separately in [release/](release/README.md). The prompt and measurements
below are the completed historical handoff; it does not need to be run again.

[PERFORMANCE_PASS16.md](../../../PERFORMANCE_PASS16.md) records the repository
boundary decision. [NEURALNETWORK_PROMPT.md](NEURALNETWORK_PROMPT.md) is a
self-contained implementation prompt for a fresh agent working only in the
NeuralNetwork repository. Run that prompt separately; this pass did not launch
another agent or implement a backend fix.

- `diagnostic_test.go.txt`: test-only backend benchmarks and finite-fixture
  bitwise/input-preservation checks. It is not included in normal builds.
- `data/summary.json`: environment, source fingerprints, all 147 benchmark
  samples, medians, ranges, CVs, B/op and allocs/op for 21 scopes.
- `data/validation.json`: invoked commands, environments, exit codes and durations.
- `data/evidence.json.gz`: lossless JSON including the above plus raw benchmark,
  filtered allocation-profile/site, escape-analysis, disassembly and check logs.
  Read using Python `gzip.open` and `json.load`. Full binaries/profiles remain in
  ignored `.cache/compare-runtime/pass16/`.

Latency and profiling are separate. The seven-repetition baseline has no new
production candidate or paired Go/ORT transfer. ONNX numerical adapters use real
MNIST ramp intermediates. Backend diagnostics use the finite deterministic
formula in the prompt. Copy/arithmetic ceilings omit parts of the public
contract and are not additive production savings.

Go prohibits adding test overlays beneath GOMODCACHE. After that initial rejected
build, replay copies **unmodified** released v0.10.9 into the ignored ONNX
evidence directory, verifies every original file byte-for-byte, and adds only a
test overlay there. No actual NeuralNetwork checkout or module cache is edited;
neither go.mod has a replace directive or production patch. The initial failed
build remains in the command ledger; later corrected builds/checks pass.
The first preparation also caught diagnostic formatting before compilation;
the preserved diagnostic source is gofmt-formatted.

Replay from the ONNX root with Go 1.27.2 and released v0.10.9 already available.
Use the investigation commit `da7ac98` for historical replay; it contains these
artifacts with the original v0.10.9 module files. Current go.mod uses v0.10.10.
Use a writable Go cache (the measured host defaults to the existing
`/private/tmp/onnx-pass6/go-cache`). Prepare the pinned model using the existing
`tools/compare_runtime.py` workflow if
`.cache/compare-runtime/mnist-ir8-opset13.onnx` is absent. Its SHA-256 must be
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
The preparation script records macOS hardware using `system_profiler`/`sw_vers`.

```sh
export ONNX_BENCHMARK_GOCACHE=/path/to/writable/go-cache
python3 tools/compare_runtime/pass16/measure.py prepare
python3 tools/compare_runtime/pass16/measure.py measure
python3 tools/compare_runtime/pass16/measure.py checks
python3 tools/compare_runtime/pass16/summarize.py
```

Run phases sequentially without competing benchmarks. Preserve the original
working results before replay; new measurements are new observations and must
not be described as the original samples. `summarize.py` verifies sample counts,
unchanged measured production/model/diagnostic source, and successful final
validation before rebuilding the bundle. It omits hardware serial/UUID data.

The later backend agent must independently validate the diagnosis on its current
base, implement and review only backend changes, and return the API/performance/
correctness/integration information specified in the prompt. ONNX integration
waits for an actual published backend version.
