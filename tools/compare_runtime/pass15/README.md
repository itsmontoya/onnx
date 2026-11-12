# Pass 15 evidence and replay

Read [PERFORMANCE_PASS15.md](../../../PERFORMANCE_PASS15.md) for the full report,
including the allocation ledger, alternative descriptor and latency limitations.

The selected production change is neuralnetwork commit
`647e0c8ef3c93d90707df5077046346cd658f68e`. The original investigation kept
ONNX's go.mod on v0.10.8 and selected the candidate with temporary module files.
ONNX now uses released v0.10.9; [release validation](release/README.md) preserves
the fresh published-module comparison separately.
The standalone [backend.patch.gz](backend.patch.gz) applies to `be95e90` and contains
only the selected implementation, tests and benchmark extensions.

- [DECISION.md](DECISION.md) records the evidence summary before production
  selection. Initial diagnostics and final paired results have distinct scopes.
- [data/summary.json](data/summary.json) contains estimators, all component
  ranges/CVs, dynamic benchmarks, environment and source/input fingerprints.
- [data/evidence.json.gz](data/evidence.json.gz) is lossless JSON: all 56
  paired comparison reports plus the separate preflight, individual inference timings, output bits, original logs,
  compiler/disassembly/profile summaries, census and three ORT node-sample sets.
  Decode with Python `gzip.open` and `json.load`.
- [data/validation.json](data/validation.json) records commands, environments,
  exit codes and durations. The initial control-overlay build failure is kept;
  it attempted to compile new private-helper tests with old production source.
  The corrected overlay excludes those new tests and successfully builds the
  original backend. Last-result checks require success without deleting failures.
- The `*.go.txt` files preserve diagnostic source. They never enter normal
  production builds. The static prototype extends existing execution-plan
  ownership through Go overlays; it is not the selected production API.

Large test binaries, full profiles and working files remain in ignored
`.cache/compare-runtime/pass15/` and the isolated backend's
`.benchmarks/add-pass15/run/`. The initial prototype's seven pairs, final source's
fourteen pairs, and descriptor experiment's seven pairs are all retained. Initial
and final source are not silently pooled. The final estimator uses all fourteen
final-source pairs, including the predefined confirmation after excess variance.

The early `ownership.txt` experiment accidentally cloned shape scratch on
success; `ownership2.txt` reused that same binary after a failed rebuild. Neither
is the selected candidate. `ownership3.txt` measures the corrected prototype.
They remain diagnostic logs, not final-source performance samples. The final
source additionally preserves public non-nil empty scalar slices and names the
stack capacity. These details explain why source stages remain separate.

## Reproduction

Use the report's Go/Python/ORT versions and the existing pinned model archive.
Set these paths for your environment:

```sh
export ONNX_REFERENCE_PYTHON=/path/to/reference/python
export ONNX_REFERENCE_PYTHONPATH=/path/to/reference/python/site-packages
export ONNX_BENCHMARK_GOCACHE=/path/to/writable/go-cache
export ONNX_PASS15_BACKEND=/path/to/neuralnetwork
```

The backend must contain the selected commit (or decompress/apply backend.patch.gz to the
recorded baseline). Keep benchmark processes sequential and use a new working
artifact directory or preserve existing results before replay. The scripts target
the measured macOS host; `system_profiler`/`sw_vers` record fresh host information.

```sh
"$ONNX_REFERENCE_PYTHON" tools/compare_runtime.py --report .cache/compare-runtime/prepare-pass15.json
python3 tools/compare_runtime/pass15/measure.py prepare
python3 tools/compare_runtime/pass15/measure.py comparisons
ONNX_PASS15_START=7 python3 tools/compare_runtime/pass15/measure.py comparisons
python3 tools/compare_runtime/pass15/measure.py components
python3 tools/compare_runtime/pass15/extras.py
python3 tools/compare_runtime/pass15/prototype.py
python3 tools/compare_runtime/pass15/measure.py checks
```

`extras.py` constructs its baseline overlay from the released module cache, so
later sibling-checkout changes do not contaminate control builds. It measures
changing runtime shapes, higher-rank fallback, binding and generated code.
`prototype.py` reconstructs the immutable descriptor overlays from the preserved
source and runs its separate seven-pair experiment. The original source-stage
archive and initial controlled backend diagnostics are already in the evidence;
they are historical measurements, not prerequisites for replaying final source.

To rerun the initial controlled decomposition independently, copy the released
v0.10.8 module to a writable checkout, add `diagnostic_test.go.txt` as a tensor
test file, and run `BenchmarkPass15` with 200 ms / five repetitions / one Go
thread. The normal complete path is compared with separately resolved arithmetic,
shape, input/destination validation and synthetic static diagnostics. The
diagnostic Add-3 arithmetic path bypasses matrix adapters; the full plan-owned
prototype intentionally preserves them.

`summarize.py` rebuilds the committed bundle from the complete original working
directory, including its `initial/` archive. It verifies nineteen cases, 1,900
samples per engine/report, exact policy/tolerances, all output bits, identical
driver/runner/model/input fingerprints and successful corrected checks. Replay
measurements are new samples and should not overwrite the committed original
data or be described as the original observations.
