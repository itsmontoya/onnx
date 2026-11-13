# Pass 17: remaining runtime costs

[PERFORMANCE_PASS17.md](../../../PERFORMANCE_PASS17.md) is the complete report.
[DECISION.md](DECISION.md) records the production gate. This pass makes **no
production change** and creates no backend implementation branch or handoff.
Pass 18 should investigate bounded ONNX destination reuse with explicit accounting.

- [ALLOCATION_LEDGER.md](ALLOCATION_LEDGER.md): semantic ownership and allocation
  accounting for the ordinary **92,826 B / 72 objects**, including qualifications.
- `data/summary.json`: environment, module provenance, fingerprints, every timing
  sample, medians/ranges/CVs, bytes, objects, diagnostic differences and roadmaps.
- `data/validation.json`: all recorded commands, overrides, exits and durations.
- `data/evidence.json.gz`: lossless JSON containing all seven fixed-policy
  reports, the separate purego report, ORT node samples, benchmark/profile/check
  logs and the summary. Hardware serial/UUID output is excluded.
- `fixture_test.go.txt`: serial ONNX diagnostics, installed only through test
  overlays; never included in normal production or test builds.
- `backend_test.go.txt`: Pool/Tensor diagnostics. Backend MatMul diagnostics reuse
  the explicit finite fixture in Pass 16, against **current released v0.10.10**.

All benchmarks run sequentially, with compilation outside timing. Latency tests
and allocation profiles are separate. Each fixed-policy report contains 1,900
calls per engine, ten warmups per each of nineteen sorted cases, owned outputs,
Go then ORT, unchanged conversion/upstream/top-1 checks and atol=1e-4/rtol=1e-5.
The primary estimator remains the difference of engine/version report medians;
paired differences are secondary. There is no production candidate in this pass.
Diagnostic Go-only comparisons have no ORT movement or valid Go/ORT ratio.

Replay **sequentially** from the ONNX root, with Go 1.27.2, published NeuralNetwork
v0.10.10, the pinned MNIST archive/model, and the reference Python dependencies in
`data/summary.json`. The baseline environment uses `GOWORK=off`, `GOARM64=v8.0`,
`GOMAXPROCS=1`, `CGO_ENABLED=0`, and no GOFLAGS. Override local paths with
`ONNX_BENCHMARK_GOCACHE`, `ONNX_REFERENCE_PYTHON`, and
`ONNX_REFERENCE_PYTHONPATH` as described by the Pass 15 measurement helper.

```sh
python3 tools/compare_runtime/pass17/measure.py prepare
python3 tools/compare_runtime/pass17/measure.py baseline
python3 tools/compare_runtime/pass17/measure.py profiles
python3 tools/compare_runtime/pass17/diagnostics.py
python3 tools/compare_runtime/pass17/backend.py
python3 tools/compare_runtime/pass17/details.py
python3 tools/compare_runtime/pass17/checks.py
python3 tools/compare_runtime/pass17/machinery.py
# Run with actual native GPU access:
python3 tools/compare_runtime/pass17/metal_host.py
python3 tools/compare_runtime/pass17/final_checks.py
python3 tools/compare_runtime/pass17/summarize.py
```

Keep original `.cache/compare-runtime/pass17/` results before replay. New
observations must not overwrite historical evidence or be described as the
original observations. All valid measured repetitions, including outliers, are
retained. An initial per-scope MemStats census was repeated with logging removed
from between scopes and 1,000 rather than 100 runs; **both logs remain**. Scope
barriers themselves perturb tiny-allocation packing; the ledger uses ordinary
benchmark counters, source sizes and allocation-removal controls instead.
Additional prepared/argument and disabled-operator probes have their own controls;
their controls are not pooled or substituted for one another.

The backend experiment copies every published file byte-for-byte into the
ignored evidence directory and overlays **test files only**. Packaging verifies
that copy again. It does not edit the sibling checkout, module cache, module
files, or backend production code, and does not use a replace directive.
The ONNX ablations overlay copies of ONNX source only in diagnostic binaries.
Some intentionally violate ownership, quota, isolation or numerical contracts;
they are measurements, never production candidates. Their limitations are in
the report. The original and corrected-unit allocation-profile top listings
are both retained; sampled profile counts are not ordinary allocation counts.

The sandbox Metal run skipped device-dependent tests. The native host rerun
executes complete serialized ONNX/backend Metal suites with **zero skips**, plus
five isolated repetitions of the historical stress test. These pass. The
unexplained Pass 16 failure remains unexplained; these results do not resolve it.
No benchmark claim uses the Metal-tagged execution path.
