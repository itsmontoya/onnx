# Pass 20: metadata proof and repeated validation

[PERFORMANCE_PASS20.md](../../../PERFORMANCE_PASS20.md) is the 26-part report.
[DECISION.md](DECISION.md) records the fourteen-question gate before production
edits. One local actual dtype extraction in evaluation.checkValue replaces
repeated extraction from the same unchanged tensor across declaration checks.
No backend production/API/dependency or Session metadata change occurs.

Artifacts:

- `data/summary.json`: fresh provenance, model/driver/runner fingerprints,
  frequencies before/after, all observations, medians/ranges/CVs, reference
  movement, ordinary counters, components and arithmetic milestones.
- `data/validation.json`: recorded commands, exits, environment overrides and
  durations, including the first rejected module-cache overlay diagnostic build.
- `data/evidence.json.gz`: lossless comparison reports/raw latency arrays, logs,
  overlays, source snapshots and hashes. Diagnostic mistakes and outliers remain.
  Hardware serial/UUID output is excluded.
- `data/*.pprof`: original CPU/allocation and final allocation profiles.
- `*.go.txt`: test-only fixtures, excluded from normal builds. Pass 19 fixture,
  benchmark and context sources are reused through overlays. Backend geometry
  tests reuse Pass 16/17/19 fixtures on a byte-identical v0.10.10 copy.

Counter instrumentation changes only diagnostic build overlays on a verified
copy; original module-cache and sibling backend files remain unchanged. The
temporary replacement modfile applies only to counter binaries. No instrumented
latency or modified backend enters a production result. Production control
overlays restore merged `06ae89f:evaluation.go`; diagnostic controls do so too.

All timings are sequential. Fresh baseline, components, complete-Run diagnostics,
final matched fixed-policy pairs and ordinary ramp are separate scopes. No valid
observation is filtered. The 53.348/52.224-us prepared-proof observations and both
negative final Go pair differences remain. Invalid direct-retention and shell
variants are labeled contract violations and never production candidates.

The original Compare probe included rank mismatches; CompareExact separately
measures nine equal distinct shapes. The initial module-cache overlay build and
an initial Python module-name conflict failed before timing. Both are documented
in archived diagnostic notes. A one-off Python escape warning affected no Go
command; codegen.py uses a raw regex string.

Replay into a **new** output directory; preserve committed observations. Prepare
the pinned MNIST fixture with `tools/compare_runtime.py` first. Published modules
and the reference Python dependencies must be available. Helper defaults use this
host's paths; override `ONNX_REFERENCE_PYTHON`, `ONNX_REFERENCE_PYTHONPATH` and
`ONNX_BENCHMARK_GOCACHE` on another host.

```sh
export ONNX_PASS20_OUT=/private/tmp/onnx-pass20-replay
python3 tools/compare_runtime/pass20/measure.py prepare
python3 tools/compare_runtime/pass20/measure.py baseline
python3 tools/compare_runtime/pass20/measure.py profiles
python3 tools/compare_runtime/pass20/diagnostics.py counters
python3 tools/compare_runtime/pass20/diagnostics.py components
python3 tools/compare_runtime/pass20/diagnostics.py ablations
python3 tools/compare_runtime/pass20/backend.py
python3 tools/compare_runtime/pass20/measure.py production
python3 tools/compare_runtime/pass20/codegen.py
python3 tools/compare_runtime/pass20/checks.py suites
python3 tools/compare_runtime/pass20/checks.py metal
# Run with native GPU access and inspect skips explicitly:
python3 tools/compare_runtime/pass20/checks.py metal_host
python3 tools/compare_runtime/pass20/final_checks.py
# Packaging writes under this checkout's data/: preserve committed originals.
python3 tools/compare_runtime/pass20/summarize.py
```

prepare may build an incidental current-tree binary first, but rebuilds the
measured baseline using the merged evaluation overlay. Candidate counters use the
current production checkValue. Original counters and every ablation use merged
evaluation. All other production Go files are unchanged. Evidence packaging
verifies measured candidate hashes, original backend Go hashes and exact-copy
bytes again.

Native Metal suites and isolated stress repetitions pass with zero skips;
sandbox backend has 46 device skips. The historical Pass 16 unrestricted stress
failure remains unexplained. No latency claim depends on Metal results.
