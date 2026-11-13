# Pass 19: post-retention critical-path investigation

[PERFORMANCE_PASS19.md](../../../PERFORMANCE_PASS19.md) contains the 26-part report.
[DECISION.md](DECISION.md) records the production selection gate before source
changes. [ALLOCATION_LEDGER.md](ALLOCATION_LEDGER.md) describes every current warm
allocation and its ownership/semantic obligations.

Only `evaluation.go` changes production behavior internally: a cleared inline
three-pointer argument array per evaluation, with fresh allocation for larger
arities. No backend production source, API or dependency change occurs.

Artifacts:

- `data/summary.json`: environment, module provenance, model and driver/runner
  fingerprints, all medians/ranges/CVs and estimator inputs, diagnostic counters,
  ORT signals, reference movement and arithmetic milestone status.
- `data/validation.json`: recorded commands, configuration overrides, exits and
  durations, including the initial failed diagnostic build.
- `data/evidence.json.gz`: lossless reports, raw latency arrays, all benchmark/
  profile/check logs, diagnostic sources/overlays, source snapshots, hashes and
  three repetitions of ORT node samples. Hardware UUID/serial output is excluded.
- `data/*.pprof`: original allocation, candidate allocation and CPU profiles.
- `*.go.txt`: test-only fixtures, loaded through overlays. No benchmark diagnostic
  type or test is added to normal builds. Backend fixtures overlay **tests only**
  onto a verified byte-identical copy of published v0.10.10.

All timing runs are sequential. Fixed-policy initial baseline, focused argument
experiment, final production comparison, ordinary ramp, prepared traversal,
shell and component scopes stay separate. The driver, inputs, conversions,
tolerances, upstream/top-1 checks, engine order and primary estimator do not change.
All valid observations and the initial attribution mistakes remain archived.

Replay from this final source revision, with a **new** output directory. Do not
replace the committed original observations with replay numbers. The pinned MNIST
model/archive must be prepared with `tools/compare_runtime.py`; Python dependencies
and published Go modules must be available. The existing helper defaults are this
host's paths; override `ONNX_REFERENCE_PYTHON`, `ONNX_REFERENCE_PYTHONPATH` and
`ONNX_BENCHMARK_GOCACHE` when replaying elsewhere.

```sh
export ONNX_PASS19_OUT=/private/tmp/onnx-pass19-replay
python3 tools/compare_runtime/pass19/measure.py prepare
python3 tools/compare_runtime/pass19/measure.py baseline
python3 tools/compare_runtime/pass19/diagnostics.py components
python3 tools/compare_runtime/pass19/measure.py profiles
python3 tools/compare_runtime/pass19/diagnostics.py ablations
python3 tools/compare_runtime/pass19/backend.py
python3 tools/compare_runtime/pass19/measure.py argument_comparisons
python3 tools/compare_runtime/pass19/measure.py production
python3 tools/compare_runtime/pass19/diagnostics.py candidate_components
python3 tools/compare_runtime/pass19/checks.py suites
python3 tools/compare_runtime/pass19/checks.py metal
# Run with actual native GPU access, inspect skips explicitly:
python3 tools/compare_runtime/pass19/checks.py metal_host
python3 tools/compare_runtime/pass19/final_checks.py
# Packaging writes under this checkout's data/; preserve committed originals first.
python3 tools/compare_runtime/pass19/summarize.py
```

The initial baseline and control overlays restore merged `2445b79:evaluation.go`;
all other production Go source is unchanged. Thus replay from the candidate does
not accidentally use candidate argument storage as the baseline. The inherited
preparation helper builds an incidental binary first; the separately rebuilt
merged `baseline.test` is the measured one. `production` independently builds
merged-control and actual candidate binaries and runs seven alternating pairs.

The original investigation also recorded compiler escape listings, ARM64 objdumps,
allocation profiles and the ORT probe separately. Reproduce those without changing
benchmark scopes:

```sh
# Set the same environment as measure.py (including GOMAXPROCS=1/CGO_ENABLED=0).
go test -c -gcflags=github.com/itsmontoya/onnx=-m=2 -o /private/tmp/pass19-escape.test .
go tool objdump -s 'onnx.\(\*evaluation\).(execute|normalize|nodeArguments)$' /private/tmp/pass19-escape.test
# Use the configured reference interpreter and PYTHONPATH:
"$ONNX_REFERENCE_PYTHON" tools/compare_runtime/pass19/ort_profile.py
```

Initial retention lookup and Conv shape diagnostics included accessor/fixture
construction work; corrected probes separate that work. Both results remain.
The first no-context transformer did not compile and was corrected before timing.
Invalid-contract overlays are diagnostic ceilings, never production candidates.
No combination of their deltas establishes recoverable work or a 1.10× claim.
Native Metal passes with zero skips; sandbox backend has 46 device skips. The
historical Pass 16 unrestricted stress failure remains unexplained.
