# Pass 16 published v0.10.10 integration

[PERFORMANCE_PASS16_RELEASE.md](../../../../PERFORMANCE_PASS16_RELEASE.md) records
the published-module integration, separately from the earlier
[investigation and backend handoff](../README.md).

Control: released NeuralNetwork v0.10.9. Candidate: released v0.10.10, tag commit
`2ffa5bd9fd2034382c49d14efc732ea4d026aa09`, module sum
`h1:CcoQQN8w9aGodz61FQGKSWtbnKzAHf/gtXbukwVUkj0=`. There is no local
replacement, backend patch, new ONNX arithmetic implementation or workspace
charge change. The production integration is limited to go.mod/go.sum.

- `data/summary.json`: environment/provenance, every derived timing statistic,
  raw benchmark samples, component allocations, loading costs and fingerprints.
- `data/evidence.json.gz`: lossless JSON containing all fourteen paired default
  comparison reports, a separate purego correctness report, individual inference
  latencies/output bits, benchmark/profile summaries and validation logs.
- `data/validation.json`: commands, environment overrides, exit codes and durations.

The fixed policy remains nineteen sorted upstream/zeros/ones/seed-0 random cases,
ten warmups and 100 timed calls per case, owned outputs, Go then ORT,
atol=1e-4 / rtol=1e-5. Seven alternating version pairs retain every valid sample.
Component measurements use real ramp intermediates and existing disjoint
destinations. Separate ordinary-ramp timing and profiled allocations are not
divided by fixed-policy ORT timing or pooled with historical measurements.

Replay sequentially from the ONNX root with the report's Go/Python/ORT versions
and the unchanged pinned model archive. Configure `ONNX_REFERENCE_PYTHON`,
`ONNX_REFERENCE_PYTHONPATH` and `ONNX_BENCHMARK_GOCACHE` as described in the
[Pass 15 replay guidance](../../pass15/README.md). The measurement script reuses
the unchanged Pass 15 command/check helpers and supplies this release's module
snapshots, paths and comparison phases. Preparation records fresh macOS host
information using `system_profiler`/`sw_vers`.

```sh
python3 tools/compare_runtime/pass16/release/measure.py prepare
python3 tools/compare_runtime/pass16/release/measure.py comparisons
python3 tools/compare_runtime/pass16/release/measure.py components
python3 tools/compare_runtime/pass16/release/measure.py checks
python3 tools/compare_runtime/pass16/release/measure.py metal-host
python3 tools/compare_runtime/pass16/release/summarize.py
```

The control's module files come from ONNX commit `0803d3e`; candidate builds use
the normal go.mod, and both versions are checked to reject replace directives.
Validation uses a separate identical candidate module snapshot. Backend tests
read the published module cache, without editing the sibling repository.
Metal-tagged checks require a suitable macOS host and CGO.
The `metal-host` phase reruns them with actual host GPU access when a sandbox
reports no default Metal device; run that phase outside such a sandbox. Both
the initial sandbox logs and the host rerun are retained, with explicit skip counts.

Working binaries and full profiles remain in ignored
`.cache/compare-runtime/pass16-release/`. Preserve original working files before
replay. New observations must not overwrite the committed historical evidence
or be described as the original samples. Packaging verifies unchanged source,
model/input/runner fingerprints, all outputs, sample counts, expected allocation
transfer and required validation before writing the committed bundle.

The native backend's first Metal run had one model resource-stress assertion
failure. The command ledger preserves it and the exact predefined follow-ups:
five isolated stress repetitions per version and one serialized full Metal suite
per version. All follow-ups pass; the failure is unexplained and is not deleted
or counted as a passing run. Packaging explicitly checks that this is the only
recorded failure and requires those follow-ups. See the report for this limitation.
