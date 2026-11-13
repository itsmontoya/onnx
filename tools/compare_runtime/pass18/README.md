# Pass 18: bounded private destination reuse

[PERFORMANCE_PASS18.md](../../../PERFORMANCE_PASS18.md) is the complete 24-part
report. [DESIGN.md](DESIGN.md) records ownership, accounting, alternatives and the
fourteen-question gate BEFORE production implementation. No backend production
change or dependency replacement was necessary.

- `data/summary.json`: fresh environment, module provenance, complete estimator
  inputs, medians/ranges/CVs, allocations, prototype/cold/amortized/workload results,
  retention and reference movement. All valid repetitions are retained.
- `data/validation.json`: recorded commands, overrides, exits and durations. The
  first prototype's manually launched full-suite rejection is included with an
  explicitly unavailable duration; its exit and full log are preserved.
- `data/evidence.json.gz`: lossless JSON with every fixed-policy report and raw
  engine timing, separate purego/initial reports, benchmark/profile/check logs,
  original/prototype/measured source snapshots and hashes, and diagnostic sources.
  Hardware serial/UUID output is excluded. No binaries are committed.
- `prototype.py`, `destination_storage.go.txt`, `destination_budget.go.txt`,
  `destination_storage_test.go.txt`: revision-2 source overlay prototype; these
  files do not enter ordinary production or tests.
- `benchmark_test.go.txt`: diagnostic cold, amortized, alternating/concurrent,
  failure/cancel-recovery, dynamic-shape and allocation/clear/copy benchmarks.

The historical first prototype and its rejection are evidence, not a production
candidate. Revision 2 preserves the existing intra-Run allocator, adds finite
Session-owned exact-shape slots, proves live headroom and charges retained bytes
in addition to legacy invocation reservations. Dynamic geometry and insufficient
headroom use ordinary allocation. Returned output cloning remains intact.

Replay sequentially from this repository with Go 1.27.2, published NeuralNetwork
v0.10.10, the pinned MNIST archive/model and Python dependencies recorded in the
summary. The fixed-policy driver, inputs, conversion, tolerances, top-1/upstream
checks, ten warmups, 100 calls, Go-then-ORT engine order and primary estimator are
unchanged. Ordinary ramp and diagnostic scopes remain separate. Cold/session
setup is explicitly timed in its own scopes. Two-worker results are throughput,
not individual latency. Compilation/loading/serialization stay outside inference.

Use a NEW output directory: replay must never overwrite original observations.
`prepare` restores control root sources from baseline `2b2a0c5` through an overlay,
including removal of new candidate-only Go files. It therefore works at the
candidate revision too. The reusable Pass 17 preparation helper also builds an
incidental `baseline.test`; only the separately built `control.test` is measured.
No replace directive, module-cache edit or backend source edit is used.

```sh
export ONNX_PASS18_OUT=/private/tmp/onnx-pass18-replay
python3 tools/compare_runtime/pass18/measure.py prepare
python3 tools/compare_runtime/pass18/prototype.py
python3 tools/compare_runtime/pass18/measure.py candidate
python3 tools/compare_runtime/pass18/measure.py comparisons
python3 tools/compare_runtime/pass18/checks.py suites
python3 tools/compare_runtime/pass18/checks.py metal
# Execute with actual native GPU access; inspect device skips explicitly:
python3 tools/compare_runtime/pass18/checks.py metal_host
python3 tools/compare_runtime/pass18/diagnostics.py
python3 tools/compare_runtime/pass18/final_checks.py
# Packaging intentionally writes a new result under this checkout's data/;
# preserve the committed historical data before generating replay summaries.
python3 tools/compare_runtime/pass18/summarize.py
```

Reference Python/cache paths inherit the Pass 15 helper's documented environment
options: ONNX_REFERENCE_PYTHON, ONNX_REFERENCE_PYTHONPATH and
ONNX_BENCHMARK_GOCACHE. Defaults point to this investigation host. The ordinary
measurement environment has GOMAXPROCS=1, CGO_ENABLED=0, GOWORK=off, GOARM64=v8.0
and empty GOFLAGS except source overlays. Validation overrides are explicit.

The default, CGO-disabled, purego, race and purego+race suites, twelve cross-builds,
four fuzz targets and native Metal validation pass. Sandbox backend Metal has 46
device-unavailable skips; native serialized ONNX/backend suites have zero skips
and five isolated repetitions of the historical stress test pass. The earlier
Pass 16 unrestricted failure remains unexplained and is not declared resolved.

Profiles contain initial cache construction plus timed invocations and tiny-object
sampling/coalescing. Ordinary B/op and allocs/op, source ownership and controlled
removal establish the ledger; do not interpret profile object counts as exact
ordinary counters. Tensor allocator bytes are allocation-volume/size-class
predictions, not a process RSS measurement. Final source fingerprints separately
identify comment/format/test changes after the fixed-policy measured source.
