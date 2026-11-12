# Pass 14 released dependency integration

The ONNX integration uses the published neuralnetwork v0.10.8 module, without a
replacement. The control uses v0.10.7 through a separate module file. Read the
[complete report](../../../PERFORMANCE_PASS14.md) before interpreting timings.

- [summary.json](data/summary.json): medians, ranges, CVs, allocation results,
  prepared-node costs, ORT node sums, release provenance and source fingerprints.
- [evidence.json.gz](data/evidence.json.gz): lossless JSON containing all fourteen
  original comparison reports, individual inference samples and output bits,
  benchmark logs, three sets of ORT node-duration samples, validation logs and
  profile summaries. Decode with Python's `gzip.open` and `json.load`.
- [validation.json](data/validation.json): every command, environment override,
  exit code and duration. Base performance environment is in summary.json.
- [backend-investigation.json](data/backend-investigation.json) and its
  [identity](data/backend-investigation-identity.json): earlier, explicitly
  pre-release diagnostics that justified the backend change. Their source hashes
  match the released ReLU files. These are separate from the fresh integration.

Full test binaries, profiles and uncompressed working data stay in the ignored
`.cache/compare-runtime/pass14-release/`. They are not needed to inspect the
committed numerical evidence. Allocation/CPU profiles are collected separately
from latency measurements. No valid timing sample is discarded.

## Reproduction

These scripts target the measured Apple M3/macOS environment. Use Go 1.27.2 and
the report's Python/NumPy/ONNX/ORT versions. Set `ONNX_REFERENCE_PYTHON`,
`ONNX_REFERENCE_PYTHONPATH` and `ONNX_BENCHMARK_GOCACHE` to your own paths; defaults
retain the original measurement environment. Run commands sequentially from the
repository root, without simultaneous performance processes.

The exact measured comparison driver and Go runner are included at
`tools/compare_runtime.py` and `tools/compare_runtime/{runner,run_result,tensor_data}.go`.
Their bytes match the source fingerprints in summary.json. The Python driver also
matches archived commit `c2d5123`; the current Go runner uses separate files for
its types. No benchmark logic or timing boundary was changed when preserving
these files for replay.

Prepare the pinned model with the existing driver before running benchmark
binaries. This preliminary report is outside the seven-pair measurement set.
An optional `--archive` argument accepts the pinned local MNIST archive.

```sh
"$ONNX_REFERENCE_PYTHON" tools/compare_runtime.py --report .cache/compare-runtime/prepare-pass14.json
python3 tools/compare_runtime/pass14/measure.py prepare
python3 tools/compare_runtime/pass14/measure.py comparisons
python3 tools/compare_runtime/pass14/measure.py components
"$ONNX_REFERENCE_PYTHON" tools/compare_runtime/pass14/ort_profile.py
python3 tools/compare_runtime/pass14/measure.py checks
python3 tools/compare_runtime/pass14/summarize.py
```

The fixed-policy driver remains unchanged: nineteen sorted cases, ten warmups
and 100 timed calls each, owned outputs, Go then ORT, unchanged inputs/conversion,
upstream and top-1 checks, atol=1e-4 and rtol=1e-5. The summary uses differences
of engine medians; paired differences and ordinary-ramp timing remain separate.
The ORT node profile uses a separate ramp workload and integer-microsecond events.
