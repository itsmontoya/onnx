# Pass 15 released v0.10.9 integration

This evidence validates the published module, separately from the earlier local
candidate and descriptor experiments. See [the release report](../../../../PERFORMANCE_PASS15_RELEASE.md).

The control is released v0.10.8 through a separate module snapshot. The candidate
uses the normal go.mod with v0.10.9. Neither module uses a replace directive.
Published source fingerprints match the four production files in the tested
candidate, and the v0.10.9 tag resolves to merge commit
`740a244bf56214bb3443a69287a2d800674620e4`.

- `data/summary.json`: fresh environment/provenance, engine medians, ranges/CVs,
  paired differences, node/allocation/loading measurements and target calculations.
- `data/evidence.json.gz`: lossless JSON with all fourteen comparison reports,
  every inference timing/output bit, original benchmark/check logs and three ORT
  node-profile sample sets. Decode with Python `gzip.open` and `json.load`.
- `data/validation.json`: commands, environment overrides, exit codes and durations.

Full binaries and profiles remain outside git in ignored
`.cache/compare-runtime/pass15-release/`. No earlier local sample is pooled into
the released-module estimator. Every valid released timing sample is retained.

Run sequentially from the repository root using the report's Go and Python/ORT
versions. Configure `ONNX_REFERENCE_PYTHON`, `ONNX_REFERENCE_PYTHONPATH` and
`ONNX_BENCHMARK_GOCACHE` for your environment, as described in the parent README.
The scripts collect macOS host information and reuse the existing pinned model
and unchanged comparison driver:

```sh
"$ONNX_REFERENCE_PYTHON" tools/compare_runtime.py --report .cache/compare-runtime/prepare-pass15-release.json
python3 tools/compare_runtime/pass15/release/measure.py prepare
python3 tools/compare_runtime/pass15/release/measure.py comparisons
python3 tools/compare_runtime/pass15/release/measure.py components
python3 tools/compare_runtime/pass15/release/measure.py checks
python3 tools/compare_runtime/pass15/release/summarize.py
```

The policy remains nineteen sorted cases, ten warmups, 100 timed calls per case,
owned outputs, Go then ORT, unchanged input generation/conversion/upstream/top-1
checks, atol=1e-4 and rtol=1e-5. Builds, loading and serialization remain outside
inference timing. Ordinary-ramp and coarse profiled-node results remain separate.
