# RT-14 external MLP

`models/mlp.onnx` preserves RT-11's MLP graph and numerical weights, moving its
four float32 initializers into `models/weights.bin`. The shared file begins with
a 13-byte prefix. Every tensor has a nonzero offset; the final tensor omits length.
The actual ONNX 1.16.2 checker and external-data loader accept the fixture, and
its reference evaluator matches the inherited batch 1 and 3 goldens exactly.
Inputs and goldens remain in `testdata/inference`; no inherited artifact changes.

`provenance.json` records model, source model, weight file, each selected range,
inputs and outputs with SHA-256 hashes. These development provenance hashes do
not implement ONNX's optional whole-file SHA-1 checksum metadata. Runtime
checksum metadata is explicitly unsupported.

Ordinary Go tests are offline and Python-free. Regenerate or verify using the
pinned reference environment documented in `docs/development.md`:

```sh
python tools/check_external.py --write
python tools/check_external.py
CGO_ENABLED=0 go run ./examples/external -batch 1
CGO_ENABLED=0 go run ./examples/external -batch 3
```

The example explicitly allows `weights.bin` through an immutable memory resolver.
The repository has no LICENSE; no separate license is declared for these original
fixtures. The upstream ONNX specification and protobuf schema retain their
recorded Apache-2.0 attribution.
