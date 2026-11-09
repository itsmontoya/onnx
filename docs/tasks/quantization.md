# Quantization with exact rounding and scale contracts

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Static affine quantization conversion, represented by the copied failing export;
no quantized network is currently enabled.

Byte-original Apache-2.0 fixture: [`test_quantizelinear.onnx`](../../testdata/corpus/models/test_quantizelinear.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 10/import 21, QuantizeLinear-21; preserve this newer dtype/profile rejection.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

First design uint8/int8 exact value storage and schema-specific
QuantizeLinear/DequantizeLinear adapters. Specify ties-to-even rounding,
zero-point addition, saturation, positive finite scales and per-axis axis
normalization. Block quantization, dynamic quantization and integer matmul
remain separate selections.

## Required library task

Separate backend quantized storage/conversion task and published kernels with per-axis scales and saturation tests at every rounding boundary. No float32 approximation may claim quantized arithmetic support.
The adjacent backend checkout stays read-only. Use a published dependency;
no local replace, private backend patch or provider framework is permitted.

## Acceptance and handoff

- Select a licensed named model and smallest exact schema/type/rank subset; keep
  original source bytes and metadata, including the failing fixture above.
- Add separate accepted-profile isolated fixtures without changing upstream
  version metadata. Compare pinned ONNX 1.16.2 and ORT 1.19.2 observations
  independently; preserve actual differences/rejections rather than skipping.
- Implement only the selected subset, with static Load checks where known and
  contextual runtime errors/recovery for dynamic values. Keep other attributes,
  types and modes explicit gaps in the canonical capability manifest.
- Preflight destinations, staging, scratch, resident and snapshot bounds before
  allocation. Verify immutable inputs, owned outputs, optional positional slots,
  branches/joins, cancellation and concurrent sessions with meaningful tests.
- Run offline Go checks in default/cgo-disabled/purego/race modes, regenerate the
  support table and provenance, and record published library/source revisions,
  model hashes, numerical results and remaining gaps in an independent handoff.

This file scopes a later independent task; it grants no current compatibility
claim or instruction to implement the entire initiative in RT-19.
