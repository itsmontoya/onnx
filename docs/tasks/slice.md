# Slice for bounded feature cropping

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Feature-window cropping; the named upstream test is the initial failing
workload, not an enabled production application.

Byte-original Apache-2.0 fixture: [`test_slice.onnx`](../../testdata/corpus/models/test_slice.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 7/import 13, Slice-13; upstream bytes remain unsupported until a separately audited profile accepts this envelope.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

Select IR 8/import 13 and17 adapters first. Slice uses int64 starts/ends,
optional axes/steps inputs; preserve omitted slots. Normalize negative
axes/indices, clamp positive and negative-step boundaries, reject zero steps and
duplicate axes, handle int64 sentinels and empty outputs without overflow.

## Required library task

Published backend has no audited Slice Into contract. Add a separate neuralnetwork task for checked SliceShape and disjoint float32/int64 SliceInto, read-only inputs, exact int64 payloads, empty dimensions and no hidden copies. Runtime work waits for its published release.
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
