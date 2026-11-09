# Sequence/map/optional/sparse value model

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Identity mapping over a sequence and optional tensor extraction. No audited
map/sparse export has been selected; selecting licensed failing fixtures is a
blocking acceptance requirement for those increments.

Byte-original Apache-2.0 fixture: [`test_sequence_map_identity_1_sequence.onnx`](../../testdata/corpus/models/test_sequence_map_identity_1_sequence.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 8/import 17, SequenceMap-17; exact accepted envelope still rejects sequence values. OptionalGetElement fixture is IR 8/import 18 and additionally needs a profile audit.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

Design a tagged immutable Value model distinct from dense Tensor, with
copying/ownership and total element/depth/byte limits. Stage sequence first,
then optional, map and sparse independently. Define empty/container element
metadata, sparse indices ordering/duplicates, map keys and nested output
selection before operators.

## Required library task

Separate backend value-model/storage investigation; do not force containers/sparse values through dense tensor APIs. Publish needed storage contracts or document runtime-owned immutable metadata before adapters. SequenceMap also depends on nested-scope bounded execution.
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
