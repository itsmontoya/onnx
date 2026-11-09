# Float16 and bfloat16 value/storage design

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Low-precision conversion workload. Add a named bfloat16 export and failure
fixture before implementing that separate format.

Byte-original Apache-2.0 fixture: [`test_cast_FLOAT_to_FLOAT16.onnx`](../../testdata/corpus/models/test_cast_FLOAT_to_FLOAT16.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 10/import 21, Cast-21; this newer envelope stays negative.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

Define raw-bit storage and explicit conversion APIs, round-to-nearest
ties-to-even, subnormals, signed zero, infinity/NaN behavior and overflow.
Decide accumulation type independently for each kernel; LayerNormalization
stash_type=1 does not add low-precision input support. Separate float16 and
bfloat16 implementations.

## Required library task

New backend storage/conversion task and published release before runtime decode or kernels. No promotion through float32 wire storage may lose original bits; exact roundtrip fixtures required.
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
