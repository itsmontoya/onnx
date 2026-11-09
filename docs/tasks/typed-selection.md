# Bool/int32, comparisons and Where

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Conditional feature masking represented by Where; record separate comparison and
int32 fixtures before their own production changes.

Byte-original Apache-2.0 fixture: [`test_where_example.onnx`](../../testdata/corpus/models/test_where_example.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 8/import 16, Where-16; import16 is currently rejected and needs its own audit or a separately sourced accepted-profile model.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

Design exact dense bool/int32 constructors/accessors, wire decode, metadata,
defaults, Constant, budgeting and output copying before operators. Then choose
Equal/Less/Greater and float32/int64 Where with boolean conditions and
multidirectional broadcasting. Define mixed-type rejection, NaN comparisons and
exact integer extrema.

## Required library task

Separate neuralnetwork value/storage task plus bool comparison and typed Where Into kernels; existing float32/int64 runtime storage cannot represent bool/int32 faithfully. Published release is required.
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
