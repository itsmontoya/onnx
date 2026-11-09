# Functions and bounded nested control flow

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Conditional graph execution from test_if. Add pinned function and bounded Loop
fixtures before their production increments.

Byte-original Apache-2.0 fixture: [`test_if.onnx`](../../testdata/corpus/models/test_if.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 6/import 11, If-11; current graph validation rejects nested scopes and this profile.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

Define lexical nested scopes, captures, local functions and domain/version
resolution without leaking names between scopes. Stage pure local-function
expansion separately from If, then Loop/Scan. Impose
depth/node/value/iteration/live-byte limits and cancellation at
nested-node/iteration boundaries; branch output type/rank joins must be
explicit.

## Required library task

Runtime-owned graph/value/scheduling design task; no new numerical backend kernel is intrinsically required for If, but every branch operator must have published support. Function recursion/cycles and unbounded loops must fail deterministically.
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
