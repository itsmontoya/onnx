# Expand for broadcast parameter grids

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

Broadcast parameter-grid expansion, represented by the named failing upstream
model.

Byte-original Apache-2.0 fixture: [`test_expand_dim_changed.onnx`](../../testdata/corpus/models/test_expand_dim_changed.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 7/import 13, Expand-13; a profile task is independently required for the unchanged fixture.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below.

Target dense float32/int64. Shape is a rank-one int64 tensor; bidirectional
broadcast result may exceed input rank. Check every target dimension, zero/one
rules, rank and dimension multiplication before allocating. Dynamic shape values
remain Run errors and recover.

## Required library task

New backend task: checked ExpandShape and disjoint ExpandInto with direct bounded copying. Publish backend APIs and allocation audit before runtime adapter integration.
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
