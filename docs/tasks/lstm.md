# LSTM after recurrent state and gate contracts

Status: **future gap, not implemented or selected by RT-19**.

## Named workload and failing fixture

LSTM default sequence evaluation. Later bidirectional/layout/peephole selections
need new model evidence.

Byte-original Apache-2.0 fixture: [`test_lstm_defaults.onnx`](../../testdata/corpus/models/test_lstm_defaults.onnx).
Its exact source/hash, inputs/outputs, nodes, attributes and deterministic failures
are recorded in [the corpus index](../../testdata/corpus/index.json).
IR 7/import 14, LSTM-14; no profile is silently converted.
Ordinary Go corpus tests currently require `ErrUnsupported`. A failed profile
check is an additional prerequisite, not operator coverage evidence.

## Contract and dependencies

RT-19 corpus audit and RT-17 profile policy; published backend work listed
below. RNN task must complete first; review GRU separately.

Depends on RNN state/layout contracts; review GRU results before expansion.
Define ONNX i/o/f/c gate order, W/R/B partitions, hidden/cell state ownership,
input_forget, peepholes, clip and activation defaults. Preserve positional
Y/Y_h/Y_c outputs; start forward layout=0 float32, no peepholes.

## Required library task

Separate published neuralnetwork LSTM task including explicit gate packing, cell-state precision, exact scratch/lifetime bounds and cancellation per timestep.
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
