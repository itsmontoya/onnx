# RT-15 resource, error and lifecycle audit

Verified on 2026-11-08 with Go 1.27.2, Darwin/arm64, against RT-14 base
`cdcfb6fdb86e7331587670c83389d6c9558540ba`. Implementation and test revision:
[`e33db19db232dab2c78ccc1a334f7c7bcb079053`](https://github.com/itsmontoya/onnx/tree/e33db19db232dab2c78ccc1a334f7c7bcb079053).
The module remains Go 1.26.1 with toolchain Go 1.27.2, neuralnetwork v0.7.0
(`6048452c57a0a293db4591cd2e2febd7d6b743a3`) and protobuf v1.36.12.
The public APIs, IR 8/opset 13 envelope and 25 implemented operators are unchanged.

## Findings and changes

Graph value limits previously failed as each producer was inserted, allowing an
initializer to be converted before a later output exceeded MaxValues. The new
preflight counts distinct input/initializer/node-output names before conversion
or external I/O. A default reuses its input name and empty optional outputs are
absent; the existing builder still rejects duplicate producers. Count rejection
can now precede a malformed tensor/reference error in an oversized graph.

Decoded payload traversal used protobuf reflection Range, whose order is
unspecified. It now walks fields in descriptor order, retaining the same checked
charges and context checks while making the first overflowing field stable.
Node input/output slots, runtime input assembly, and both Concat inference passes
and runtime shape assembly now poll cancellation per record.

No additional numerical buffers or kernels were introduced. The existing
runBudget accumulates simultaneous workspace charges: shape/axes staging remains
charged while reserving its destination. Workspace is released on every node
exit. Shared model constants count once per run, including overridden defaults;
inputs, aliases, all intermediates and requested output clones count separately.
Plan snapshots retain their conservative 256-byte fixed numeric charge per Step,
including CNN configuration. The pinned backend's copying M1 paths and direct
M2/M3 Into paths still match the [scratch table](runtime-contract.md#limits-and-cancellation).

## Limit evidence

All fourteen zero-field defaults and negative/positive/host-arithmetic validation
are directly checked by [TestLimitsDefaultsAndInvalidFields](../limits_test.go).
The following enforcement cases use small exact/one-under configured limits;
they exercise the same paths selected by defaults without deliberately allocating
default-sized heaps or triggering process OOM.

| Resource and finite default | Direct enforcement evidence |
| --- | --- |
| MaxModelBytes 64 MiB; MaxRecursion 64 | [protobuf tests](../internal/decode/protobuf_test.go): exact/oversize bytes, read limit+1, malformed packed fields, known nesting, raised recursion, unknown-group boundary; [public loader tests](../model_test.go) retain actual noncanonical wire length and causes |
| MaxLoadBytes 1 GiB | [payload widths](../internal/decode/payload_test.go), [conversion/aggregate accounting](../internal/decode/tensor_limits_test.go), [graph snapshots](../internal/graph/limits_test.go), [compiler/executable snapshots](../internal/operators/plan_test.go), and [exact public loads](../model_test.go) |
| MaxInitializerBytes 256 MiB; MaxTensorBytes 64 MiB | [tensor validation and conversion](../internal/decode/tensor_decoder_test.go), [shared initializer/Constant aggregate](../internal/graph/limits_test.go), [runtime input/output limits](../session_test.go) and M1/M2/M3 adapter boundary tests |
| MaxNodes 100,000; MaxValues 200,000; MaxAttributes 400,000 | [whole-graph limits](../internal/graph/limits_test.go), including aggregate attributes and the new pre-conversion value-count allocation regression; [public enforcement](../model_test.go) |
| MaxNameBytes 4,096 | [graph limits](../internal/graph/limits_test.go): names, symbols, producers, documentation, metadata and scalar/repeated attribute byte strings |
| MaxRank 32; MaxDimension 2^31-1 | [tensor validation](../internal/decode/tensor_decoder_test.go), [metadata/empty-axis limits](../internal/graph/limits_test.go), [static planner limits](../internal/operators/plan_test.go), [runtime rank/dimensions](../session_test.go) and adapter output bounds |
| MaxLiveBytes 512 MiB; MaxWorkspaceBytes 256 MiB | [checked cumulative reservations](../run_budget_test.go), [M1 scratch](../numerical_call_test.go), [M2 simultaneous staging/destination](../shape_call_test.go), [M3 destination/publication](../cnn_call_test.go), [resident constants/aliases and allocation rejection](../session_test.go) |
| MaxSessions 16 | [default cap and exact slot reuse](../model_test.go), model Close/NewSession races, [overlapping Run/Close and independent sessions](../session_test.go) |

The new allocation regressions measure rejected calls with a 1 MiB payload.
Whole-graph count rejection stays below 32 KiB allocated per operation, preceding
both raw staging and initializer ownership. Rejected scratch, intermediate, and
requested-output-copy reservations also stay below 32 KiB per operation, rather
than allocating their 1 MiB numeric destination first. These are focused regression
thresholds, not a general allocation or RSS guarantee. Successful retries and
retained output values are checked afterward.

External conversion still reserves three payload widths before resolution/read:
raw range bytes, typed staging and backend ownership. Typed/raw inline conversion
reserves one/two additional widths respectively. Failed or cancelled conversion
commits no new decoder usage. [External range tests](../internal/decode/external_range_test.go)
cover exactly-once reader closure, including readers returned with resolution
errors, joined resolve/read/close causes, partial/short/invalid/no-progress reads,
zero-byte ranges and cancellation. Reads use at most 32 KiB chunks and reject 100
consecutive empty reads. Public reader loads consume at most MaxModelBytes+1 and
retain caller ownership. Successful plans retain no wire object or resolver.

These are logical payload reservations. Go protobuf constructs its object tree
before semantic count checks; object headers, allocator/GC retention, slice
capacity and caller-retained outputs from earlier runs are excluded. The finite
wire/known-recursion limits restrict decoder input but do not establish a hard
heap quota. No second wire parser or arbitrary panic recovery was added.
MaxRecursion does not control unknown groups: the pinned official decoder's fixed
10,000-descent allowance accepts 10,001 nested group levels and rejects 10,002,
as the existing direct boundary test demonstrates.

## Cancellation, errors and lifecycle

[Load](../model_test.go), [decoder](../internal/decode/cancel_context_test.go),
[graph](../internal/graph/cancel_context_test.go), [compiler](../internal/operators/cancel_context_test.go),
[session](../session_test.go), [M1](../m1_integration_test.go), [M2](../m2_integration_test.go)
and [M3](../m3_integration_test.go) tests cancel at every observed context poll,
require nil partial outputs and preserve context.Canceled/DeadlineExceeded causes.
The new M2 cases cover all eight shape handlers and compare retained outputs
against their independent fixtures after cancellation and successful retry.
The M3 composed CNN and numerical inference/external MLP tests exercise shared
immutable weights, batches 1 and 3, two independent sessions, retained outputs
after failures/later runs/Close, and lifecycle contention under the race detector.

Backend v0.7.0 supplies synchronous kernels, copies and shape helpers without a
context/check callback. Polls bracket these calls; no in-kernel interruption or
cancellation-latency guarantee is claimed. Official protobuf unmarshal and deep
snapshot/copy accessors similarly complete between polls. Caller Resolve/Read
must cooperate to interrupt a blocked call. Runtime-owned conversion/Relu repair
loops poll every 1,024 elements; structural/input/Concat loops poll per record.
The final node and final output copies are checked before publication.

Errors retain their classification and wrapped causes through errors.Is/As.
Existing tests cover deterministic unexpected-name selection, load/run stage,
node index/name/domain/operator/opset, concrete input shapes, value context,
filesystem and resolver failures. Descriptor-order traversal adds a repeated
first-overflowing-field regression; graph fuzzing compares repeated failure text.
Failed loads publish no Model, failed runs discard private state, and successful
Close releases exactly one slot and is idempotent. Model.Close prevents new
sessions while existing sessions remain usable. Run/Close overlap returns ErrBusy.

DirectoryResolver still uses os.Root during open, confined relative POSIX paths
and regular-file ranges. It remains unsupported on js/plan9; os.Root does not
isolate hard links, mounts or filesystem boundaries. The [RT-14 report](integration-rt14.md)
retains the complete resolver/checksum policy; its racing-replacement and lifecycle
tests are rerun unchanged.

## Bounded fuzzing and checks

Four native Go targets run offline and never invoke an external resolver:

| Target | Scope and finite limits |
| --- | --- |
| [FuzzProtobufEntrypoints](../internal/decode/protobuf_fuzz_test.go) | Model/tensor byte-versus-reader agreement; 16 KiB wire, known recursion 16 |
| [FuzzTensorDecode](../internal/decode/tensor_decoder_fuzz_test.go) | Conversion validity and failed-reservation rollback; 16 KiB wire/tensor, 128 KiB load, 32 KiB initializers, rank 8, dimension 1,024, recursion 16 |
| [FuzzGraphReferences](../internal/graph/graph_fuzz_test.go) | Missing/forward/self references, structural validation, deterministic failure and usage; same tensor/shape bounds, 256 KiB load, 32 nodes/attributes, 64 values, 256-byte strings |
| [FuzzShapeAttributePlanning](../planning_fuzz_test.go) | Public load/shape/attribute planning and bounded no-input runs; 512 KiB load, 128 KiB live, 64 KiB workspace, one session, all remaining finite bounds as above |

Seeds use actual Identity, branch/default, scalar/empty/exact-int64/nonfinite,
Reshape/Transpose/Concat/Conv/pooling/Constant fixtures plus literal malformed
wire, missing/self references, repeated permutation axes, wrong discriminators
and extreme signed axes. Inputs over 16 KiB are skipped and are outside these
fuzz targets' scope. Normal tests run seed corpora without fuzzing. Fuzz results
are robustness evidence, not expanded ONNX compatibility claims.

Each actual smoke used GOMAXPROCS=2, -parallel=2 and -fuzztime=3s: protobuf
109,984 executions, tensor 117,421, graph 17,714 and planning 144,789; all passed.
The [Go workflow](../.github/workflows/go.yml) runs the same four bounded smokes.
See [development commands](development.md#resource-hardening-and-fuzzing).

Passed gofmt, tidy-diff, module verification, build, vet, uncached normal,
CGO_ENABLED=0, CGO_ENABLED=0 purego and race tests with cached offline dependencies
and GOCACHE=/private/tmp/onnx-runtime-gocache. Protobuf generation with protoc 33.2
and all profile/M1/M2/M3/conformance/inference/external generators reproduced
unchanged artifacts; the pinned conformance archive was local. Expected upstream
reference rejections/differences remain recorded as such. Ordinary tests require
no Python. Local Markdown/diff and handwritten spacing/type-colocation checks
pass; hosted Linux results are attached to the stacked PR.

RT-15 completes hardening for the existing M1–M4 scope. RT-18 usage/compatibility
documentation remains next in the roadmap; RT-16/NN-13 optimization and further
operators/versions remain separate. No release readiness or new licensing claim
is made; this repository still has no LICENSE. Keep the RT-15 PR open on RT-14.
