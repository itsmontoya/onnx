# MNIST performance pass 5 — 2026-11-10

This investigation records the v0.10.1 baseline. The subsequent
[v0.10.2 upgrade and retest](v0.10.2/README.md) measures a 17.2% inference latency
reduction from the published residual-column kernel and a 3.77× gap versus ORT.

The highest-leverage next change is to extend ARM64 convolution acceleration to
the columns that still use `tensor.conv2DFourChannels`. Geometry-matched timings
put these columns at approximately **84.7 µs, or 42% of inference**. The second
convolution's two fully interior tail columns alone cost **31.9 µs**.

This change belongs in `github.com/itsmontoya/neuralnetwork`. Following the
requested boundary, this pass changes **diagnostics and documentation only**.
There is no production optimization or claimed speedup. The complete handoff is
[neuralnetwork-prompt.md](neuralnetwork-prompt.md).

## Baseline and final verification

Baseline: ONNX revision `85bc301`, published neuralnetwork **v0.10.1**, upstream
commit `ea17b585c42264ca85a637705608c284b80464bc`. The module has no replacement
in the inference measurements. Hardware is Apple M3, macOS arm64; Go 1.27.2,
`GOARM64=v8.0`, `GOMAXPROCS=1`, `CGO_ENABLED=0`, `GOWORK=off`.
Python 3.9.6, NumPy 1.26.4, ONNX 1.16.2, ORT 1.19.2, protobuf 4.25.3.

The unchanged comparison driver uses CPU ORT, sequential execution, one
intra-op/inter-op thread, **graph optimization disabled**, 10 warmups and 100
measurements for each of the original 19 cases: upstream, zeros, ones and 16
seeded random inputs. Model SHA-256:
`310384d2500a063def1ed3ad5b448baeda1edd5a570f137cb82cb11b7c25c05d`.
All model, source, input, driver and runner hashes match the previous pass.
No case counts, tolerances, input preparation or numerical precision changed.

Seven complete, serial driver invocations yield 13,300 samples per runtime:

| Measurement | Go | ORT |
| --- | ---: | ---: |
| Median of the seven run medians | 204.771 µs | 42.500 µs |
| Range of run medians | 197.896–205.813 µs | 41.708–44.959 µs |
| Sample standard deviation of run medians | 2.670 µs | 1.283 µs |
| Coefficient of variation of run medians | 1.31% | 2.98% |
| Median of run p95 values | 277.281 µs | 46.334 µs |

The **ratio of these medians is 4.818×**. The median of the seven individually
paired ratios is **4.745×**, ranging 4.562–4.876×. These are different statistics;
neither indicates a regression against the earlier single-run 4.64× result.
Raw samples are retained in [the seven compressed reports](data/comparison-1.json.gz)
(`comparison-1` through `comparison-7`), with
[run summaries](data/comparison-summary.json) and [variance calculations](data/statistics.json).

The original, unmodified `BenchmarkMNIST` uses its existing deterministic
`index%256 / 255` input. Nine 1-second repetitions in a precompiled binary give
**201.927 µs/op** median, range 196.486–204.426 µs, SD 2.496 µs, CV 1.24%,
**179,440 B/op and 174 allocations/op**. Its input differs from the comparison
driver's 19-case distribution; kernel diagnostics use this benchmark input
unless explicitly described as synthetic.

| Before → final verification | Result |
| --- | --- |
| Go driver median | 204.771 → 204.313 µs |
| Final ORT median | 44.042 µs |
| Final slowdown | **4.639×** |
| Go allocation cost | **179,440 → 179,440 B/op; 174 → 174 allocs/op** |
| Production behavior | Unchanged |

The first Go number aggregates seven runs; final verification is one additional
1,900-call run. ORT variation explains the ratio difference. Seven further
alternating binary pairs reverse order each repetition and find a median paired
diagnostic-minus-baseline difference of **+0.106 µs**. This is noise, not a gain.
See [confirmation samples](data/confirmation-summary.json) and
[the final comparison](data/final-comparison.json.gz).

## Approximate accounting of the gap

Prepared node measurements retain the real intermediate tensors and disjoint
destinations. Five 300-ms repetitions per node give the following medians.
ORT numbers are **instrumented node-event means**, rounded by its µs-resolution
profiler, after discarding ten warmups; they include instrumentation overhead.
They are useful cross-checks, not interchangeable with Go microbenchmarks.

| Class | Go prepared time | Share of 201.927 µs | ORT profiled time |
| --- | ---: | ---: | ---: |
| Two convolutions | 118.482 µs | 58.7% | 34.47 µs |
| Three additions, principally broadcast channel biases | 26.047 µs | 12.9% | 2.45 µs |
| Two max pools | 20.852 µs | 10.3% | 3.07 µs |
| Two ReLUs including ONNX NaN correction | 16.791 µs | 8.3% | 1.76 µs |
| Matrix multiplication | 1.724 µs | 0.9% | 1.01 µs |
| Remainder by subtraction | 18.031 µs | 8.9% | Not independently comparable |

The remainder includes binding, constant clones, reshapes, destination
allocation/retirement, metadata, validation, publication and GC effects.
Independent medians do not form an exact additive trace.

For scale, the baseline benchmark is approximately **159.4 µs above ORT's
42.5 µs**. The categories above suggest approximately 84.0 µs extra convolution,
23.6 µs extra additions, 17.8 µs extra pooling, 15.0 µs extra activation work,
0.7 µs extra MatMul, and about 18.4 µs residual. This is approximate accounting,
not a matched subtraction of identically instrumented executions. It supports
the conclusion that **slower numerical traversal is the main difference**, with
allocation and framework work secondary on this small graph.

[Raw node measurements](data/nodes-bench.txt),
[all benchmark summaries](data/benchmark-summary.json), and
[ORT graph/events/settings](data/ort-diagnostic.json) support these estimates.

## Convolution: acceleration coverage matters more than its fastest loop

Both convolutions use 5×5 kernels, padding 2 on every side, unit stride/dilation,
one group, and nil bias. Their separate Add nodes supply bias:

| Node | Input | Weights | Output | Real-tensor prepared time |
| --- | --- | --- | --- | ---: |
| 0 | `[1,1,28,28]` | `[8,1,5,5]` | `[1,8,28,28]` | 18.194 µs |
| 4 | `[1,8,14,14]` | `[16,8,5,5]` | `[1,16,14,14]` | 100.288 µs |

The dispatch path is `Conv2DInto → conv2DValuesInto →
conv2DChannelBlocksInto`. Four valid adjacent columns use `conv2DFourByFour →
conv2DInteriorTile`, whose assembly executes NEON `VFMLA`. Residual columns use
`conv2DFourChannels`: four **scalar** accumulators across four filters.

For width 28, the SIMD tile starts are 2, 6, 10, 14, 18, 22; scalar columns are
0, 1, 26, 27. For width 14, tile starts are only 2 and 6; scalar columns are
**0, 1, 10, 11, 12, 13**. Columns **10 and 11 are fully interior** but cannot
form another non-overlapping four-column tile under the current traversal.

The second convolution therefore executes **42.86% of output columns** and
**37.5% of valid multiply-accumulates** in the scalar helper. This is not merely
padding overhead. The first convolution has 14.29% scalar output columns and
10.45% scalar valid multiply-accumulates.

To test this explanation without implementing an optimization,
[conv_paths_test.go.txt](conv_paths_test.go.txt) invokes the **existing** helpers
on disjoint column partitions. It uses finite synthetic payloads with the exact
MNIST geometry, includes geometry setup and destination stores, and verifies
that scalar plus tiled output matches `Conv2DInto` bit for bit before timing.
Seven 500-ms repetitions give:

| Existing path | First convolution | Second convolution |
| --- | ---: | ---: |
| Complete synthetic kernel | 17.477 µs | 99.482 µs |
| All scalar columns | **6.269 µs** | **78.400 µs** |
| Padded columns only | 6.323 µs | 46.881 µs |
| Interior tail columns only | None | **31.870 µs** |
| SIMD interior tiles | 10.975 µs | 20.489 µs |

Overlapping rows in this table must not be added. Partition harness traversal
differs slightly from the full kernel; its two disjoint totals are close to
the complete timings. Scalar helper timings allocate **zero bytes**. Avoiding
their per-call allocations cannot explain or fix this cost.

The second convolution's scalar partition reaches approximately **5.0 GFLOP/s**
against **32.0 GFLOP/s** for its SIMD partition, counting two operations per
valid multiply-accumulate. These are effective algorithm-throughput figures,
not hardware peak or bandwidth measurements. The two full real-tensor
convolutions together reach about 11.3 GFLOP/s on 667,936 valid MACs.

Assembly and compiler analysis show five remaining inner-loop bounds checks
for the scalar helper: the input load and four filter loads at
`tensor/conv2d.go:288–292`. Four scalar `FMADDS` operations accompany address
arithmetic, loop branches and reloads. The existing interior assembly already
uses vector arithmetic. This implicates **scalar execution and repeated
indexing/control work**, rather than an entirely missing SIMD dispatch.
See [assembly](data/conv-objdump.txt) and [compiler excerpts](data/compiler-excerpts.txt).

Cache sensitivity is small in the existing diagnostic: real second-convolution
latencies are 100.147, 100.880 and 101.275 µs for 1, 64 and 1,024 independent
tensor sets. The largest set spans about 30.9 MiB of numerical payload.
This argues against cache residency being the principal limiter here; it does
not establish the absence of memory stalls or identify a particular cache.
`purego` full inference is 327.346 µs versus 201.927 µs in the default build,
with unchanged allocations, confirming substantial benefit from ARM64 dispatch.

## Distributed work and memory

[BenchmarkMNISTPipeline](../../../mnist_pipeline_benchmark_test.go) provides two additional
diagnostics, each measured seven times for 500 ms:

| Experiment | Median | B/op | Allocs/op |
| --- | ---: | ---: | ---: |
| New evaluation plus input/constant binding | 3.698 µs | 26,288 | 36 |
| Entire graph with prepared calls and persistent disjoint destinations | 187.449 µs | 41,920 | 33 |
| Ordinary complete inference, baseline | 201.927 µs | 179,440 | 174 |

PreparedGraph executes all twelve original numerical/reshape calls in order,
including ReLU correction and both reshape copies, and compares its final
values with ordinary execution. It excludes call planning, per-step validation,
destination allocation/retirement and output publication. It also retains all
intermediates; it is **not** a proposed production memory plan or a comparison
workload replacement.

Removing roughly 77% of allocated bytes and 81% of allocations in that
experiment saves only **14.478 µs / 7.2%**. This is an evidence-supported estimate
of the combined planning/destination/framework opportunity, not a rigorous
upper bound: retained buffers, warm caches and GC behavior differ.

The allocation-space profile attributes approximately 71% to tensor construction
and 22.7% to `Float32Values` snapshots. ReLU correction copies and scans both
inputs on every inference; its two correction-only measurements total
**10.368 µs**, versus **6.332 µs** for the backend ReLUs. The snapshots alone
account for about 40.8 kB of allocator-rounded bytes per inference.
Constants are cloned every Run; binding includes about 24.0 kB of raw constant
payload. The two reshape operators copy another 11,264 raw payload bytes.
Current storage reuses one compatible retired destination within a Run and
retains none across Runs.

Repeated costs are spread across shape snapshots, call structs, argument slices,
broadcast shapes/strides, matrix adapters, tensor construction and validation.
The compiler confirms call-state escapes and the scalar bounds checks. These
are real aggregate opportunities even where no individual function dominates.
However, prepared execution's limited gain argues against building a large
arena or planner as this pass's first optimization. The post-benchmark live-heap
profile is mostly process/protobuf metadata after cleanup; it is **not** peak
inference memory and cannot be used to estimate a production arena size.

Nine alternating 2,000-call pairs with normal GC versus `GOGC=off` find a median
paired difference of **1.668 µs**, approximately 0.8% of full inference. Disabling
GC retains hundreds of MB and changes page-allocation behavior; this is only an
ablation, not a GC-free production design. The trace contains 48 GC cycles in
211.4 ms, about 1.48 ms total GC stop-the-world ranges and 10.72 ms mark-assist
ranges. Concurrent-mark phase elapsed time overlaps inference and must not be
counted as GC CPU time. These observations do not support GC causing most of
the remaining 160 µs gap.

In ORT, separately disabling both its CPU arena and memory pattern changes
the median of run medians from 42.459 to 44.083 µs; paired median difference is
2.104 µs. Arena-only disable is essentially unchanged. All seven trials are
retained, including an 84.188-µs disabled-setting outlier. ORT documents
[memory patterns and arenas](https://onnxruntime.ai/docs/api/c/struct_ort_api.html)
separately from graph optimization. Their small absolute effect in ORT does
not directly predict Go savings; the prepared Go experiment is stronger evidence.

## Graph work and parallelism

With optimization disabled, ORT retains all **12 nodes** and executes the same
operator sequence, although it can reorder the independent weight reshape.
It is not obtaining this baseline advantage by removing Add/ReLU nodes.
Its reshape implementation can alias storage, and its kernel machinery avoids
many Go allocations/copies; thus identical node counts do not mean identical
memory work. The pinned
[float convolution implementation](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/providers/cpu/nn/conv.cc)
calls MLAS, whose
[convolution implementation](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/core/mlas/lib/convolve.cpp)
supports blocked patch conversion and GEMM. The exact MLAS microkernel selected
by this wheel was not established by native profiling.

A separate optimized ORT session reduces the graph to **six nodes**: two
Conv/Add/ReLU sequences become `FusedConv`, MatMul/Add becomes Gemm, the constant
weight reshape disappears, and activation reshape remains. Latency changes from
42.459 to **40.042 µs**. This is a different experiment and never replaces the
disabled baseline. ORT's savings do not cap a possible Go fusion improvement:
Go's standalone bias and ReLU paths are substantially more expensive.

Potential Go fusion could remove expensive broadcast indexing, activation
snapshots and intermediate tensors. It requires preserving graph outputs,
fan-out, cancellation, budgets and NaN behavior. Constant folding could remove
the weight reshape, while reshape views require a backend ownership/aliasing
design. There are no Transpose operators in this workload and no evidence that
transposing its graph is the next win. Folding cannot remove its data-dependent
convolutions, additions or pooling.

Baseline ORT's measured CPU/wall ratio is approximately **1.00**; its trace has
no worker activity. The Go profiled process uses 3.59 seconds CPU in 3.79 seconds
wall including profiler startup/teardown. Baseline threading does not explain
the gap. Separate ORT experiments reach **30.0 µs** with two intra-op threads
and **24.5 µs** with four, at approximately 1.91 and 3.64 CPU/wall ratios.
These show potential parallel compute capacity, but change the comparison's
single-thread setting. There is little inter-op parallelism in this chain
beyond the independent weight reshape. No concurrency was introduced. A future
Go intra-op experiment must include worker synchronization and small-kernel
scheduling costs, and be reported separately from single-thread improvements.

## Profiling limits

The 12-second whole-inference CPU profile gives **80.8% to `runtime.kevent`**.
A second profile repeats this anomaly, and a no-GC profile instead overattributes
time to `madvise` compared with process system CPU time. The cause is unresolved.
Do **not** interpret those raw percentages as inference waiting, GC, memory
bandwidth or scheduler costs. Focused samples within `Conv2DInto` put the scalar
helper at about 36% of first-convolution and 77% of second-convolution samples,
consistent with the independently timed partitions. These conditional
percentages also are not direct end-to-end CPU percentages.

The execution trace's scheduler-delay totals overlap across goroutines and
exceed trace wall time; they cannot be added to inference latency. Its sync
profile shows benchmark/trace channel waiting, without evidence of Session lock
contention. `sample` cannot inspect the process under current access checks;
`xcrun xctrace` is unavailable without the developer tool. Hardware counters
were not available through the installed tooling. No cache-miss, instruction
retirement or hardware-peak claims are inferred from absent counters.

Portable tool binaries for `trace` and `objdump` are missing from this Go install;
their shipped sources work through `go run cmd/trace` and `go run cmd/objdump`.
Instrumented timings are separate from reported unprofiled benchmarks. Full
escape-analysis logs remain in `/private/tmp/onnx-pass5`; compact evidence,
raw profiles, trace, assembly and latency reports are retained under [data](data).

## Ranked opportunities

Estimates use the 201.927-µs baseline and Amdahl's law:
`speedup = 1 / ((1-f) + f/s)`. Improvement means latency reduction, not speedup.
Rows overlap and must not be summed. Forecasts are hypotheses to test.

| Rank | Opportunity and evidence | Expected end-to-end reduction | Complexity / correctness risk / confidence |
| --- | --- | --- | --- |
| **1** | **Accelerate residual convolution columns**. Existing-helper partition ≈84.7 µs / 42%; scalar indexing, five inner bounds checks and missing vector coverage, especially node 4. Extend clipped/tail SIMD or remove repeated checked scalar indexing with a proved span. | **≈21% at 2× residual-path speed**, 1.27× inference speedup; 31% at 4×. Two interior tail columns alone offer ≈13% if reduced from 31.9 to about 5 µs. | Medium; medium/high at padded/nonfinite edges; high bottleneck confidence, medium gain confidence. Backend boundary. |
| 2 | Fuse Conv/Add/ReLU for eligible single-consumer sequences. Current Add+ReLU calls cost ≈42.6 µs, with repeated materialization/snapshots; ORT optimized graph proves the pattern exists. Add a fused backend entry point plus ONNX planner eligibility. | Hypothesis **12–18%**, if 25–36 µs is removed after epilogue/semantic costs; overlaps rank 1 and ranks 3/5. No Go fusion prototype was timed. | High; high graph/NaN/budget risk; medium/low gain confidence. Joint library/runtime design. |
| 3 | Contiguous channel-bias broadcast specialization. Two bias Adds total 25.836 µs / 12.8%; assembly shows an indirect scalar operation call per element plus coordinate carry/stride bookkeeping. Hoist broadcast index work and use contiguous spans. | **≈10–11.5% at 5–10× bias-path speed**. | Low/medium; medium broadcasting/nonfinite risk; medium/high confidence. Backend boundary. |
| 4 | Pooling traversal specialization. Two max pools total 20.852 µs / 10.3%; per-window clipping, integer division, input NaN tests and scalar reduction recur. Hoist geometry or specialize common windows, preserving general fallback. | **≈5.2–7.7% at 2–4×**. | Medium; medium NaN/padding/ceil/dilation risk; medium confidence. Backend boundary. |
| 5 | Prepared metadata and destination reuse across Run. Current prepared experiment removes most allocations and planning and saves ≈14.5 µs; constants/reshapes add repeated copies. Prototype bounded per-session buffers with explicit lifetimes and limits before a general arena. | **About 3–7%** on this graph; memory savings more certain than latency. | High; high retained-output, alias, limits and cancellation risk; medium confidence. Runtime plus ownership considerations. |
| 6 | Further tune the already-vectorized convolution interior and tile setup. Geometry-matched tiles total ≈31.5 µs / 15.6%; first-convolution setup is a larger fraction than second-convolution setup. | **≈3–5%** at 1.25–1.5× tile-path speed; cannot fix uncovered residual columns. | Medium; medium assembly/portability risk; medium/low forecast confidence. Backend boundary. |
| 7 | Native ONNX-compatible ReLU or a backend correction API. Correction-only scans/snapshots total 10.368 µs / 5.1%. Current `ReLUInto` intentionally maps NaNs to zero, so its contract cannot simply be changed. | **≈2.6–4.1% at 2–5× correction improvement**; overlaps fusion and allocation rows. | Medium; medium/high NaN/zero/cancellation risk; medium confidence. API design needed. |

Pure framework dispatch, mutexes, tiny MatMul and standalone constant folding
do not currently have evidence for a larger gain. Parallelism is outside the
fixed single-thread comparison; test it as a separate configuration later.

**Measured facts:** repeated latencies and allocations, actual graph counts,
helper partition timings, dispatch/assembly, output agreement and trace ranges.
**Evidence-supported hypotheses:** residual-column acceleration is the best
next step; general memory planning is secondary here; fusion may remove much
more work in Go than in ORT.
**Unestablished:** a particular new microkernel's speedup, exact cache or
instruction-throughput limit, selected ORT microkernel, and future Go threading
benefit. No production result is claimed for these hypotheses.

## Validation and reproduction

All eight full comparisons pass numerical, shape, name, dtype, top-1,
original/adapted-model and upstream expected-output checks. Maximum Go/ORT
absolute error stays **1.9073486328125e-5**. All Go output bits match the previous
v0.10.1 comparison and final verification.
Default and `purego` full suites, race suite, `go vet`, and the three Python
comparison-policy tests pass. Both new diagnostic benchmarks ran successfully;
their correctness assertions execute before/after measurement. See
[validation summary](data/validation-summary.json) and test logs in `data`.

Run measurements serially, without tests, compilation or another benchmark in
parallel. Use the pinned Python environment described in the parent README.
The original environment used `/private/tmp/neuralnetwork-onnx-reference/bin/python`
and `GOCACHE=/private/tmp/onnx-runtime-gocache`.

```sh
export GOWORK=off CGO_ENABLED=0 GOMAXPROCS=1 GOARM64=v8.0
mkdir -p /tmp/onnx-pass5-repeat
go test -c -o /tmp/onnx-pass5-repeat/onnx.test .
/tmp/onnx-pass5-repeat/onnx.test -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=1s -test.count=9 -test.benchmem
/tmp/onnx-pass5-repeat/onnx.test -test.run '^$' -test.bench '^BenchmarkMNISTNodes$' -test.benchtime=300ms -test.count=5 -test.benchmem
/tmp/onnx-pass5-repeat/onnx.test -test.run '^$' -test.bench '^BenchmarkMNISTPipeline$' -test.benchtime=500ms -test.count=7 -test.benchmem
/tmp/onnx-pass5-repeat/onnx.test -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=12s -test.cpuprofile=/tmp/onnx-pass5-repeat/cpu.pprof
/tmp/onnx-pass5-repeat/onnx.test -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=1000x -test.memprofile=/tmp/onnx-pass5-repeat/mem.pprof -test.memprofilerate=1
/tmp/onnx-pass5-repeat/onnx.test -test.run '^$' -test.bench '^BenchmarkMNIST$' -test.benchtime=1000x -test.trace=/tmp/onnx-pass5-repeat/trace.out
go tool pprof -top /tmp/onnx-pass5-repeat/cpu.pprof
go run cmd/trace -d=parsed /tmp/onnx-pass5-repeat/trace.out
go run cmd/objdump -s 'tensor.conv2DFourChannels|tensor.conv2DInteriorTile' /tmp/onnx-pass5-repeat/onnx.test
```

Run the unchanged driver seven times with distinct `--report` destinations,
`--archive .cache/compare-runtime/mnist-12.tar.gz`, and no other options.
Run the separate ORT configuration experiment with:

```sh
python tools/compare_runtime/pass5/ort_diagnostics.py /tmp/onnx-pass5-repeat/ort
```

To reproduce partition timings, copy published backend v0.10.1 to a writable
temporary directory, copy `conv_paths_test.go.txt` into its `tensor/` directory
as `pass5_diagnostic_test.go`, then build its test binary there with `GOWORK=off`.
Do not modify the module cache or use this diagnostic copy for the baseline.
Run `-test.run '^$' -test.bench '^BenchmarkPass5ConvPaths$'
-test.benchtime=500ms -test.count=7 -test.benchmem`.

Original local binaries and extended compiler logs are in
`/private/tmp/onnx-pass5`. The checked-in raw profiles can be read with `pprof`
without a binary for symbol summaries; line/assembly inspection needs a matching
binary or the retained disassembly. Compressed JSON uses gzip and retains every
raw comparison latency. No executable binaries or dependency source trees are
included in this report.
