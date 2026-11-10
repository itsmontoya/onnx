Optimize neuralnetwork's Conv2DInto inference path, starting from the current
repository and using v0.9.0 as the performance baseline. Read and follow this
project's AGENTS.md and STYLEGUIDE.md. Production kernel changes are authorized
for this task; keep them focused on convolution and preserve the public API.

First read the investigation in the sibling ONNX repository:

`../onnx/docs/performance/mnist-round3/README.md`

The directory includes raw benchmarks, CPU and allocation profiles as text,
a diagnostic benchmark, results.json, and neuralnetwork-experiment.patch.
Treat the patch as evidence and a starting point for review, not a finished
implementation to apply blindly.

The next bottleneck is tensor/conv2d.go. With published v0.9.0 on Apple M3,
Go 1.27.2, GOMAXPROCS=1 and CGO_ENABLED=0, Conv2DInto accounts for 79.4% of
sampled Session.Run stacks. Scalar conv2DValue alone accounts for 46.3%.
The second MNIST convolution is input [1,8,14,14], weights [16,8,5,5],
output [1,16,14,14], stride/dilation 1, groups 1, padding 2 on all sides.
Six of fourteen columns use the scalar path: four padding-boundary columns
and two interior remainders. The first convolution is input [1,1,28,28],
weights [8,1,5,5], output [1,8,28,28], with the same spatial configuration.

An isolated prototype preserves four-spatial-column processing in the interior
and computes four output channels together at boundaries/remainders, sharing
input samples and geometry while keeping four independent accumulators.
Seven rotating end-to-end measurements improved median latency from
433.08 to 318.64 microseconds, 26.4%, with unchanged 179440 B/op and
174 allocs/op. The second isolated convolution improved from 284.52 to
181.02 microseconds. All 131 end-to-end probes were bit-identical to v0.9.0.
The full backend and ONNX default suites, backend purego tensor suite and full
ONNX purego suite passed; the patch adds 400 scalar-reference cases.

Implement and tune a maintainable version of this optimization, or a better
measured alternative. Keep channel blocks within groups, preserve the spatial
fast path for leftover filters, and retain appropriate scalar/depthwise
fallbacks. Benchmark dispatch choices across narrow/wide outputs, filter
remainders, groups, depthwise, strides, dilation, padding, batches and empty
dimensions. Further tile widths or compiler bounds-check improvements are
reasonable experiments if measured; do not assume SIMD or packing is needed.

Preserve each output's input-channel/kernel-row/kernel-column accumulation
order, existing float32/FMA behavior and bias-last semantics. Skip padded
positions rather than multiplying zero by nonfinite weights. Preserve NaNs,
infinities, signed zero, source immutability, alias rejection, destination
preservation on errors, concurrency and empty-tensor behavior. Do not introduce
goroutines, cgo, hidden mutable caches, packed weights, or new scratch-memory
requirements as part of this focused change.

Validate against the retained scalar computation with explicit table-driven
cases, including groups with 4/5/7/8 filters, channel tails and adversarial
rounding. Run the full backend suite, purego tensor suite and relevant race/vet
checks, then validate the sibling ONNX default/purego suites with an isolated
workspace. Do not change ONNX production code or its committed dependency pin.

Use precompiled binaries and alternating repeated benchmarks with profiling
disabled, matching thread settings and owned-output semantics. Reprofile
separately and report median/range, B/op, allocs/op, regressions and module
identity. Rerun the standard 19-case Go-versus-ORT comparison without changing
inputs, tolerances or timing settings. Its Python driver explicitly sets
GOWORK=off: use a temporary driver copy targeting the experimental workspace,
as documented in the earlier round-two report, and record that change.
The projected 7.5x ORT gap has not been measured; do not report it as a result.

Deliver a focused, reviewable backend change with correctness tests, meaningful
kernel benchmarks, repeated end-to-end evidence and a short performance report.
If making a PR, follow this repository's exact PR template and formatting rules.
