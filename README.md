# onnx

A Go-native ONNX inference runtime. Load an `.onnx` model, supply named tensors,
and read named outputs on the CPU, without Python, cgo, or native runtime libraries.
The numerical backend is
[`neuralnetwork` v0.10.10](https://github.com/itsmontoya/neuralnetwork/tree/v0.10.10).

The latest recorded MNIST comparison measures **50.917 µs for Go versus
44.667 µs for ONNX Runtime (ORT)** on Apple M3: **1.14× ORT latency**.
ORT graph optimizations are disabled in this comparison.
See [performance and progression](#performance-vs-onnx-runtime) for the workload,
settings, and history.

## Install

Requires **Go 1.27.2** or later, as specified in [go.mod](go.mod).

```sh
go get github.com/itsmontoya/onnx
```

Build with `CGO_ENABLED=0` to keep deployment free of cgo dependencies.
The optional `purego` build tag selects the portable CPU backend instead of
architecture-specific acceleration.

## Try it

From a checkout of this repository:

```sh
go mod download
CGO_ENABLED=0 go run ./examples/inference -graph mlp -batch 1
```

This loads a checked-in MLP, creates a session, and prints its input metadata and
output probabilities. Models and reference values for the examples are included;
after the initial module download, these examples run offline.

## Use from Go

This complete example uses the same MLP and one input row, `[1, 2]`. Run it with
the repository root as the working directory so the fixture path resolves.
For your own model, replace the path, input name, shape, and values to match
`model.Metadata()`.

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/itsmontoya/onnx"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		ctx     = context.Background()
		model   *onnx.Model
		session *onnx.Session
		input   *onnx.Tensor
		outputs []onnx.NamedTensor
		values  []float32
	)

	if model, err = onnx.LoadFile(ctx, "testdata/inference/models/mlp.onnx", onnx.Options{}); err != nil {
		return fmt.Errorf("load model: %w", err)
	}

	defer func() { err = errors.Join(err, model.Close()) }()
	if session, err = model.NewSession(); err != nil {
		return fmt.Errorf("create session: %w", err)
	}

	defer func() { err = errors.Join(err, session.Close()) }()
	if input, err = onnx.NewFloat32([]int64{1, 2}, []float32{1, 2}); err != nil {
		return fmt.Errorf("create input: %w", err)
	}

	if outputs, err = session.Run(ctx, map[string]*onnx.Tensor{"features": input}, onnx.RunOptions{}); err != nil {
		return fmt.Errorf("run inference: %w", err)
	}

	if values, err = outputs[0].Tensor.Float32Values(); err != nil {
		return fmt.Errorf("read probabilities: %w", err)
	}

	fmt.Printf("%s %v = %.6f\n", outputs[0].Name, outputs[0].Tensor.Shape(), values)
	return nil
}
```

Output:

```text
probabilities [1 2] = [0.968477 0.031523]
```

`Load` accepts an `io.Reader`; `LoadBytes` accepts model bytes. `NewFloat32` and
`NewInt64` copy row-major values and shapes. A tensor's value count must match
its concrete shape. Typed output accessors also return copies.

Use `model.Metadata().Inputs` and `.Outputs` to inspect names, types, dimensions,
and defaulted inputs. Input names are the keys in the map passed to `Run`.
An omitted defaulted input uses its initializer; supplying it overrides that
default within the declared type and shape.

`RunOptions{}` returns all declared outputs in declaration order. To select
outputs in a particular order, pass their names:

```go
outputs, err = session.Run(ctx, inputs, onnx.RunOptions{
	Outputs: []string{"probabilities"},
})
```

Selecting fewer outputs still executes the whole graph. Results own independent
storage and remain valid after later runs or closing the session and model.

### Reuse models and sessions

Load a model once and reuse its session for repeated inference. Eligible static
graphs reuse bounded private destinations after the first call; dynamic geometry
uses ordinary allocation. Each session permits one active run. Use separate
sessions from the same model for concurrent requests; overlapping `Run` or `Close`
on one session returns `ErrBusy` immediately.

Close sessions when finished to release their retained storage and session slots.
`Model.Close` prevents new sessions; existing sessions remain usable until closed.
Failed or cancelled runs publish no partial outputs and leave the session reusable.
Pass a context to loading and inference for cancellation. Most synchronous kernels
complete before cancellation is observed; ReLU also checks between chunks.

## More examples

Run these commands from the repository root with `CGO_ENABLED=0`:

| Example | Command | Demonstrates |
| --- | --- | --- |
| Batched MLP | `go run ./examples/inference -graph mlp -batch 3` | Symbolic leading batch dimension |
| Branching graph | `go run ./examples/inference -graph branch -batch 3` | Multiple named inputs and outputs |
| Shape subgraph | `go run ./examples/inference -graph shape -batch 3` | Runtime shape operations and int64 outputs |
| CNN | `go run ./examples/inference -graph cnn -batch 3` | NCHW convolution and pooling |
| PyTorch export | `go run ./examples/inference -graph exported -batch 3` | Unmodified compatible IR-8/opset-17 MLP export |
| External weights | `go run ./examples/external -batch 3` | Explicit external-data resolver |
| Layer normalization | `go run ./examples/normalization` | Upstream opset-17 normalization model |

The source is in [examples/inference](examples/inference/main.go),
[examples/external](examples/external/main.go), and
[examples/normalization](examples/normalization/main.go).

External weights require `Options.ExternalData`, including with `LoadFile`.
Use `NewDirectoryResolver` for weights beneath an explicitly chosen directory,
or implement `ExternalDataResolver` for your storage. The resolver remains
caller-owned; close a directory resolver after loading. `LoadFile` never infers
permission to read neighboring weight files.

## Compatible models

The supported profile is **IR 8**, exactly one empty standard-domain import at
**opset 13 or 17**, dense **float32/int64** tensors, and serial CPU execution.
The [capability table](CAPABILITIES.md) lists 25 operators at opset 13 and 26 at
opset 17, with exact dtype, rank, attribute, and optional-slot restrictions.

Supported graphs include MLPs, branching, multiple named inputs/outputs,
symbolic leading batches, scalar/empty tensors, runtime shape subgraphs,
NCHW 2D CNNs, and float32 LayerNormalization at opset 17. Check the capability
table before exporting or choosing a model; support for a backend kernel alone
does not establish model compatibility. Arbitrary ONNX models are not supported.

`Options{}` selects finite resource limits. Configure `Options.Limits` when you
need explicit load, tensor, live-memory, workspace, or session budgets. These
are logical runtime budgets rather than process RSS limits. See the archived
[runtime contract](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/runtime-contract.md)
for accounting and numerical policies, including pooling and normalization
reference differences.

## Performance vs ONNX Runtime

The latest [Pass 20 measurements](PERFORMANCE_PASS20.md) compare the same pinned
pretrained MNIST CNN and named inputs on **Apple M3, macOS arm64, Go 1.27.2**,
using `neuralnetwork` v0.10.10 and **ORT 1.19.2 CPUExecutionProvider**.

| Runtime | Median inference latency | Range of seven report medians |
| --- | ---: | ---: |
| Go `onnx` | **50.917 µs** | 50.750–51.438 µs |
| ONNX Runtime | **44.667 µs** | 41.708–45.042 µs |

Go takes **6.250 µs more per inference**, about **14.0% higher latency**, under
these settings. Go uses `CGO_ENABLED=0`, `GOMAXPROCS=1`, and `GOARM64=v8.0`,
without the `purego` tag. ORT uses sequential execution, one intra-op and one
inter-op thread, and **disabled graph optimizations**. This compares the fixed
graph through Go and Python APIs; it does not measure ORT's fully optimized
deployment configuration or general model performance.

Each report includes 19 inputs: the upstream vector, zeros, ones, and sixteen
seeded random inputs of shape `[1,1,28,28]`. Each case has ten warmups and 100
timed calls per runtime, including owned outputs. Building, loading, input
preparation, and serialization are outside timing. The table uses the median of
seven per-report medians. All numerical, top-1, original/converted-model, and
upstream reference checks pass; maximum Go/ORT absolute error is
`1.9073486328125e-5` with `atol=1e-4`, `rtol=1e-5`.

A separate ordinary Go benchmark records **1,265 B/op and 24 allocs/op** after
warmup. It uses a different deterministic input and timing scope; its latency
is not substituted into the ORT comparison. Static MNIST sessions retain
**83,664 bytes of logical destination payload** after first use, plus metadata.
That retention reduces recurring allocations and increases cold-call cost;
[Pass 18](PERFORMANCE_PASS18.md) records the memory and lifecycle tradeoff.

### Performance progression

Recorded Go latency has fallen from **1,187.81 µs to 50.917 µs**, about **23.3×**
across these historical observations. The table shows selected milestones on
the same MNIST workload and fixed comparison policy. Early entries are single
comparison runs; entries from v0.10.5 onward summarize repeated comparisons.
Each row retains its own contemporaneous ORT reference. Hardware state, variance,
and ORT movement affect the ratios across these separate experiments.
Go / ORT divides the two latency medians; lower is better. Backend version labels
below refer to `neuralnetwork`.

| Milestone / evidence | Go median (µs) | ORT median (µs) | Go / ORT |
| --- | ---: | ---: | ---: |
| [Initial CPU baseline][early-comparison] | 1,187.810 | 41.790 | 28.42× |
| [v0.8.0 integration][early-comparison] | 689.580 | 41.540 | 16.60× |
| [v0.9.0 integration][early-comparison] | 452.900 | 44.620 | 10.15× |
| [v0.10.0 integration][early-comparison] | 324.210 | 44.210 | 7.33× |
| [v0.10.2 integration][early-comparison] | 161.810 | 42.960 | 3.77× |
| [v0.10.5: convolution tiles][v0105-results] | 98.855 | 44.375 | 2.23× |
| [v0.10.7: preserving ReLU][v0107-results] | 68.042 | 44.750 | 1.52× |
| [Pass 11: reuse decoded constants](PERFORMANCE_PASS11.md) | 65.813 | 44.750 | 1.47× |
| [Pass 12: prepare call metadata](PERFORMANCE_PASS12.md) | 62.792 | 44.833 | 1.40× |
| [Pass 13: retain constant Reshape](PERFORMANCE_PASS13.md) | 61.750 | 44.167 | 1.40× |
| [Pass 14: v0.10.8 ARM64 ReLU](PERFORMANCE_PASS14.md) | 55.146 | 42.417 | 1.30× |
| [Pass 15: v0.10.9 broadcast metadata](PERFORMANCE_PASS15_RELEASE.md) | 54.750 | 42.375 | 1.29× |
| [Pass 16: v0.10.10 Add adapter](PERFORMANCE_PASS16_RELEASE.md) | 54.625 | 42.084 | 1.30× |
| [Pass 18: retain static destinations](PERFORMANCE_PASS18.md) | 51.333 | 44.209 | 1.16× |
| [Pass 19: reuse node arguments](PERFORMANCE_PASS19.md) | 50.917 | 44.375 | 1.15× |
| [Pass 20: reuse actual dtype](PERFORMANCE_PASS20.md) | **50.917** | **44.667** | **1.14×** |

[early-comparison]: https://github.com/itsmontoya/onnx/blob/c2d5123/tools/compare_runtime/README.md#timing-and-observed-result
[v0105-results]: https://github.com/itsmontoya/onnx/blob/58ee8a1/README.md
[v0107-results]: https://github.com/itsmontoya/onnx/blob/3441c84/README.md

[Pass 17](PERFORMANCE_PASS17.md) was an investigation without a production change.
The reports preserve before/after measurements, allocation counts, variance,
correctness checks, and rejected experiments. Small timing changes in Passes 15,
16, 19, and 20 should be read with their recorded variance; allocation reductions
and changes in ORT's reference timing are reported separately.

### Run benchmarks

Benchmark the checked-in inference fixtures without Python or downloads:

```sh
CGO_ENABLED=0 GOMAXPROCS=1 go test . -run '^$' -bench '^BenchmarkInference$' -benchmem
```

For the recorded MNIST workload, the comparison driver and Go runner are
[archived at this revision](https://github.com/itsmontoya/onnx/tree/c2d5123/tools/compare_runtime).
The top-level `tools/compare_runtime.py`, runner, and requirements file are no
longer tracked. The archived [setup and usage guide](https://github.com/itsmontoya/onnx/blob/c2d5123/tools/compare_runtime/README.md)
describes fixture preparation and ORT configuration; use those tool files with
the current runtime to compare current code. The
[Pass 20 evidence](tools/compare_runtime/pass20/README.md) retains raw samples,
fingerprints, summaries, and replay scripts, which also require the archived driver.

Once `.cache/compare-runtime/mnist-ir8-opset13.onnx` is prepared, run the separate
ordinary Go benchmark:

```sh
CGO_ENABLED=0 GOMAXPROCS=1 go test . -run '^$' -bench '^BenchmarkMNIST$' -benchmem
```

This benchmark skips when the optional model is absent. Its results alone do not
reproduce the 19-case Go/ORT comparison above.

## Development and project status

Run the offline test suites after downloading Go modules:

```sh
CGO_ENABLED=0 go test ./...
CGO_ENABLED=0 go test -tags=purego ./...
```

Read [AGENTS.md](AGENTS.md), [STYLEGUIDE.md](STYLEGUIDE.md), and the
[PR template](.github/PULL_REQUEST_TEMPLATE.md) before contributing.
[ROADMAP.md](ROADMAP.md), [TODO.md](TODO.md), and [NEEDS.md](NEEDS.md) record
planned work and backend handoffs. Detailed older guides remain archived:
[getting started](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/getting-started.md),
[development](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/development.md), and
[release checklist and fixture provenance](https://github.com/itsmontoya/onnx/blob/a5ea458f0ec147f7878d838730751a841549fb46/docs/release-checklist.md).

**There is no project LICENSE.** Runtime source and original fixtures still need
a maintainer licensing decision before public release. Existing
[third-party notices](third_party/README.md) cover their identified upstream
artifacts and dependencies.
