# Implemented capabilities

This table is generated from [capabilities.json](capabilities.json), using the
same records as `Capabilities()` and the compiler registry. CI checks it with
`go run ./tools/capabilitiesgen`; update it with
`go run ./tools/capabilitiesgen -write`.

Each row uses the empty domain and its exact tested import. Imports 14–16 are
unsupported. Optional input slots are zero-based. Only listed attributes are
accepted, with the recorded types and defaults. Required inputs and outputs
cannot have empty names. Each operator has exactly one nonempty output.
Gemm C may be omitted or have an empty slot; a supplied C must be compatible
even when beta is zero.

<!-- Implemented capabilities start -->

| Operator (schema) | Tested imports | Types | Inputs → outputs | Attributes | Restrictions |
| --- | --- | --- | --- | --- | --- |
| Identity (13) | 13–13 | float32, int64 | 1 → 1 | None | static planning and serial execution; session lifecycle implemented by RT-08; dense float32/int64; arbitrary rank including scalar/empty; output shape/type preserved; requested outputs own independent copies, including Identity aliases |
| Constant (13) | 13–13 | float32, int64 | 0 → 1 | `value:TENSOR`, `value_float:FLOAT`, `value_floats:FLOATS`, `value_int:INT`, `value_ints:INTS` | exactly one dense float32/int64 tensor, float/int scalar or float/int vector attribute; exact int64 storage; arbitrary tensor rank including scalar/empty; reject sparse/string/other forms; scalar/list materialization shares aggregate load and initializer budgets |
| Add (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Sub (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Mul (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Div (13) | 13–13 | float32 | 2 → 1 | None | float32 multidirectional broadcasting; IEEE division by zero |
| MatMul (13) | 13–13 | float32 | 2 → 1 | None | float32 rank >= 1; vector promotion and batch broadcasting; empty contractions |
| Gemm (13) | 13–13 | float32 | 2–3; optional slots [2] → 1 | `alpha:FLOAT=1.0`, `beta:FLOAT=1.0`, `transA:INT=0`, `transB:INT=0` | float32 rank-two A/B; optional scalar/vector/row/column/full C with unidirectional broadcast; transA/transB restricted to 0/1; alpha/beta default 1, including nonfinite coefficients |
| Relu (13) | 13–13 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty; propagate NaN; negative zero becomes positive zero |
| Sigmoid (13) | 13–13 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Tanh (13) | 13–13 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Softmax (13) | 13–13 | float32 | 1 → 1 | `axis:INT=-1` | float32 rank >= 1; axis default -1; empty tensor accepted after axis validation as explicit reference extension; nonfinite slices preserve NaN behavior |
| Flatten (13) | 13–13 | float32, int64 | 1 → 1 | `axis:INT=1` | type preserving; split axis in [-rank,rank], default 1; explicit concrete products for empty prefixes; documented evaluator rejection difference |
| Reshape (13) | 13–13 | float32, int64 | 2 → 1 | None | int64 rank-one shape input; schema-13 zero/-1 policy; reject allowzero and ambiguous zero-product inference |
| Transpose (13) | 13–13 | float32, int64 | 1 → 1 | `perm:INTS` | type preserving; perm defaults to reverse; complete unique nonnegative permutation |
| Squeeze (13) | 13–13 | float32, int64 | 1–2; optional slots [1] → 1 | None | optional int64 rank-one axes input; omitted axes remove all unit dimensions; explicit axes must be unique and name unit dimensions; explicit empty axes are identity |
| Unsqueeze (13) | 13–13 | float32, int64 | 2 → 1 | None | required int64 rank-one axes input; unique axes normalized in output rank |
| Shape (13) | 13–13 | float32, int64 | 1 → 1 | None | int64 vector output from concrete runtime shape; no start/end attributes |
| Gather (13) | 13–13 | float32, int64 | 2 → 1 | `axis:INT=0` | float32/int64 rank >= 1 data; arbitrary-rank int64 indices; negative indices supported; exact prefix/indices/suffix shape including empties |
| Concat (13) | 13–13 | float32, int64 | 1–2147483647 → 1 | `axis:INT` | float32/int64 inputs of one type with equal rank >= 1 and equal non-axis dimensions; checked axis sum; required axis; nonempty variadic input list |
| Conv (11) | 13–13 | float32 | 2–3; optional slots [2] → 1 | `auto_pad:STRING=NOTSET`, `dilations:INTS`, `group:INT=1`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; ONNX [M,C/group,kH,kW] weights; positive channels/filters/kernels/strides/dilations; group divides input/output channels; optional [M] bias; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; kernel_shape must match weights when present; dynamic weights/bias; nonpositive floor spatial extents clamp to zero |
| MaxPool (12) | 13–13 | float32 | 1 → 1–2; optional slots [1] | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `dilations:INTS`, `kernel_shape:INTS`, `pads:INTS`, `storage_order:INT=0`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides/dilations; nonnegative pads; values output only; second empty slot allowed, nonempty indices unsupported; storage_order must be 0; ceil_mode must be 0/1; documented empty/nonfinite differences; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| AveragePool (11) | 13–13 | float32 | 1 → 1 | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `count_include_pad:INT=0`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides; nonnegative pads; no dilation attribute; ceil_mode/count_include_pad must be 0/1; full-kernel divisor when including pad; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| GlobalAveragePool (1) | 13–13 | float32 | 1 → 1 | None | float32 NCHW rank four; output [N,C,1,1]; empty spatial reduction NaN |
| BatchNormalization (9) | 13–13 | float32 | 5 → 1 | `epsilon:FLOAT=9.999999747378752e-06`, `momentum:FLOAT=0.8999999761581421` | float32 rank-two or NCHW rank-four input and [C] parameters; exactly one inference output slot; reject all additional output slots and training_mode; finite nonnegative epsilon; finite momentum in [0,1] accepted but no statistics update; independent stored-statistics reference; documented evaluator/ORT differences |
| Identity (16) | 17–17 | float32, int64 | 1 → 1 | None | static planning and serial execution; session lifecycle implemented by RT-08; dense float32/int64; arbitrary rank including scalar/empty; output shape/type preserved; requested outputs own independent copies, including Identity aliases |
| Constant (13) | 17–17 | float32, int64 | 0 → 1 | `value:TENSOR`, `value_float:FLOAT`, `value_floats:FLOATS`, `value_int:INT`, `value_ints:INTS` | exactly one dense float32/int64 tensor, float/int scalar or float/int vector attribute; exact int64 storage; arbitrary tensor rank including scalar/empty; reject sparse/string/other forms; scalar/list materialization shares aggregate load and initializer budgets |
| Add (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Sub (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Mul (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting, including scalars/empty dimensions |
| Div (14) | 17–17 | float32 | 2 → 1 | None | float32 multidirectional broadcasting; IEEE division by zero |
| MatMul (13) | 17–17 | float32 | 2 → 1 | None | float32 rank >= 1; vector promotion and batch broadcasting; empty contractions |
| Gemm (13) | 17–17 | float32 | 2–3; optional slots [2] → 1 | `alpha:FLOAT=1.0`, `beta:FLOAT=1.0`, `transA:INT=0`, `transB:INT=0` | float32 rank-two A/B; optional scalar/vector/row/column/full C with unidirectional broadcast; transA/transB restricted to 0/1; alpha/beta default 1, including nonfinite coefficients |
| Relu (14) | 17–17 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty; propagate NaN; negative zero becomes positive zero |
| Sigmoid (13) | 17–17 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Tanh (13) | 17–17 | float32 | 1 → 1 | None | float32 arbitrary rank including scalar/empty |
| Softmax (13) | 17–17 | float32 | 1 → 1 | `axis:INT=-1` | float32 rank >= 1; axis default -1; empty tensor accepted after axis validation as explicit reference extension; nonfinite slices preserve NaN behavior |
| Flatten (13) | 17–17 | float32, int64 | 1 → 1 | `axis:INT=1` | type preserving; split axis in [-rank,rank], default 1; explicit concrete products for empty prefixes; documented evaluator rejection difference |
| Reshape (14) | 17–17 | float32, int64 | 2 → 1 | `allowzero:INT=0` | int64 rank-one shape input; at most one inferred -1; equal element counts; allowzero=0 copies input dimensions; allowzero=1 uses literal zeros and forbids any -1 with zero; reject ambiguous zero-product inference |
| Transpose (13) | 17–17 | float32, int64 | 1 → 1 | `perm:INTS` | type preserving; perm defaults to reverse; complete unique nonnegative permutation |
| Squeeze (13) | 17–17 | float32, int64 | 1–2; optional slots [1] → 1 | None | optional int64 rank-one axes input; omitted axes remove all unit dimensions; explicit axes must be unique and name unit dimensions; explicit empty axes are identity |
| Unsqueeze (13) | 17–17 | float32, int64 | 2 → 1 | None | required int64 rank-one axes input; unique axes normalized in output rank |
| Shape (15) | 17–17 | float32, int64 | 1 → 1 | `end:INT`, `start:INT=0` | int64 vector output from concrete runtime shape; schema-15 start/end clamp to input rank after negative indexing; start >= end yields an empty vector |
| Gather (13) | 17–17 | float32, int64 | 2 → 1 | `axis:INT=0` | float32/int64 rank >= 1 data; arbitrary-rank int64 indices; negative indices supported; exact prefix/indices/suffix shape including empties |
| Concat (13) | 17–17 | float32, int64 | 1–2147483647 → 1 | `axis:INT` | float32/int64 inputs of one type with equal rank >= 1 and equal non-axis dimensions; checked axis sum; required axis; nonempty variadic input list |
| Conv (11) | 17–17 | float32 | 2–3; optional slots [2] → 1 | `auto_pad:STRING=NOTSET`, `dilations:INTS`, `group:INT=1`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; ONNX [M,C/group,kH,kW] weights; positive channels/filters/kernels/strides/dilations; group divides input/output channels; optional [M] bias; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; kernel_shape must match weights when present; dynamic weights/bias; nonpositive floor spatial extents clamp to zero |
| MaxPool (12) | 17–17 | float32 | 1 → 1–2; optional slots [1] | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `dilations:INTS`, `kernel_shape:INTS`, `pads:INTS`, `storage_order:INT=0`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides/dilations; nonnegative pads; values output only; second empty slot allowed, nonempty indices unsupported; storage_order must be 0; ceil_mode must be 0/1; documented empty/nonfinite differences; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| AveragePool (11) | 17–17 | float32 | 1 → 1 | `auto_pad:STRING=NOTSET`, `ceil_mode:INT=0`, `count_include_pad:INT=0`, `kernel_shape:INTS`, `pads:INTS`, `strides:INTS` | float32 NCHW rank four; positive 2D kernel/strides; nonnegative pads; no dilation attribute; ceil_mode/count_include_pad must be 0/1; full-kernel divisor when including pad; auto_pad NOTSET/VALID/SAME_UPPER/SAME_LOWER; reject conflicting pads; ceil mode trims starts in right padded region |
| GlobalAveragePool (1) | 17–17 | float32 | 1 → 1 | None | float32 NCHW rank four; output [N,C,1,1]; empty spatial reduction NaN |
| BatchNormalization (15) | 17–17 | float32 | 5 → 1 | `epsilon:FLOAT=9.999999747378752e-06`, `momentum:FLOAT=0.8999999761581421`, `training_mode:INT=0` | float32 rank-two or NCHW rank-four input and [C] parameters; exactly one inference output slot; reject all additional output slots and nonzero training_mode; finite nonnegative epsilon; finite momentum in [0,1] accepted but no statistics update; independent stored-statistics reference; documented evaluator/ORT differences; schema-15 inference only: training_mode=0, all five inputs float32, exactly one output slot |
| LayerNormalization (17) | 17–17 | float32 | 2–3; optional slots [2] → 1–3; optional slots [1 2] | `axis:INT=-1`, `epsilon:FLOAT=9.999999747378752e-06`, `stash_type:INT=1` | IR 8/import 17 only; schema 17; dense float32 inputs and all outputs; stash_type=1 only; rank >=1; axis defaults to -1 and selects the trailing normalized dimensions; nonempty normalized dimensions required; epsilon defaults to float32 1e-5; finite nonnegative values only; Scale and optional B must be unidirectionally broadcastable to X; Y preserves X shape; optional Mean/InvStdDev slots preserve position; reduced dimensions retained as 1; NN-12 ReduceMean/Sqrt/arithmetic composition; synchronous cancellation between kernels; conservative workspace reservations |

<!-- Implemented capabilities end -->
