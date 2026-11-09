// Package onnx loads and evaluates a bounded Go-native ONNX inference subset.
//
// Load, LoadBytes and LoadFile accept IR 8, standard-domain opset 13 models.
// M1 arithmetic/activations, M2 shape transforms and M3 CNNs execute serially.
// Capabilities lists exact restrictions and explicitly planned operators.
// Models share immutable plans across sessions; each session permits one Run
// at a time and returns owned outputs.
// Tensor constructors and accessors copy, and no mutable backend is exposed.
//
// Options applies finite logical load/run limits. External float32/int64 tensors
// require an explicit ExternalDataResolver, including when using LoadFile.
// DirectoryResolver confines regular-file ranges on supported Go platforms.
// Cancellation is checked around synchronous backend/protobuf calls,
// between nodes and before publication.
// Ordinary builds and tests require no cgo, Python or native inference runtime.
package onnx
