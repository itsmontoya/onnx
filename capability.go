package onnx

import "github.com/itsmontoya/onnx/internal/capability"

// Capability describes an implemented subset or explicitly planned operator.
// Implemented records describe the restrictions enforced by loading and execution.
type Capability = capability.Record
