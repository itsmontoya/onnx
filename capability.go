package onnx

import "github.com/itsmontoya/onnx/internal/capability"

// Capability describes an implemented subset or explicitly planned operator.
// Implemented records currently describe static planning, not an execution API.
type Capability = capability.Record
