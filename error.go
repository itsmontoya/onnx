package onnx

import "github.com/itsmontoya/onnx/internal/runtimeerror"

// Error describes a compilation failure while preserving its original cause.
type Error = runtimeerror.Error
