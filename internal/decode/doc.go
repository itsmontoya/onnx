// Package decode bounds protobuf wire decoding and converts validated inline
// TensorProto payloads into owned backend tensors before graph validation.
// It preserves unknown fields and never takes ownership of supplied readers.
package decode
