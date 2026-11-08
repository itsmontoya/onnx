// Package decode bounds protobuf wire decoding and converts validated dense
// TensorProto payloads into owned backend tensors before graph validation.
// It preserves unknown fields and leaves supplied model readers caller-owned.
// Readers opened by an explicit external resolver are closed after each range.
package decode
