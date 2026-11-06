package graph

// AttributeKind describes a structurally supported attribute payload.
// Operator-specific permitted names and kinds are validated later by RT-07.
type AttributeKind uint8

const (
	// FloatAttribute contains one float32.
	FloatAttribute AttributeKind = iota + 1
	// IntAttribute contains one exact int64.
	IntAttribute
	// StringAttribute contains a byte string.
	StringAttribute
	// TensorAttribute contains one immutable dense constant.
	TensorAttribute
	// FloatsAttribute contains float32 values.
	FloatsAttribute
	// IntsAttribute contains exact int64 values.
	IntsAttribute
	// StringsAttribute contains byte strings.
	StringsAttribute
	// TensorsAttribute contains immutable dense constants.
	TensorsAttribute
)
