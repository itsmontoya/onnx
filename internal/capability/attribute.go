package capability

import "encoding/json"

// Attribute describes an allowed attribute and an optional ONNX default.
// Default is JSON to preserve exact integers and distinguish absence from zero.
type Attribute struct {
	Type     string          `json:"type"`
	Required bool            `json:"required"`
	Default  json.RawMessage `json:"default,omitempty"`
}
