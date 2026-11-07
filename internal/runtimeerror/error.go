// Package runtimeerror shares diagnostics without importing the public facade.
package runtimeerror

import "fmt"

// Error retains a classification and underlying cause with deterministic context.
// Shapes uses nil for unknown shapes and a nonnil empty slice for scalar shapes.
type Error struct {
	Kind      error
	NodeIndex int
	NodeName  string
	Domain    string
	Operator  string
	Opset     int64
	Value     string
	Shapes    [][]int64
	Cause     error
}

// Error formats compilation context without iterating over unordered maps.
func (e *Error) Error() (message string) {
	if e == nil {
		return "onnx: <nil error>"
	}

	message = "onnx compile"
	if e.NodeIndex >= 0 {
		message += fmt.Sprintf(" node[%d] name %q domain %q operator %q opset %d", e.NodeIndex, e.NodeName, e.Domain, e.Operator, e.Opset)
	}

	if e.Value != "" {
		message += fmt.Sprintf(" value %q", e.Value)
	}

	if len(e.Shapes) != 0 {
		message += " input shapes ["
		for index, shape := range e.Shapes {
			if index != 0 {
				message += ", "
			}

			if shape == nil {
				message += "unknown"
			} else {
				message += fmt.Sprint(shape)
			}
		}
		message += "]"
	}

	if e.Cause != nil {
		return message + ": " + e.Cause.Error()
	}

	return fmt.Sprintf("%s: %v", message, e.Kind)
}

// Unwrap preserves the original cause for errors.Is and errors.As.
func (e *Error) Unwrap() (cause error) {
	if e == nil {
		return nil
	}

	return e.Cause
}

// Is matches the classification independently from the cause chain.
func (e *Error) Is(target error) (matches bool) {
	return e != nil && e.Kind != nil && target == e.Kind
}
