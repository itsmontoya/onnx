package operators

import (
	"fmt"
	"slices"
	"sort"

	"github.com/itsmontoya/onnx/internal/capability"
	"github.com/itsmontoya/onnx/internal/graph"
)

func validateNode(node graph.Node, record capability.Record) (err error) {
	if err = validateSlots("input", node.Inputs, record.Inputs); err != nil {
		return err
	}

	if err = validateSlots("output", node.Outputs, record.Outputs); err != nil {
		return err
	}

	seen := make(map[string]bool)
	for _, attribute := range node.Attributes {
		allowed, exists := record.Attributes[attribute.Name]
		if !exists {
			return fmt.Errorf("%w: attribute %q is not supported by %s-%d", graph.ErrUnsupported, attribute.Name, node.Operator, record.SchemaVersion)
		}

		if attributeType(attribute.Kind) != allowed.Type {
			return fmt.Errorf("%w: attribute %q has type %s, require %s", graph.ErrInvalidModel, attribute.Name, attributeType(attribute.Kind), allowed.Type)
		}

		seen[attribute.Name] = true
	}

	var names []string
	for name := range record.Attributes {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if record.Attributes[name].Required && !seen[name] {
			return fmt.Errorf("%w: required attribute %q missing", graph.ErrInvalidModel, name)
		}
	}
	return nil
}

func validateSlots(label string, slots []string, arity capability.Arity) (err error) {
	if len(slots) < arity.Min {
		return fmt.Errorf("%w: %s slot count %d, require [%d,%d]", graph.ErrInvalidModel, label, len(slots), arity.Min, arity.Max)
	}

	if len(slots) > arity.Max {
		return fmt.Errorf("%w: %s slot count %d, support [%d,%d]", graph.ErrUnsupported, label, len(slots), arity.Min, arity.Max)
	}

	for index, name := range slots {
		if name == "" && !slices.Contains(arity.OptionalSlots, index) {
			return fmt.Errorf("%w: required %s[%d] is empty", graph.ErrInvalidModel, label, index)
		}
	}
	return nil
}

func attributeType(kind graph.AttributeKind) (name string) {
	switch kind {
	case graph.FloatAttribute:
		return "FLOAT"
	case graph.IntAttribute:
		return "INT"
	case graph.StringAttribute:
		return "STRING"
	case graph.TensorAttribute:
		return "TENSOR"
	case graph.FloatsAttribute:
		return "FLOATS"
	case graph.IntsAttribute:
		return "INTS"
	case graph.StringsAttribute:
		return "STRINGS"
	case graph.TensorsAttribute:
		return "TENSORS"
	}
	return "UNDEFINED"
}
