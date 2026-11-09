package operators

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/itsmontoya/onnx/internal/capability"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

func TestRegistryVersionSelection(t *testing.T) {
	// Actual ONNX 1.16.2 schema differences. These synthetic planning profiles
	// exercise resolution only; graph.Build accepts only the two audited production profiles.
	// Softmax-11 coerces to 2D with axis=1; Softmax-13 uses axis=-1.
	// Squeeze-11 uses an axes attribute; Squeeze-13 uses an optional input.
	records := []capability.Record{
		{Operator: "Softmax", SchemaVersion: 11, MinOpset: 11, MaxOpset: 12, Status: "planned", Attributes: map[string]capability.Attribute{"axis": {Type: "INT", Default: json.RawMessage("1")}}},
		{Operator: "Softmax", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "planned", Attributes: map[string]capability.Attribute{"axis": {Type: "INT", Default: json.RawMessage("-1")}}},
		{Operator: "Squeeze", SchemaVersion: 11, MinOpset: 11, MaxOpset: 12, Status: "planned", Inputs: capability.Arity{Min: 1, Max: 1}, Outputs: capability.Arity{Min: 1, Max: 1}, Attributes: map[string]capability.Attribute{"axes": {Type: "INTS"}}},
		{Operator: "Squeeze", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "planned", Inputs: capability.Arity{Min: 1, Max: 2, OptionalSlots: []int{1}}, Outputs: capability.Arity{Min: 1, Max: 1}},
		{Operator: "Conv", SchemaVersion: 11, MinOpset: 13, MaxOpset: 13, Status: "planned"},
	}
	registry, err := NewRegistry(records)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		operator string
		imported int64
		version  int64
		axis     string
	}{
		{"Softmax", 11, 11, "1"},
		{"Softmax", 12, 11, "1"},
		{"Softmax", 13, 13, "-1"},
		{"Squeeze", 12, 11, ""},
		{"Squeeze", 13, 13, ""},
		{"Conv", 13, 11, ""},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%s/import%d", tt.operator, tt.imported), func(t *testing.T) {
			selected, resolveErr := registry.Resolve("", tt.operator, tt.imported)
			if resolveErr != nil || selected.SchemaVersion != tt.version || string(selected.Attributes["axis"].Default) != tt.axis {
				t.Fatalf("Resolve = %+v, %v", selected, resolveErr)
			}
		})
	}

	oldSqueeze, err := registry.Resolve("", "Squeeze", 12)
	if err != nil {
		t.Fatal(err)
	}

	newSqueeze, err := registry.Resolve("", "Squeeze", 13)
	if err != nil {
		t.Fatal(err)
	}

	attributeForm := graph.Node{Operator: "Squeeze", Inputs: []string{"x"}, Outputs: []string{"y"}, Attributes: []graph.Attribute{{Name: "axes", Kind: graph.IntsAttribute, Ints: []int64{0}}}}
	inputForm := graph.Node{Operator: "Squeeze", Inputs: []string{"x", "axes"}, Outputs: []string{"y"}}
	if err = validateNode(attributeForm, oldSqueeze); err != nil {
		t.Fatal(err)
	}

	if err = validateNode(attributeForm, newSqueeze); !errors.Is(err, graph.ErrUnsupported) {
		t.Fatalf("new schema accepted old attribute: %v", err)
	}

	if err = validateNode(inputForm, newSqueeze); err != nil {
		t.Fatal(err)
	}

	if err = validateNode(inputForm, oldSqueeze); !errors.Is(err, graph.ErrUnsupported) {
		t.Fatalf("old schema accepted axes input: %v", err)
	}

	inputForm.Inputs[1] = ""
	if err = validateNode(inputForm, newSqueeze); err != nil {
		t.Fatalf("optional input slot: %v", err)
	}
}

func TestRegistryRejectsUntestedImports(t *testing.T) {
	registry, err := NewRegistry([]capability.Record{
		{Operator: "Identity", SchemaVersion: 1, MinOpset: 1, MaxOpset: 99, Status: "planned"},
		{Operator: "Identity", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "implemented"},
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		domain   string
		operator string
		imported int64
	}{{"", "Identity", 14}, {"", "Identity", 99}, {"ai.onnx", "Identity", 13}, {"", "Unknown", 13}} {
		if _, err = registry.Resolve(tt.domain, tt.operator, tt.imported); !errors.Is(err, graph.ErrUnsupported) {
			t.Fatalf("Resolve(%q,%q,%d) = %v", tt.domain, tt.operator, tt.imported, err)
		}
	}
}

func TestRegistryOwnershipAndInvalidArguments(t *testing.T) {
	records := []capability.Record{{Operator: "Softmax", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "planned", Attributes: map[string]capability.Attribute{"axis": {Default: json.RawMessage("-1")}}}}
	registry, err := NewRegistry(records)
	if err != nil {
		t.Fatal(err)
	}

	records[0].Attributes["axis"].Default[0] = '9'
	first, err := registry.Resolve("", "Softmax", 13)
	if err != nil {
		t.Fatal(err)
	}

	first.Attributes["axis"].Default[0] = '8'
	second, err := registry.Resolve("", "Softmax", 13)
	if err != nil || string(second.Attributes["axis"].Default) != "-1" {
		t.Fatalf("registry was mutated: %+v, %v", second, err)
	}

	for _, record := range []capability.Record{
		{},
		{Operator: "Identity", SchemaVersion: 14, MinOpset: 13, MaxOpset: 13, Status: "planned"},
		{Operator: "Identity", SchemaVersion: 13, MinOpset: 14, MaxOpset: 13, Status: "planned"},
		{Operator: "Identity", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "unknown"},
		{Operator: "Unknown", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "implemented"},
	} {
		if _, err = NewRegistry([]capability.Record{record}); !errors.Is(err, runtimeerror.ErrInvalidArgument) {
			t.Fatalf("NewRegistry(%+v) = %v", record, err)
		}
	}

	if _, err = NewRegistry(append(records, records[0])); !errors.Is(err, runtimeerror.ErrInvalidArgument) {
		t.Fatalf("duplicate record = %v", err)
	}

	if _, err = (*Registry)(nil).Resolve("", "Identity", 13); !errors.Is(err, runtimeerror.ErrInvalidArgument) {
		t.Fatalf("nil registry = %v", err)
	}

	if _, err = registry.Resolve("", "Softmax", 0); !errors.Is(err, runtimeerror.ErrInvalidArgument) {
		t.Fatalf("invalid import = %v", err)
	}
}

func TestRegistryDisjointProfiles(t *testing.T) {
	records := []capability.Record{
		{Operator: "Softmax", SchemaVersion: 13, MinOpset: 13, MaxOpset: 13, Status: "implemented"},
		{Operator: "Softmax", SchemaVersion: 13, MinOpset: 17, MaxOpset: 17, Status: "implemented"},
		{Operator: "Identity", SchemaVersion: 13, MinOpset: 13, MaxOpset: 99, Status: "implemented"},
		{Operator: "Identity", SchemaVersion: 16, MinOpset: 17, MaxOpset: 17, Status: "implemented"},
	}
	registry, err := NewRegistry(records)
	if err != nil {
		t.Fatal(err)
	}

	for _, operator := range []string{"Softmax", "Identity"} {
		for _, imported := range []int64{13, 17} {
			selected, resolveErr := registry.Resolve("", operator, imported)
			if resolveErr != nil || selected.MinOpset > imported || selected.MaxOpset < imported {
				t.Fatalf("%s import %d = %+v, %v", operator, imported, selected, resolveErr)
			}
		}
	}
	for _, imported := range []int64{14, 15, 16, 18, 99} {
		if _, err = registry.Resolve("", "Softmax", imported); !errors.Is(err, graph.ErrUnsupported) {
			t.Fatalf("disjoint interval gap %d = %v", imported, err)
		}
	}
	for _, imported := range []int64{16, 18, 99} {
		if _, err = registry.Resolve("", "Identity", imported); !errors.Is(err, graph.ErrUnsupported) {
			t.Fatalf("new schema fell back at %d = %v", imported, err)
		}
	}
	overlap := records[0]
	overlap.MaxOpset = 17
	if _, err = NewRegistry([]capability.Record{overlap, records[1]}); !errors.Is(err, runtimeerror.ErrInvalidArgument) {
		t.Fatalf("overlapping same-schema intervals = %v", err)
	}
}
