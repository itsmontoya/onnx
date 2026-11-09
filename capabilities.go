package onnx

import (
	"context"
	_ "embed"
	"encoding/json"
	"fmt"

	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/operators"
)

//go:embed capabilities.json
var capabilityJSON []byte

// Capabilities returns deep copies of the canonical manifest's operator records.
// Planned entries do not load. Implemented entries include their exact limits.
func Capabilities() (capabilities []Capability) {
	var manifest struct {
		Operators []Capability `json:"operators"`
	}

	// The checked-in embedded manifest is a build artifact, never caller input.
	if err := json.Unmarshal(capabilityJSON, &manifest); err != nil {
		panic(fmt.Sprintf("invalid embedded capability manifest: %v", err))
	}

	return manifest.Operators
}

// compileGraph is the loader's facade boundary. The registry and
// public query consume the same embedded records; no second support list exists.
func compileGraph(ctx context.Context, source *graph.Graph, limits operators.Limits) (out *operators.Plan, err error) {
	var registry *operators.Registry
	if registry, err = operators.NewRegistry(Capabilities()); err != nil {
		return nil, err
	}

	return operators.Compile(ctx, source, registry, limits)
}
