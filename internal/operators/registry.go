// Package operators resolves explicit tested schemas and compiles static plans.
package operators

import (
	"fmt"

	"github.com/itsmontoya/onnx/internal/capability"
	"github.com/itsmontoya/onnx/internal/graph"
	"github.com/itsmontoya/onnx/internal/runtimeerror"
)

// NewRegistry snapshots manifest records. An implemented entry must have a real
// planner; planned records remain visible to resolution but cannot compile.
func NewRegistry(records []capability.Record) (out *Registry, err error) {
	var r Registry
	for _, record := range records {
		if record.Operator == "" || record.SchemaVersion <= 0 || record.SchemaVersion > record.MinOpset || record.MinOpset > record.MaxOpset {
			return nil, fmt.Errorf("%w: invalid capability version/operator", runtimeerror.ErrInvalidArgument)
		}

		if record.Status != "implemented" && record.Status != "planned" {
			return nil, fmt.Errorf("%w: capability status %q", runtimeerror.ErrInvalidArgument, record.Status)
		}

		if record.Status == "implemented" && !hasPlanner(record) {
			return nil, fmt.Errorf("%w: implemented %s-%d has no planner", runtimeerror.ErrInvalidArgument, record.Operator, record.SchemaVersion)
		}

		for _, previous := range r.records {
			if previous.Domain == record.Domain && previous.Operator == record.Operator && previous.SchemaVersion == record.SchemaVersion && previous.MinOpset <= record.MaxOpset && record.MinOpset <= previous.MaxOpset {
				return nil, fmt.Errorf("%w: overlapping capability %s-%d", runtimeerror.ErrInvalidArgument, record.Operator, record.SchemaVersion)
			}
		}
		r.records = append(r.records, record.Clone())
	}
	return &r, nil
}

// Registry is immutable after construction and safe for concurrent resolution.
type Registry struct {
	records []capability.Record
}

// Resolve selects the newest schema at or before the imported version, then
// requires that schema's tested interval. It never falls back to older semantics
// when a newer schema exists but lacks coverage for the requested import.
func (r *Registry) Resolve(domain, operator string, imported int64) (record capability.Record, err error) {
	if r == nil || imported <= 0 {
		return record, fmt.Errorf("%w: registry or imported opset", runtimeerror.ErrInvalidArgument)
	}

	var selected *capability.Record
	for index := range r.records {
		candidate := &r.records[index]
		if candidate.Domain != domain || candidate.Operator != operator || candidate.SchemaVersion > imported {
			continue
		}

		if selected == nil || selected.SchemaVersion < candidate.SchemaVersion {
			selected = candidate
		}
	}

	if selected == nil {
		return record, fmt.Errorf("%w: no schema for domain %q operator %q imported opset %d", graph.ErrUnsupported, domain, operator, imported)
	}

	for _, candidate := range r.records {
		if candidate.Domain == domain && candidate.Operator == operator && candidate.SchemaVersion == selected.SchemaVersion && imported >= candidate.MinOpset && imported <= candidate.MaxOpset {
			return candidate.Clone(), nil
		}
	}
	return record, fmt.Errorf("%w: %s-%d has no tested interval for requested import %d", graph.ErrUnsupported, operator, selected.SchemaVersion, imported)
}

func hasPlanner(record capability.Record) (supported bool) {
	if record.Domain != "" {
		return false
	}

	if IsCNNOperator(record.Operator) {
		versions := map[string]int64{"Conv": 11, "MaxPool": 12, "AveragePool": 11, "GlobalAveragePool": 1, "BatchNormalization": 9}
		return record.SchemaVersion == versions[record.Operator] || record.Operator == "BatchNormalization" && record.SchemaVersion == 15
	}

	versions := map[string]int64{"Identity": 16, "Add": 14, "Sub": 14, "Mul": 14, "Div": 14, "Relu": 14, "Reshape": 14, "Shape": 15}
	if record.SchemaVersion != 13 && record.SchemaVersion != versions[record.Operator] {
		return false
	}

	switch record.Operator {
	case "Identity", "Constant", "Add", "Sub", "Mul", "Div", "MatMul", "Gemm", "Relu", "Sigmoid", "Tanh", "Softmax", "Shape", "Flatten", "Transpose", "Reshape", "Squeeze", "Unsqueeze", "Gather", "Concat":
		return true
	}
	return false
}
