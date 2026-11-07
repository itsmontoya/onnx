// Package capability defines manifest records shared by the facade and planner.
package capability

import (
	"maps"
	"slices"
)

// Record describes one implemented subset or planned operator schema.
type Record struct {
	Domain        string               `json:"domain"`
	Operator      string               `json:"operator"`
	SchemaVersion int64                `json:"schema_version"`
	MinOpset      int64                `json:"min_opset"`
	MaxOpset      int64                `json:"max_opset"`
	Status        string               `json:"status"`
	Task          string               `json:"task"`
	DataTypes     []string             `json:"data_types"`
	Inputs        Arity                `json:"inputs"`
	Outputs       Arity                `json:"outputs"`
	Attributes    map[string]Attribute `json:"attributes"`
	Restrictions  []string             `json:"restrictions"`
}

// Clone returns an independent record, including default JSON payloads.
func (r Record) Clone() (out Record) {
	out = r
	out.DataTypes = slices.Clone(r.DataTypes)
	out.Inputs.OptionalSlots = slices.Clone(r.Inputs.OptionalSlots)
	out.Outputs.OptionalSlots = slices.Clone(r.Outputs.OptionalSlots)
	out.Restrictions = slices.Clone(r.Restrictions)
	out.Attributes = maps.Clone(r.Attributes)
	for name, attribute := range out.Attributes {
		attribute.Default = slices.Clone(attribute.Default)
		out.Attributes[name] = attribute
	}
	return out
}
