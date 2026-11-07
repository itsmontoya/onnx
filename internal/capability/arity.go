package capability

// Arity counts positional slots, including any permitted empty names.
type Arity struct {
	Min           int   `json:"min"`
	Max           int   `json:"max"`
	OptionalSlots []int `json:"optional_slots"`
}
