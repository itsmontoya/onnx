package graph

import "slices"

// Metadata preserves model identity and harmless descriptive fields. ModelDomain
// is descriptive, independent of the strictly empty operator import/node domain.
type Metadata struct {
	Name            string
	IRVersion       int64
	Opset           int64
	Producer        string
	ProducerVersion string
	ModelDomain     string
	ModelVersion    int64
	DocString       string
	GraphDocString  string
	Properties      []Property
	GraphProperties []Property
	Annotations     []Annotation
}

func (m Metadata) clone() (out Metadata) {
	out = m
	out.Properties = slices.Clone(m.Properties)
	out.GraphProperties = slices.Clone(m.GraphProperties)
	out.Annotations = make([]Annotation, len(m.Annotations))
	for index, annotation := range m.Annotations {
		out.Annotations[index] = annotation.clone()
	}
	return out
}
