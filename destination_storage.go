package onnx

import (
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func newDestinationStorage(limit, minimumLive int64, slots int) (out destinationStorage) {
	var s destinationStorage
	s.limit, s.minimumLive = limit, minimumLive
	if limit > 0 {
		s.values = make([]*tensor.Tensor, slots)
	}

	return s
}

// destinationStorage owns exact-shape private tensors between serialized Runs.
// Its payload stays charged in addition to the original invocation reservations.
type destinationStorage struct {
	values      []*tensor.Tensor
	bytes       int64
	limit       int64
	minimumLive int64
}

func (s *destinationStorage) get(slot int, dataType DataType, shape []int64) (out *tensor.Tensor) {
	if s == nil || slot < 0 || slot >= len(s.values) {
		return nil
	}

	candidate := s.values[slot]
	if candidate == nil || candidate.DataType() != dataType || !slices.Equal(candidate.Shape(), shape) {
		return nil
	}

	return candidate
}

func (s *destinationStorage) keep(slot int, value *tensor.Tensor, budget *runBudget) {
	if s == nil || slot < 0 || slot >= len(s.values) || s.values[slot] != nil {
		return
	}

	bytes := value.ByteSize()
	if bytes == 0 || bytes > s.limit-s.bytes {
		return
	}

	if budget.reserve(bytes, 0) != nil {
		return
	}

	s.values[slot] = value
	s.bytes += bytes
}
