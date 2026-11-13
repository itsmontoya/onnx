package onnx

import (
	"slices"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func newRunStorage(lifetimes *valueLifetimes, budget *runBudget) (out runStorage) {
	var s runStorage
	s.lifetimes, s.budget = lifetimes, budget
	s.values = make([]*tensor.Tensor, len(lifetimes.owners))
	return s
}

// runStorage keeps one disjoint private destination for reuse within a call.
// The candidate keeps its invocation charge until borrowed or dropped. Optional
// Session retention has a separate residency charge that retirement never releases.
type runStorage struct {
	lifetimes    *valueLifetimes
	budget       *runBudget
	values       []*tensor.Tensor
	free         *tensor.Tensor
	reuses       int
	destinations *destinationStorage
	slot         int
	fresh        bool
}

func (s *runStorage) destination(dataType DataType, shape []int64, bytes, workspace int64) (out *tensor.Tensor, err error) {
	candidate := s.free
	if candidate != nil {
		s.free = nil
		if candidate.DataType() == dataType && slices.Equal(candidate.Shape(), shape) {
			if err = s.reserve(0, workspace); err != nil {
				s.budget.release(candidate.ByteSize())
				return nil, err
			}

			s.reuses++
			return candidate, nil
		}

		s.budget.release(candidate.ByteSize())
	}

	if err = s.reserve(bytes, workspace); err != nil {
		return nil, err
	}

	if out = s.destinations.get(s.slot, dataType, shape); out != nil {
		return out, nil
	}

	s.fresh = true
	return nil, nil
}

func (s *runStorage) reserve(bytes, workspace int64) (err error) {
	if err = s.budget.reserve(bytes, workspace); err == nil {
		return nil
	}

	// A free buffer must not cause a false destination or scratch limit failure.
	s.trim()
	return s.budget.reserve(bytes, workspace)
}

func (s *runStorage) after(position int) {
	for _, owner := range s.lifetimes.retire[position] {
		s.trim()
		s.free = s.values[owner]
		for _, index := range s.lifetimes.members[owner] {
			s.values[index] = nil
		}
	}
}

func (s *runStorage) trim() {
	if s.free == nil {
		return
	}

	s.budget.release(s.free.ByteSize())
	s.free = nil
}
