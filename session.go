package onnx

import (
	"context"
	"errors"
	"sync"
)

func newSession(plan *executionPlan) (out *Session) {
	var s Session
	s.plan, s.initialized = plan, true
	s.destinations = newDestinationStorage(plan.destinationLimit, plan.destinationMinimum, len(plan.steps))
	if plan.metal != nil {
		s.metalCache = newMetalCache(plan.metal.device)
		s.gpuStats.Active = true
	}

	return &s
}

// Session evaluates one call at a time with private values, budgets and bounded
// exact-shape destinations retained when static geometry and live limits permit.
// Overlapping Run or Close returns ErrBusy immediately. Failed/cancelled runs
// leave it usable; outputs own copies that survive later runs and any Close.
// A Session must not be copied after first use. Its zero value is invalid.
type Session struct {
	mu           sync.Mutex
	destinations destinationStorage
	metalCache   *metalCache
	gpuStats     SessionGPUStats
	plan         *executionPlan
	initialized  bool
	closed       bool
}

// GPUStats returns Metal execution counters from the most recent Run and
// current cached activation buffer accounting. CPU, nil, or zero sessions
// return the zero value. Overlapping with an active Run returns ErrBusy.
func (s *Session) GPUStats() (stats SessionGPUStats, err error) {
	if s == nil {
		return stats, ErrInvalidArgument
	}

	if !s.mu.TryLock() {
		return stats, ErrBusy
	}

	defer s.mu.Unlock()
	if !s.initialized {
		return stats, ErrInvalidArgument
	}

	stats = s.gpuStats
	if s.metalCache != nil {
		stats.ActivationBuffers, stats.ActivationBytes = s.metalCache.stats()
	}

	return stats, nil
}

// Run binds all required/defaulted inputs and executes every node serially.
// Input tensors are immutable; callers must not change inputs or options during
// the call. Cancellation is checked around synchronous copies and every node,
// during runtime loops and before publication. ReLU also polls between chunks;
// other synchronous kernels run to completion without interruption.
func (s *Session) Run(ctx context.Context, inputs map[string]*Tensor, options RunOptions) (outputs []NamedTensor, err error) {
	if s == nil {
		return nil, ErrInvalidArgument
	}

	if !s.mu.TryLock() {
		return nil, ErrBusy
	}

	defer s.mu.Unlock()
	if !s.initialized {
		return nil, ErrInvalidArgument
	}

	if s.closed {
		return nil, ErrClosed
	}

	if ctx == nil {
		return nil, ErrInvalidArgument
	}

	var run *evaluation
	if run, err = newEvaluation(ctx, s.plan); err != nil {
		return nil, operationError("run", err, "")
	}

	if s.plan.metal != nil {
		run.metal = newMetalEvaluation(ctx, s.plan.metal, s.metalCache, len(s.plan.values))
	}

	if s.plan.limits.MaxLiveBytes < s.destinations.minimumLive {
		s.destinations = destinationStorage{}
	}

	run.storage.destinations = &s.destinations
	// Admission reserves room for residency in addition to every legacy Run
	// reservation, including workspace, invalid bindings and output clones.
	run.budget.live += s.destinations.bytes
	if outputs, err = run.run(inputs, options); err != nil {
		if run.metal != nil {
			_ = run.metal.close()
			_ = s.metalCache.close()
			s.gpuStats = run.metal.stats
		}

		return nil, operationError("run", err, "")
	}

	if run.metal != nil {
		if closeErr := run.metal.close(); closeErr != nil {
			_ = s.metalCache.close()
			return nil, operationError("run", closeErr, "")
		}

		s.gpuStats = run.metal.stats
		s.gpuStats.ActivationBuffers, s.gpuStats.ActivationBytes = s.metalCache.stats()
	}

	return outputs, nil
}

// Close releases private destinations, the session slot and plan reference. It is
// idempotent after success, and returns ErrBusy without waiting for or cancelling
// an active Run.
func (s *Session) Close() (err error) {
	if s == nil {
		return ErrInvalidArgument
	}

	if !s.mu.TryLock() {
		return ErrBusy
	}

	defer s.mu.Unlock()
	if !s.initialized {
		return ErrInvalidArgument
	}

	if s.closed {
		return nil
	}

	s.destinations = destinationStorage{}
	if s.metalCache != nil {
		err = errors.Join(err, s.metalCache.close())
		s.metalCache = nil
	}

	if s.plan != nil {
		err = errors.Join(err, s.plan.release())
	}

	s.closed, s.plan = true, nil
	return err
}
