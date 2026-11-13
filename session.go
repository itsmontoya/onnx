package onnx

import (
	"context"
	"sync"
)

func newSession(plan *executionPlan) (out *Session) {
	var s Session
	s.plan, s.initialized = plan, true
	s.destinations = newDestinationStorage(plan.destinationLimit, plan.destinationMinimum, len(plan.steps))
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
	plan         *executionPlan
	initialized  bool
	closed       bool
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

	if s.plan.limits.MaxLiveBytes < s.destinations.minimumLive {
		s.destinations = destinationStorage{}
	}

	run.storage.destinations = &s.destinations
	// Admission reserves room for residency in addition to every legacy Run
	// reservation, including workspace, invalid bindings and output clones.
	run.budget.live += s.destinations.bytes
	if outputs, err = run.run(inputs, options); err != nil {
		return nil, operationError("run", err, "")
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
	s.plan.release()
	s.closed, s.plan = true, nil
	return nil
}
