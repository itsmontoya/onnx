package onnx

import "context"

func newValueLifetimes(ctx context.Context, plan *executionPlan) (out *valueLifetimes, err error) {
	var l valueLifetimes
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	count := len(plan.values)
	l.owners, l.consumers = make([]int, count), make([]int, count)
	l.members, l.retire = make([][]int, count), make([][]int, len(plan.steps))
	l.pinned = make([]bool, count)
	work := make([]int, count)
	if err = l.aliases(ctx, plan); err != nil {
		return nil, err
	}

	l.groupMembers(work)
	if err = l.countConsumers(ctx, plan, work); err != nil {
		return nil, err
	}

	if err = l.scheduleRetirements(ctx, len(plan.steps), work); err != nil {
		return nil, err
	}

	return &l, nil
}

// valueLifetimes groups Identity aliases and pins source/constant/output storage.
// Consumer occurrences include duplicate inputs; retirement follows the final
// completed consumer, after all node outputs have been validated.
type valueLifetimes struct {
	owners    []int
	consumers []int
	members   [][]int
	retire    [][]int
	pinned    []bool
}

func (l *valueLifetimes) aliases(ctx context.Context, plan *executionPlan) (err error) {
	for index, value := range plan.values {
		if err = ctx.Err(); err != nil {
			return err
		}

		l.owners[index] = index
		l.pinned[index] = value.Producer < 0
	}
	for _, step := range plan.steps {
		if err = ctx.Err(); err != nil {
			return err
		}

		if step.Operator == "Identity" {
			l.owners[step.Outputs[0]] = l.owners[step.Inputs[0]]
		}

		if step.Operator == "Constant" {
			l.pinned[step.Outputs[0]] = true
		}

		if step.Index >= 0 && step.Index < len(plan.calls) && plan.calls[step.Index] != nil && plan.calls[step.Index].retained != nil {
			l.pinned[step.Outputs[0]] = true
		}
	}

	return nil
}

func (l *valueLifetimes) groupMembers(work []int) {
	count := len(l.owners)
	for _, owner := range l.owners {
		work[owner]++
	}
	members := make([]int, count)
	offset := 0
	for owner, size := range work {
		l.members[owner] = members[offset : offset+size : offset+size]
		offset += size
		work[owner] = 0
	}
	for index, owner := range l.owners {
		l.members[owner][work[owner]] = index
		work[owner]++
	}

}

func (l *valueLifetimes) countConsumers(ctx context.Context, plan *executionPlan, work []int) (err error) {
	for _, index := range plan.outputIndices {
		l.pinned[l.owners[index]] = true
	}
	// Reuse the membership cursor for the final consumer position of each group.
	for index := range work {
		work[index] = -1
	}
	for position, step := range plan.steps {
		if err = ctx.Err(); err != nil {
			return err
		}

		for _, index := range step.Inputs {
			if index >= 0 {
				l.consumers[index]++
				work[l.owners[index]] = position
			}
		}
		for _, index := range step.Outputs {
			if index >= 0 && work[l.owners[index]] < 0 {
				work[l.owners[index]] = position
			}
		}
	}

	return nil
}

func (l *valueLifetimes) scheduleRetirements(ctx context.Context, steps int, work []int) (err error) {
	counts := make([]int, steps)
	total := 0
	for owner, position := range work {
		if position >= 0 && l.owners[owner] == owner && !l.pinned[owner] {
			counts[position]++
			total++
		}
	}
	retired := make([]int, total)
	offset := 0
	for position, size := range counts {
		l.retire[position] = retired[offset : offset+size : offset+size]
		offset += size
		counts[position] = 0
	}
	for owner, position := range work {
		if err = ctx.Err(); err != nil {
			return err
		}

		if position >= 0 && l.owners[owner] == owner && !l.pinned[owner] {
			l.retire[position][counts[position]] = owner
			counts[position]++
		}
	}
	return nil
}
