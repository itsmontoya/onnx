package onnx

import "fmt"

func newRunBudget(limits Limits, residentBytes int64) (out *runBudget, err error) {
	var b runBudget
	b.limits, b.live = limits, residentBytes
	if err = b.reserve(0, 0); err != nil {
		return nil, err
	}

	return &b, nil
}

type runBudget struct {
	limits    Limits
	live      int64
	workspace int64
}

func (b *runBudget) reserve(bytes, workspace int64) (err error) {
	if bytes < 0 || bytes > b.limits.MaxTensorBytes {
		return fmt.Errorf("%w: tensor bytes %d", ErrLimit, bytes)
	}

	if workspace < 0 || b.workspace > b.limits.MaxWorkspaceBytes || workspace > b.limits.MaxWorkspaceBytes-b.workspace {
		return fmt.Errorf("%w: copy workspace bytes %d", ErrLimit, workspace)
	}

	if b.live < 0 || b.live > b.limits.MaxLiveBytes || bytes > b.limits.MaxLiveBytes-b.live || b.workspace > b.limits.MaxLiveBytes-b.live-bytes || workspace > b.limits.MaxLiveBytes-b.live-bytes-b.workspace {
		return fmt.Errorf("%w: live tensor and workspace reservation", ErrLimit)
	}

	b.live += bytes
	b.workspace += workspace
	return nil
}

func (b *runBudget) releaseWorkspace() { b.workspace = 0 }

func (b *runBudget) release(bytes int64) { b.live -= bytes }
