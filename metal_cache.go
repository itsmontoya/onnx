package onnx

import (
	"errors"
	"slices"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

func newMetalCache(device *gpu.Device) (out *metalCache) {
	var c metalCache
	c.device = device
	c.entries = make(map[metalCacheKey]*gpu.Tensor)
	return &c
}

type metalCacheKey struct {
	step int
	slot int
}

// metalCache retains session-private FP32 Metal activation buffers across runs
// to avoid repeated MTLBuffer allocations on static or repeated shapes.
type metalCache struct {
	device  *gpu.Device
	entries map[metalCacheKey]*gpu.Tensor
	bytes   uint64
}

func (c *metalCache) acquire(step, slot int, shape []int64) (out *gpu.Tensor, err error) {
	key := metalCacheKey{step: step, slot: slot}
	if existing, ok := c.entries[key]; ok {
		if slices.Equal(existing.Shape(), shape) {
			return existing, nil
		}

		c.bytes -= existing.Bytes()
		delete(c.entries, key)
		if err = existing.Close(); err != nil {
			return nil, err
		}
	}

	if out, err = c.device.NewFloat32(shape); err != nil {
		return nil, err
	}

	c.entries[key] = out
	c.bytes += out.Bytes()
	return out, nil
}

func (c *metalCache) stats() (buffers int, bytes uint64) {
	if c == nil {
		return 0, 0
	}

	return len(c.entries), c.bytes
}

func (c *metalCache) close() (err error) {
	if c == nil {
		return nil
	}

	for key, entry := range c.entries {
		if entry != nil {
			err = errors.Join(err, entry.Close())
		}

		delete(c.entries, key)
	}

	c.bytes = 0
	return err
}
