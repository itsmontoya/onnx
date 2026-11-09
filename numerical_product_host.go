//go:build !metal || purego || !darwin || !cgo

package onnx

const copyingProductWorkspace = false

// productWorkspace covers NN-13's one call-owned output staging buffer.
func productWorkspace(c *numericalCall, left, right []int64) (bytes int64, err error) {
	if c.step.Operator == "MatMul" && left[len(left)-1] == 0 {
		return 0, nil
	}

	return c.bytes, nil
}
