//go:build metal && !purego && darwin && cgo

package onnx

const copyingProductWorkspace = true

// productWorkspace conservatively covers the retained Matrix copying fallback.
func productWorkspace(c *numericalCall, left, right []int64) (bytes int64, err error) {
	switch c.step.Operator {
	case "Gemm":
		inner := left[1]
		if c.step.Gemm.TransposeA {
			inner = left[0]
		}

		if inner == 0 {
			return c.bytes, nil
		}

		parts := []int64{c.inputs[0].ByteSize(), c.inputs[1].ByteSize(), c.bytes, c.bytes}
		if c.step.Gemm.TransposeA && c.step.Gemm.TransposeB {
			parts = append(parts, c.inputs[1].ByteSize())
		}

		return sumScratch(parts...)
	case "MatMul":
		inner := left[len(left)-1]
		if inner == 0 {
			return 0, nil
		}

		rows, cols := int64(1), int64(1)
		if len(left) > 1 {
			rows = left[len(left)-2]
		}

		if len(right) > 1 {
			cols = right[len(right)-1]
		}

		// Input shapes and output bytes have already passed checked validation.
		// Divide output elements by each free dimension to avoid unchecked products.
		elements := c.bytes / 4
		var leftCopies, rightCopies, products int64
		if leftCopies, err = checkedScratch(elements/cols*4, inner); err != nil {
			return 0, err
		}

		if rightCopies, err = checkedScratch(elements/rows*4, inner); err != nil {
			return 0, err
		}

		if products, err = checkedScratch(c.bytes, 3); err != nil {
			return 0, err
		}

		return sumScratch(leftCopies, rightCopies, products)
	}
	return 0, nil
}
