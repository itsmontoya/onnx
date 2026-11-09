"""Check numerical comparison boundaries used by the optional runtime driver."""
import unittest

import numpy as np

from compare_runtime import compare, decode_tensor, encode_tensor


class ComparisonTests(unittest.TestCase):
    def test_comparison(self):
        cases = [
            ("absolute tolerance", np.array([0.00009], np.float32), np.array([0], np.float32), True),
            ("outside absolute tolerance", np.array([0.0002], np.float32), np.array([0], np.float32), False),
            ("relative tolerance", np.array([1000.005], np.float32), np.array([1000], np.float32), True),
            ("outside relative tolerance", np.array([1000.02], np.float32), np.array([1000], np.float32), False),
            ("NaN agreement", np.array([np.nan], np.float32), np.array([np.nan], np.float32), True),
            ("NaN mismatch", np.array([np.nan], np.float32), np.array([1], np.float32), False),
            ("infinity agreement", np.array([-np.inf], np.float32), np.array([-np.inf], np.float32), True),
            ("infinity sign mismatch", np.array([np.inf], np.float32), np.array([-np.inf], np.float32), False),
            ("finite versus infinity", np.array([1], np.float32), np.array([np.inf], np.float32), False),
            ("signed zero ignored", np.array([-0.0], np.float32), np.array([0.0], np.float32), True),
            ("scalar", np.array(2, np.float32), np.array(2, np.float32), True),
            ("empty", np.empty((2, 0), np.float32), np.empty((2, 0), np.float32), True),
            ("shape mismatch", np.zeros((1, 2), np.float32), np.zeros((2,), np.float32), False),
            ("dtype mismatch", np.array([1], np.int64), np.array([1], np.float32), False),
            ("exact large integers", np.array([2**63 - 1], np.int64), np.array([2**63 - 1], np.int64), True),
            ("integer difference beyond float precision", np.array([2**63 - 2], np.int64), np.array([2**63 - 1], np.int64), False),
        ]
        for name, actual, expected, passed in cases:
            with self.subTest(name=name):
                result = compare(actual, expected, atol=1e-4, rtol=1e-5)
                self.assertEqual(result["passed"], passed)

    def test_exact_float_comparison(self):
        actual = np.array([1, np.nextafter(np.float32(1), np.float32(2))], np.float32)
        expected = np.array([1, 1], np.float32)
        result = compare(actual, expected, atol=0, rtol=0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatched_elements"], 1)

    def test_tensor_encoding(self):
        cases = [
            ("float bits", np.array([0, 0x80000000, 0x7fc00001, 0x7f800000], np.uint32).view(np.float32)),
            ("integer limits", np.array([-2**63, 2**63 - 1], np.int64)),
            ("scalar", np.array(-0.0, np.float32)),
            ("empty", np.empty((2, 0, 3), np.int64)),
            ("noncontiguous", np.arange(12, dtype=np.float32).reshape(3, 4).T),
        ]
        for name, array in cases:
            with self.subTest(name=name):
                result = decode_tensor(encode_tensor("input", array))
                self.assertEqual(result.shape, array.shape)
                self.assertEqual(result.dtype, array.dtype)
                self.assertEqual(result.tobytes(), array.tobytes())


if __name__ == "__main__":
    unittest.main()
