"""Known exact sums for the adversarial input constructors (CPU only)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from expressions_extended import numpy_reference
from test_reduction_inputs import input_values


class ReductionInputTests(unittest.TestCase):
    def test_exact_expected_sums_and_wide_integer_low_bits(self):
        for dtype in ("f16", "f32", "f64", "i32", "i64"):
            for pattern in ("zeros", "negative", "cancellation", "large"):
                with self.subTest(dtype=dtype, pattern=pattern):
                    x = input_values(dtype, (3, 257), pattern)
                    result = numpy_reference(x, operation="row_sum", dtype=dtype)[0]
                    if pattern == "zeros":
                        expected = [0, 0, 0]
                    elif pattern == "negative":
                        expected = [-sum(index % 7 + 1 for index in range(row * 257, (row + 1) * 257))
                                    for row in range(3)]
                    elif pattern == "cancellation":
                        expected = [1, 1, 1]
                    else:
                        expected = [(2 ** 30 + 1 if dtype.startswith("i") else 64) * 257] * 3
                    np.testing.assert_array_equal(result, expected)
                    if dtype.startswith("i") and pattern == "large":
                        self.assertEqual(int(x[0, 0]), 2 ** 30 + 1)
                        self.assertEqual(result.dtype, np.dtype("int64"))


if __name__ == "__main__":
    unittest.main()
