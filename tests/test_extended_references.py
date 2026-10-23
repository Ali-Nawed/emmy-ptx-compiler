"""Independent CPU tests for the test suite's NumPy oracle (no runner calls)."""

import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from expressions_extended import build_specs
from test_extended import check_runner, validate_numpy


class ExtendedReferenceTests(unittest.TestCase):
    def test_all_default_references_match_torch(self):
        specs = build_specs(sizes=(1, 129), row_counts=(3,))
        self.assertEqual(len(specs), 80)
        for spec in specs:
            with self.subTest(case=spec["id"]):
                validate_numpy(spec)

    def test_matrix_has_unique_case_names(self):
        specs = build_specs(row_counts=(1, 3, 17))
        self.assertEqual(len(specs), len({spec["id"] for spec in specs}))

    def test_checker_preserves_f64_and_exact_i64(self):
        for dtype, value in ((np.float64, 1 + 2 ** -30), (np.int64, 2 ** 54 + 3)):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                np.savez(folder / "expected.npz", out=np.array([value], dtype=dtype))
                meta = {"rtol": 0, "atol": 0, "kernels": [{
                    "entry": "k_test", "grid": [1, 1, 1], "block": [1, 1, 1],
                    "smem_bytes": 0, "zero_outputs": [],
                    "args": [{"name": "out", "shape": [1], "output": True}],
                }]}
                responses = [{"ok": True, "cubin_id": "unit-test"},
                             {"ok": True, "outputs": {"out": {"values": [value]}}}]
                with patch("test_extended.post", side_effect=responses):
                    status, _ = check_runner(folder, meta, {"k_test": "unit-test PTX"})
                    self.assertEqual(status, "pass")
                # A small difference must still fail an exact comparison at this magnitude.
                different = value + (2 ** -30 if dtype == np.float64 else 1)
                responses[1] = {"ok": True, "outputs": {"out": {"values": [different]}}}
                with patch("test_extended.post", side_effect=responses):
                    status, _ = check_runner(folder, meta, {"k_test": "unit-test PTX"})
                    self.assertEqual(status, "wrong_result")


if __name__ == "__main__":
    unittest.main()
