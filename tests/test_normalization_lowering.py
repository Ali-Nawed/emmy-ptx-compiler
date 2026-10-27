"""Normalization and unary math using upstream-generated complete kernels."""

import json
import shutil
import unittest
import numpy as np

from fixture_checks import ROOT, assemble, simulate, compare_outputs
from ir_reference import calculate


class NormalizationLoweringTests(unittest.TestCase):
    def test_reference_special_values_and_precision(self):
        # Pure numeric oracle checks: these construct no IR and emit no PTX.
        for dtype in ("f16", "f32", "f64"):
            values = np.array([0., -0., 4., np.inf, -1., np.nan], dtype=dtype.replace("f", "float"))
            roots = calculate("sqrt", [values], dtype)
            self.assertEqual(roots[2], 2)
            self.assertTrue(np.signbit(roots[1]))
            self.assertTrue(np.isnan(roots[4:]).all())
            inverse = calculate("reciprocal", [values], dtype)
            self.assertTrue(np.isposinf(inverse[0]))
            self.assertTrue(np.isneginf(inverse[1]))
        np.testing.assert_array_equal(calculate("sqrt", [np.array([1e200])], "f64"), [1e100])

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_generated_unary_and_public_normalization(self):
        folders = [ROOT / "problems" / name for name in ("f03_l2norm", "f01_rmsnorm")]
        folders += [path.parent for path in (ROOT / "tests/normalization").glob("norm_*/meta.json")]
        self.assertEqual(len(folders), 20)
        for folder in folders:
            with self.subTest(case=folder.name):
                assemble(folder, self)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)


if __name__ == "__main__":
    unittest.main()
