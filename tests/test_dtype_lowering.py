"""Conversions are traced from PyTorch, never assembled from synthetic IR."""

import json
import shutil
import unittest
import numpy as np

from fixture_checks import ROOT, assemble, simulate, compare_outputs


@unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
class DtypeLoweringTests(unittest.TestCase):
    def test_all_generated_conversion_pairs(self):
        for source in ("f16", "f32", "f64", "i32", "i64"):
            for target in ("f16", "f32", "f64", "i32", "i64"):
                with self.subTest(source=source, target=target):
                    folder = ROOT / f"tests/pytorch-only/convert_{source}_to_{target}"
                    assemble(folder, self, require_proof=True)
                    meta = json.loads((folder / "meta.json").read_text())
                    with np.load(folder / "expected.npz") as expected:
                        status, details = compare_outputs(simulate(folder), expected, meta)
                        self.assertEqual(status, "pass", details)


if __name__ == "__main__":
    unittest.main()
