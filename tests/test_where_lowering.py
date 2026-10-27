"""Boolean tests use complete kernels produced by PyTorch and pinned Emmy."""

import json
import shutil
import unittest

import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from compile import emit_ptx
from emmy_fixtures import verify


class WhereTests(unittest.TestCase):
    def test_adjacent_wide_integer_comparison_never_uses_float(self):
        for operation in ("gt", "ge", "lt", "le", "eq", "ne"):
            folder = ROOT / f"tests/where/where_wide_{operation}_i64"
            verify(folder)
            ptx = emit_ptx(load_kernel(next(folder.glob("*.ir.json"))))
            self.assertIn(f"setp.{operation}.s64", ptx)
            self.assertIn("selp.b64", ptx)
            self.assertNotIn(".f32", ptx)
            self.assertNotIn(".f64", ptx)

    def test_nan_not_equal_is_unordered_and_selection_uses_predicates(self):
        folder = ROOT / "tests/where/where_nan_ne_f32"
        verify(folder)
        ptx = emit_ptx(load_kernel(next(folder.glob("*.ir.json"))))
        self.assertIn("setp.neu.f32", ptx)
        self.assertIn("selp.f32", ptx)
        self.assertNotIn("cvt.rn.f32.pred", ptx)

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_all_generated_examples_assemble_and_match_expression(self):
        folders = [path.parent for path in (ROOT / "tests/where").glob("*/meta.json")]
        self.assertEqual(len(folders), 41)
        for folder in folders:
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)


if __name__ == "__main__":
    unittest.main()
