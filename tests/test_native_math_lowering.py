"""Native math checks on unchanged IR generated from PyTorch by Emmy."""

import json
import shutil
import unittest
import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from compile import emit_ptx
from emmy_fixtures import verify


class NativeMathTests(unittest.TestCase):
    def test_f32_and_half_use_native_instructions(self):
        for dtype in ("f16", "f32"):
            for operation, instruction in (("exp", "ex2.approx.f32"), ("log", "lg2.approx.f32")):
                folder = ROOT / f"tests/pytorch-only/range_{operation}_{dtype}"
                verify(folder)
                ptx = emit_ptx(load_kernel(next(folder.glob("*.ir.json"))))
                self.assertIn(instruction, ptx)
                self.assertNotIn(".f64", ptx)
                self.assertNotIn(".ftz", ptx)

    def test_f64_retains_separate_implementation(self):
        baseline = json.loads((ROOT / "review/native-math/before.json").read_text())
        for operation in ("exp", "log", "power"):
            name = f"range_{operation}_f64"
            folder = ROOT / "tests/pytorch-only" / name
            source = load_kernel(next(folder.glob("*.ir.json")))
            record = next(row for row in baseline["cases"] if row["fixture"].endswith("/"+name))
            before_path = ROOT / record["kernels"][source.kernel.name]["ptx"]
            self.assertEqual(emit_ptx(source), before_path.read_text())

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_fresh_emmy_edge_cases(self):
        folders = [path.parent for path in (ROOT/"tests/native-math-edges").glob("*/meta.json")]
        self.assertEqual(len(folders), 6)
        for folder in folders:
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder/"meta.json").read_text())
                with np.load(folder/"expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)


if __name__ == "__main__":
    unittest.main()
