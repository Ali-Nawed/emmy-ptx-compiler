"""Activation checks on unchanged kernels produced by pinned Emmy."""

import json
import shutil
import unittest

import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from compile import emit_ptx
from emmy_fixtures import verify


class ActivationTests(unittest.TestCase):
    def test_native_float_paths_and_full_width_double(self):
        for operation in ("sigmoid", "tanh"):
            for dtype in ("f16", "f32", "f64"):
                folder = ROOT / f"tests/activations/{operation}_{dtype}_129"
                verify(folder)
                ptx = emit_ptx(load_kernel(next(folder.glob("*.ir.json"))))
                if dtype == "f64":
                    self.assertNotIn(".f32", ptx)
                    self.assertNotIn("tanh.approx", ptx)
                    self.assertIn(".f64", ptx)
                else:
                    self.assertIn("tanh.approx.f32" if operation == "tanh" and dtype == "f16" else "ex2.approx.f32", ptx)
                    self.assertNotIn(".f64", ptx)
                    self.assertNotIn(".ftz", ptx)

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_generated_examples_assemble_and_preserve_typed_semantics(self):
        folders = [path.parent for path in (ROOT / "tests/activations").glob("*/meta.json")]
        self.assertEqual(len(folders), 30)
        for folder in folders:
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder / "meta.json").read_text())
                # Keep an independent expression comparison; never overwrite
                # its expected values with results from the IR interpreter.
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    if folder.name == "tanh_sum_f64":
                        # Preserve the producer's f32 reduction stages and
                        # explicitly retain their expression-level difference.
                        self.assertEqual(status, "wrong_result", details)
                    else:
                        self.assertEqual(status, "pass", details)

    def test_double_tanh_edge_reference_preserves_signed_zero(self):
        folder = ROOT / "tests/activations/tanh_edges_f64"
        verify(folder)
        outputs = simulate(folder)
        with np.load(folder / "expected.npz") as expected:
            for name, actual in outputs.items():
                zeros = expected[name] == 0
                self.assertTrue(np.array_equal(np.signbit(actual[zeros]), np.signbit(expected[name][zeros])))


if __name__ == "__main__":
    unittest.main()
