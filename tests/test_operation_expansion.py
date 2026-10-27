"""Expanded operations tested exclusively on upstream Emmy IR."""

import json
import shutil
import unittest
import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from ir_reference import Block, calculate
from emmy.compiler.ir.kernel.ir import WarpShuffle


class OperationExpansionTests(unittest.TestCase):
    def test_public_online_softmax_reference(self):
        folder = ROOT / "problems/f02_row_softmax"
        meta = json.loads((folder / "meta.json").read_text())
        with np.load(folder / "expected.npz") as expected:
            status, details = compare_outputs(simulate(folder), expected, meta)
            self.assertEqual(status, "pass", details)

    def test_promoted_merge_from_real_emmy_schedule(self):
        for dtype, value in (("f16", 1.), ("f64", 1.+2**-40), ("i64", 2**60+3)):
            folder = ROOT / f"tests/operations/ops_maximum_{dtype}_3x129_w128"
            source = load_kernel(next(folder.glob("*.ir.json")))
            shuffle = next(stmt for stmt in source.kernel.body.iter() if isinstance(stmt, WarpShuffle))
            numpy_type = {"f16": np.float16, "f64": np.float64, "i64": np.int64}[dtype]
            mask = np.ones(32, bool)
            block = Block(32, {}, {})
            block.bind(shuffle.state[0], np.full(32, value, dtype=numpy_type), mask)
            block.statement(shuffle, mask)
            expected = np.asarray(value, np.float32).astype(numpy_type)
            np.testing.assert_array_equal(block.read(shuffle.state[0], mask), np.full(32, expected, dtype=numpy_type))

    def test_integer_power_numeric_oracle(self):
        np.testing.assert_array_equal(calculate("power", [np.array([2, -3, 0], np.int32), np.array([31, 3, 0], np.int32)], "i32"),
                                      [-2147483648, -27, 1])
        with self.assertRaises(NotImplementedError):
            calculate("power", [np.array([2]), np.array([-1])], "i64")

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_genuine_range_kernels(self):
        folders = [path.parent for path in (ROOT / "tests/pytorch-only").glob("range_*/meta.json")]
        self.assertEqual(len(folders), 9)
        for folder in folders:
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)


if __name__ == "__main__":
    unittest.main()
