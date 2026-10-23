"""Independent numerical checks for the CPU IR oracle; no GPU calls."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from ir_reference import Block, calculate, cast, save_reference, simulate
from test_extended import check_runner, compare_outputs
from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import WarpShuffle, TreeHalve
from emmy.compiler.ir.stmt import Assign, Load


class IRReferenceTests(unittest.TestCase):
    def test_all_80_generated_cases_and_known_precision_gaps(self):
        folders = sorted((ROOT / "tests/extended").glob("ext_*"))
        self.assertEqual(len(folders), 80)
        failures = []
        known = {"ext_sum_f64_129": [7.388847351074219],
                 "ext_mean_f64_129": [0.0572778864424358],
                 "ext_row_sum_f64_3x129": [7.388847351074219, -7.408513069152832, -14.822206497192383]}
        for folder in folders:
            with self.subTest(case=folder.name):
                original = hashlib.sha256((folder / "expected.npz").read_bytes()).hexdigest()
                reference = simulate(folder)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, _ = compare_outputs(reference, expected, meta)
                    self.assertEqual(set(reference), set(expected.files))
                    for name, value in reference.items():
                        self.assertEqual(value.dtype, expected[name].dtype)
                        # Metadata represents a scalar buffer as [1]; NumPy's
                        # expression reduction is a zero-dimensional scalar.
                        self.assertEqual(value.size, expected[name].size)
                if status != "pass":
                    failures.append(folder.name)
                    np.testing.assert_array_equal(next(iter(reference.values())).ravel(), known[folder.name])
                self.assertEqual(original, hashlib.sha256((folder / "expected.npz").read_bytes()).hexdigest())
        self.assertEqual(set(failures), set(known))

    def test_native_precision_and_half_operation_boundaries(self):
        x = np.array([2 ** 60], np.int64)
        np.testing.assert_array_equal(calculate("add", [x, np.array([3], np.int64)], "i64"), [2 ** 60 + 3])
        np.testing.assert_array_equal(calculate("multiply", [np.array([2 ** 62], np.int64), np.array([4], np.int64)], "i64"), [0])
        delta = calculate("subtract", [np.array([1 + 2 ** -40]), np.array([1.0])], "f64")
        np.testing.assert_array_equal(delta, [2 ** -40])
        rounded = calculate("add", [np.array([2048], np.float16), np.array([1], np.float16)], "f16")
        np.testing.assert_array_equal(calculate("subtract", [rounded, np.array([2048], np.float16)], "f16"), [0])
        with self.assertRaises(NotImplementedError):
            cast(np.array([float(2 ** 63)]), "i64")

    def test_saved_numeric_edges_and_larger_cases(self):
        families = ("f16-edges", "extended-f16-large", "i64-output-bits", "numeric-edges")
        for family in families:
            folders = sorted((ROOT / "tests" / family).glob("*/meta.json"))
            self.assertTrue(folders)
            for path in folders:
                with self.subTest(case=str(path.parent.relative_to(ROOT))):
                    meta = json.loads(path.read_text())
                    result = simulate(path.parent)
                    with np.load(path.parent / "expected.npz") as expected:
                        status, details = compare_outputs(result, expected, meta)
                    self.assertEqual(status, "pass", details)

    def test_masked_load_and_uninitialized_reads(self):
        block = Block(4, {"x": np.array([10, 20], np.float32)}, {"x": np.ones(2, bool)})
        mask = np.array([True, True, False, False])
        block.bind("index", np.arange(4, dtype=np.int64), np.ones(4, bool))
        load = Load(names=("value",), input="x", index=(Var("index"),), dtype=F32)
        block.statement(load, mask)
        np.testing.assert_array_equal(block.read("value", mask)[mask], [10, 20])
        with self.assertRaises(ValueError):
            block.read("value", np.ones(4, bool))
        with self.assertRaises(IndexError):
            block.statement(load, np.ones(4, bool))

    def test_shuffle_keeps_warps_separate_and_updates_simultaneously(self):
        block = Block(64, {}, {})
        mask = np.ones(64, bool)
        block.bind("state", np.arange(64, dtype=np.float32), mask)
        merge = (Assign(name="state", op="add", args=("state", "other")),)
        stmt = WarpShuffle(state=("state",), state_b=("other",), combine_states=merge, length=32, dtype=F32)
        block.statement(stmt, mask)
        np.testing.assert_array_equal(block.read("state", mask), [496] * 32 + [1520] * 32)
        with self.assertRaises(NotImplementedError):
            block.statement(stmt, np.arange(64) != 1)

    def test_shared_tree_uses_halving_order_and_broadcasts(self):
        values = np.array([1e20, 3, -1e20, 4], np.float32)
        block = Block(128, {"scratch": values.copy()}, {"scratch": np.ones(4, bool)})
        mask = np.ones(128, bool)
        block.bind("warp", np.arange(128, dtype=np.int64) // 32, mask)
        block.bind("state", np.zeros(128, np.float32), mask)
        merge = (Assign(name="state", op="add", args=("state", "other")),)
        stmt = TreeHalve(bufs=("scratch",), state=("state",), state_b=("other",),
                         combine_states=merge, length=4, tid_var="warp", dtype=F32)
        block.statement(stmt, mask)
        np.testing.assert_array_equal(block.read("state", mask), [7] * 128)

    def test_saving_does_not_replace_expression_expectations(self):
        import shutil
        source = ROOT / "tests/extended/ext_sum_f64_129"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for path in (source / "meta.json", source / "expected.npz", *source.glob("*.ir.json")):
                shutil.copyfile(path, folder / path.name)
            original = (folder / "expected.npz").read_bytes()
            save_reference(folder)
            self.assertEqual(original, (folder / "expected.npz").read_bytes())
            with np.load(folder / "expected_ir.npz") as saved:
                np.testing.assert_array_equal(saved["sum_1"], [7.388847351074219])
            provenance = json.loads((folder / "expected_ir.provenance.json").read_text())
            self.assertIn("ir_reference.py", provenance["source_sha256"])

    def test_gpu_dual_report_retains_expression_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            np.savez(folder / "expected.npz", out=np.array([1.0], np.float64))
            meta = {"rtol": 0, "atol": 0, "kernels": [{
                "entry": "k_test", "grid": [1, 1, 1], "block": [1, 1, 1],
                "smem_bytes": 0, "zero_outputs": [],
                "args": [{"name": "out", "shape": [1], "output": True}]}]}
            ir_reference = {"out": np.array([2.0], np.float64)}
            replies = [{"ok": True, "cubin_id": "unit-test"},
                       {"ok": True, "outputs": {"out": {"values": [2.0]}}}]
            with patch("test_extended.post", side_effect=replies):
                status, details = check_runner(folder, meta, {"k_test": "unit-test PTX"}, ir_reference)
            self.assertEqual(status, "wrong_result")
            self.assertEqual(details["comparisons"]["expression"]["status"], "wrong_result")
            self.assertEqual(details["comparisons"]["kernel_ir"]["status"], "pass")
            replies[1]["outputs"]["out"]["values"] = [3.0]
            with patch("test_extended.post", side_effect=replies):
                _, details = check_runner(folder, meta, {"k_test": "unit-test PTX"}, ir_reference)
            self.assertEqual(details["comparisons"]["kernel_ir"]["status"], "wrong_result")


if __name__ == "__main__":
    unittest.main()
