"""Keep kernel tests on the required PyTorch -> Emmy -> compiler path."""

import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fixture_checks import ROOT
from emmy_fixtures import verify


class FixturePolicyTests(unittest.TestCase):
    def test_active_test_code_never_constructs_or_rewrites_ir(self):
        constructors = {"KernelOp", "Tile", "Assign", "Accum", "Load", "Write", "Let", "Init",
                        "Loop", "StridedLoop", "WarpShuffle", "TreeHalve", "Cond", "Smem", "Sync",
                        "Axis", "Dim", "Var", "Literal", "BinaryExpr", "TernaryExpr", "ElementwiseImpl", "KernelInput"}
        constructors.update({"RegFragment", "LdmatrixLoad", "MmaSyncPtx", "RegStore", "CpAsyncCopy", "CpAsyncCommit", "CpAsyncWait"})
        for path in [*list((ROOT / "tests").glob("*.py")), *list((ROOT / "tools").glob("*.py"))]:
            if path.name == "generate.py":
                continue  # Competition-owned upstream generator: read-only.
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
                    self.assertFalse(any(alias.name == "replace" for alias in node.names), str(path))
                if isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else ""
                    self.assertNotIn(name, constructors, f"{path}:{node.lineno}: invented IR constructor")
                    if isinstance(node.func, ast.Attribute):
                        self.assertNotEqual(node.func.attr, "with_bodies", f"{path}:{node.lineno}: IR rewriting")

    def test_fresh_fixture_hashes_match_emmy_generation(self):
        proofs = list((ROOT / "tests/pytorch-only").glob("*/emmy-proof.json"))
        self.assertEqual(len(proofs), 34, "Generate conversion and range fixtures first")
        for path in proofs:
            with self.subTest(case=path.parent.name):
                proof = verify(path.parent)
                self.assertIn("PyTorch", proof["pipeline"])

    def test_changed_ir_hash_is_rejected_without_editing_any_ir(self):
        folder = ROOT / "tests/pytorch-only/convert_f32_to_f64"
        with patch("emmy_fixtures.digest", return_value="different-file-hash"):
            with self.assertRaisesRegex(ValueError, "changed after Emmy"):
                verify(folder)

    def test_active_metadata_has_no_synthetic_kernel_fixtures(self):
        for path in (ROOT / "tests").rglob("meta.json"):
            meta = json.loads(path.read_text())
            self.assertNotIn("Synthetic typed-IR", meta.get("code", ""), str(path))
            self.assertFalse(meta.get("id", "").startswith("typed_backend"), str(path))
        self.assertFalse((ROOT / "tools/test_operation_backends.py").exists())
        self.assertFalse((ROOT / "tools/test_custom.py").exists())


if __name__ == "__main__":
    unittest.main()
