"""Compile and assemble the fresh Emmy-generated f16/f32 fixture forms locally."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from expressions_extended import build_specs


@unittest.skipUnless(shutil.which("ptxas"), "requires CUDA ptxas (run in Ubuntu)")
class CompilerRegressionTests(unittest.TestCase):
    def test_generated_float_cases_compile_and_assemble(self):
        specs = build_specs(dtypes=("f16", "f32"), sizes=(1, 129), row_counts=(3,))
        cases = [ROOT / "tests/extended" / spec["id"] for spec in specs]
        with tempfile.TemporaryDirectory() as temporary:
            for case in cases:
                with self.subTest(case=case.name):
                    inputs = list(case.glob("*.ir.json"))
                    self.assertTrue(inputs, "generate the default extended fixtures first")
                    for ir in inputs:
                        compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                                  capture_output=True, text=True, timeout=30)
                        self.assertEqual(compiled.returncode, 0, compiled.stderr)
                        self.assertFalse(compiled.stderr)
                        ptx = Path(temporary) / "out.ptx"
                        ptx.write_text(compiled.stdout)
                        assembled = subprocess.run(["ptxas", "-arch=sm_89", str(ptx),
                                                    "-o", str(ptx.with_suffix(".cubin"))],
                                                   capture_output=True, text=True, timeout=35)
                        self.assertEqual(assembled.returncode, 0, assembled.stderr)


if __name__ == "__main__":
    unittest.main()
