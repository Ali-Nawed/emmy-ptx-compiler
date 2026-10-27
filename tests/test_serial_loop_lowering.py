"""Exercise complete, unchanged Emmy-generated Loop programs."""

import json
import shutil
import unittest

import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from compile import PTXEmitter
from emmy.compiler.ir.stmt import Accum, Loop
from emmy_fixtures import verify


class SerialLoopTests(unittest.TestCase):
    def test_loop_scope_preserves_accumulators_and_hides_locals(self):
        folder = ROOT / "tests/serial-loops/loop_3sums_f32_3x129_w128"
        verify(folder)
        source = load_kernel(next(folder.glob("*.ir.json")))
        loops = [s for s in source.kernel.body.iter() if isinstance(s, Loop)]
        self.assertGreaterEqual(len(loops), 1)
        emitter = PTXEmitter(source)
        emitter.compile()
        for loop in loops:
            self.assertNotIn(loop.axis.name, emitter.values)
            for statement in loop.body:
                if isinstance(statement, Accum):
                    self.assertIn(statement.name, emitter.values)
                    self.assertIn(statement.name, emitter.value_dtypes)
                for name in getattr(statement, "names", ()):
                    self.assertNotIn(name, emitter.values)

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_generated_datatypes_sizes_and_schedules(self):
        folders = [path.parent for path in (ROOT / "tests/serial-loops").glob("*/meta.json")]
        self.assertEqual(len(folders), 18)
        for folder in folders:
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)


if __name__ == "__main__":
    unittest.main()
