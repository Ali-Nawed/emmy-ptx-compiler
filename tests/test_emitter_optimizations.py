"""Check real Tile schedules and grouped writes from saved Emmy fixtures."""

import math
import re
import shutil
import unittest
import numpy as np

from fixture_checks import ROOT, assemble
from compile import PTXEmitter, load_kernel
from emmy.compiler.ir.stmt import Write


class EmitterOptimizationTests(unittest.TestCase):
    def test_real_tile_coordinate_decoding(self):
        folders = [ROOT / f"tests/extended/{name}" for name in
                   ("ext_add_f32_1", "ext_add_f32_129", "ext_row_sum_f32_3x129")]
        for folder in folders:
            source = load_kernel(next(folder.glob("*.ir.json")))
            tile = source.kernel.body[0]  # Actual Emmy node; no fake Tile/axes.
            emitter = PTXEmitter(source)
            emitter.tile_coordinates(tile, "%worker")
            for worker in range(tile.n_elements):
                registers = {"%worker": worker}
                for instruction in emitter.lines:
                    match = re.fullmatch(r"\s*(shr.u32|and.b32|div.u32|rem.u32) (%r\d+), (%\w+|\d+), (\d+);", instruction)
                    self.assertIsNotNone(match, instruction)
                    opcode, target, operand, immediate = match.groups()
                    value = registers[operand] if operand.startswith("%") else int(operand)
                    divisor = int(immediate)
                    registers[target] = {"shr.u32": lambda: value >> divisor, "and.b32": lambda: value & divisor,
                                         "div.u32": lambda: value // divisor, "rem.u32": lambda: value % divisor}[opcode]()
                got = tuple(registers[v] if v.startswith("%") else int(v) for axis in tile.axes for v in (emitter.values[axis.name],))
                expected = tuple(np.unravel_index(worker, tile.extents)) if tile.axes else ()
                self.assertEqual(got, expected)

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_generated_grouped_stores_and_scalar_fallback(self):
        folders = [path.parent for family in ("vector-store-cases", "vector-store-wide")
                   for path in (ROOT / "tests" / family).glob("*/meta.json")]
        self.assertEqual(len(folders), 15)
        for folder in folders:
            with self.subTest(case=folder.name):
                for source, ptx in assemble(folder, self):
                    writes = [stmt for stmt in source.kernel.body.iter() if isinstance(stmt, Write)]
                    self.assertTrue(writes)
                    for stmt in writes:
                        dtype = next(arg["dtype"] for arg in source.metadata["args"] if arg["name"] == stmt.output)
                        if dtype in ("f64", "i64"):
                            self.assertNotIn(f"st.global.v4.{PTXEmitter.ptx_type(dtype)}", ptx)
                    if "scale_shift_f32" in folder.name:
                        self.assertIn("st.global.v4.f32", ptx)


if __name__ == "__main__":
    unittest.main()
