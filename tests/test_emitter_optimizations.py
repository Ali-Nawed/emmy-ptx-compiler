"""Check coordinate semantics and alignment-sensitive vector-store lowering."""

import copy
from dataclasses import replace
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import PTXEmitter, emit_ptx, load_kernel
from emmy.compiler.dtype import F16, F32, F64, I32, I64
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Load, Write


class EmitterOptimizationTests(unittest.TestCase):
    def source(self, dtype="f32"):
        return load_kernel(next((ROOT / f"tests/extended/ext_add_{dtype}_1").glob("*.ir.json")))

    def test_coordinate_decode_matches_row_major_indices(self):
        for extents in ((), (1,), (17, 128), (1, 128), (3, 5, 7), (2, 1, 4), (2, 4, 8)):
            with self.subTest(extents=extents):
                emitter = PTXEmitter(self.source())
                axes = [SimpleNamespace(name=f"axis{i}") for i in range(len(extents))]
                tile = SimpleNamespace(axes=axes, extents=extents, n_elements=math.prod(extents))
                emitter.tile_coordinates(tile, "%worker")
                for worker in range(tile.n_elements):
                    registers = {"%worker": worker}
                    for instruction in emitter.lines:
                        match = re.fullmatch(r"\s*(shr.u32|and.b32|div.u32|rem.u32) (%r\d+), (%\w+|\d+), (\d+);", instruction)
                        self.assertIsNotNone(match, instruction)
                        opcode, target, operand, immediate = match.groups()
                        value = registers[operand] if operand.startswith("%") else int(operand)
                        divisor = int(immediate)
                        registers[target] = {"shr.u32": lambda: value >> divisor,
                                             "and.b32": lambda: value & divisor,
                                             "div.u32": lambda: value // divisor,
                                             "rem.u32": lambda: value % divisor}[opcode]()
                    coordinates = tuple(registers[value] if value.startswith("%") else int(value)
                                        for axis in axes for value in (emitter.values[axis.name],))
                    expected = tuple(np.unravel_index(worker, extents)) if extents else ()
                    self.assertEqual(coordinates, expected)
                if extents == (17, 128):
                    self.assertEqual(len(emitter.lines), 2)
                    self.assertNotIn("div.u32", "\n".join(emitter.lines))
                    self.assertNotIn("rem.u32", "\n".join(emitter.lines))

    def test_alignment_proof_respects_rows_offsets_and_shared_base(self):
        emitter = PTXEmitter(self.source())
        out = next(arg["name"] for arg in emitter.source.metadata["args"] if arg.get("output"))
        emitter.values.update(a="%f1", b="%f2", c="%f3", d="%f4")
        emitter.value_dtypes.update({name: "f32" for name in "abcd"})
        twice = BinaryExpr("*", Var("col"), Literal(2, "int"))
        for columns, index, count, alignment, expected in (
                (5, (Var("row"), twice), 2, 16, False),
                (8, (Var("row"), twice), 2, 16, True),
                (8, (Var("row"), twice), 4, 16, False),
                (8, (Var("row"), BinaryExpr("*", twice, Literal(2, "int"))), 4, 16, True),
                (8, (Literal(0, "int"), Literal(1, "int")), 4, 16, False),
                (8, (Literal(0, "int"), Literal(0, "int")), 4, 4, False)):
            with self.subTest(columns=columns, count=count, alignment=alignment, index=index):
                emitter.buffers[out] = {"dtype": "f32", "shape": [3, columns]}
                emitter.buffer_alignments[out] = alignment
                stmt = Write(output=out, index=index, values=tuple("abcd"[:count]), value_dtype=F32)
                self.assertEqual(emitter.vector_store(stmt, "%rd1", "f32"), expected)

    @unittest.skipUnless(shutil.which("ptxas"), "requires CUDA ptxas in Ubuntu")
    def test_vector_datatypes_assemble_and_wide_vectors_fall_back(self):
        for dtype in (F16, F32, F64, I32, I64):
            for count in (2, 4):
                with self.subTest(dtype=dtype.name, width=count), tempfile.TemporaryDirectory() as temporary:
                    source = self.source(dtype.name)
                    metadata = copy.deepcopy(source.metadata)
                    out = next(arg for arg in metadata["args"] if arg.get("output"))
                    out["shape"] = [count]
                    inputs = [arg["name"] for arg in metadata["args"] if not arg.get("output")]
                    body = (Load(names=("a",), input=inputs[0], index=(Literal(0, "int"),), dtype=dtype),
                            Load(names=("b",), input=inputs[1], index=(Literal(0, "int"),), dtype=dtype),
                            Write(output=out["name"], index=(Literal(0, "int"),),
                                  values=("a", "b") * (count // 2), value_dtype=dtype))
                    kernel = replace(source.kernel, body=(replace(source.kernel.body[0], body=body),))
                    ptx = emit_ptx(replace(source, kernel=kernel, metadata=metadata))
                    should_vectorize = count * PTXEmitter.element_bytes(dtype.name) <= 16
                    self.assertEqual(f"st.global.v{count}." in ptx, should_vectorize)
                    path = Path(temporary) / "kernel.ptx"
                    path.write_text(ptx)
                    result = subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                                            capture_output=True, text=True, timeout=35)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
