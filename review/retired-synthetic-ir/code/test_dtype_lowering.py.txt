"""Backend unit cases, separate from fresh PyTorch-generated extended tests."""

import copy
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import emit_ptx, load_kernel
from emmy.compiler.dtype import F16, F32, F64, I32, I64
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import Smem, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, Assign, Load, Write

DTYPES = {dtype.name: dtype for dtype in (F16, F32, F64, I32, I64)}


@unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
class DtypeLoweringTests(unittest.TestCase):
    def assemble(self, source):
        with tempfile.TemporaryDirectory() as temporary:
            ptx = Path(temporary) / "test.ptx"
            ptx.write_text(emit_ptx(source))
            result = subprocess.run(["ptxas", "-arch=sm_89", str(ptx), "-o", str(ptx.with_suffix(".cubin"))],
                                    capture_output=True, text=True, timeout=35)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_numeric_conversion_pairs(self):
        for input_name, input_dtype in DTYPES.items():
            for output_name, output_dtype in DTYPES.items():
                with self.subTest(source=input_name, destination=output_name):
                    path = next((ROOT / f"tests/extended/ext_add_{input_name}_1").glob("*.ir.json"))
                    source = load_kernel(path)
                    metadata = copy.deepcopy(source.metadata)
                    output = next(arg for arg in metadata["args"] if arg.get("output"))
                    output["dtype"] = output_name
                    x = metadata["arg_order"][0]
                    body = (Load(names=("input_value",), input=x, index=(Literal(0, "int"),), dtype=input_dtype),
                            Assign(name="converted", op=ElementwiseImpl("copy"), args=("input_value",), dtype=output_dtype),
                            Write(output=output["name"], index=(Literal(0, "int"),), values=("converted",), value_dtype=output_dtype))
                    tile = replace(source.kernel.body[0], body=body)
                    kernel = replace(source.kernel, body=(tile,))
                    self.assemble(replace(source, kernel=kernel, metadata=metadata))

    def test_wide_accumulator_and_shuffle_backends(self):
        # The pinned generator emits f32 sums. Hand-typed backend unit cases
        # additionally exercise 64-bit state, shared memory and split shuffles.
        for dtype, shared_name in ((F64, "double"), (I64, "long long")):
            with self.subTest(dtype=dtype.name):
                source = load_kernel(next((ROOT / f"tests/extended/ext_sum_{dtype.name}_129").glob("*.ir.json")))

                def rewrite(stmt):
                    nested = stmt.nested()
                    if nested:
                        stmt = stmt.with_bodies(tuple(tuple(rewrite(child) for child in body) for body in nested))
                    if isinstance(stmt, (Accum, WarpShuffle, TreeHalve)):
                        return replace(stmt, dtype=dtype)
                    if isinstance(stmt, Load) and stmt.input == "x":
                        return replace(stmt, dtype=dtype)
                    if isinstance(stmt, Smem):
                        return replace(stmt, dtype=shared_name)
                    return stmt

                kernel = replace(source.kernel, body=tuple(rewrite(stmt) for stmt in source.kernel.body))
                self.assemble(replace(source, kernel=kernel))


if __name__ == "__main__":
    unittest.main()
