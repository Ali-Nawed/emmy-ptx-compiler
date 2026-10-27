#!/usr/bin/env python3
"""Verify wide integer arithmetic before the runner serializes its output.

Fresh PyTorch -> Emmy fixtures extract low/high words on the GPU. Returned
values fit in 16 bits, so the runner's apparent float serialization cannot
erase the low bits we are testing. Original full-output failures are retained.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from generate import trace
from emmy_fixtures import package, verify
from test_extended import ROOT, check_runner, save_expression, validate_fixture
from test_numeric_edges import inputs
from compile import load_kernel
from emmy.compiler.ir.stmt import Assign


def reference(*arrays, operation, shift, mask):
    x = arrays[0]
    if operation == "add":
        result = x + arrays[1]
    elif operation == "subtract":
        result = x - arrays[1]
    elif operation == "multiply":
        result = x * arrays[1]
    else:
        result = x * np.int64(2) - np.int64(3)
    return [np.bitwise_and(np.right_shift(result, np.int64(shift)), np.int64(mask))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="reuse the saved genuine Emmy fixtures")
    args = parser.parse_args()
    results = []
    emmy = str(Path(sys.executable).parent / "emmy")
    for operation in ("add", "subtract", "multiply", "scale_shift"):
        expression = {"add": "x + y", "subtract": "x - y", "multiply": "x * y",
                      "scale_shift": "x * 2 - 3"}[operation]
        for shift, mask in ((0, 255), (32, 65535)):
            name = f"i64_{operation}_bits_{shift}"
            folder = ROOT / "tests/i64-output-bits" / name
            constructor = "torch.randint(-8, 9, (129,), dtype=torch.int64)"
            setup = f"x = {constructor}; "
            if operation != "scale_shift":
                setup += f"y = {constructor}; "
            # Emmy traces the final expression. Keep arithmetic inside it;
            # computing z in the setup would make z a precomputed input buffer.
            code = setup + f"(({expression}) >> {shift}) & {mask}"
            spec = dict(id=name, code=code, dtype="i64", shape=[129], rtol=0, atol=0,
                        ref=partial(reference, operation=operation, shift=shift, mask=mask))
            if not args.reuse:
                with tempfile.TemporaryDirectory(prefix="emmy-i64-bits-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(emmy, spec, dump, "sm_89")
                    package(spec, dump, folder)
            else:
                verify(folder)
            save_expression(folder, code)
            meta = json.loads((folder / "meta.json").read_text())
            arrays = inputs("i64", operation)
            expected = spec["ref"](*arrays)[0]
            env = {"x": torch.from_numpy(arrays[0])}
            if len(arrays) == 2:
                env["y"] = torch.from_numpy(arrays[1])
            eager = eval(f"(({expression}) >> {shift}) & {mask}", {"__builtins__": {}}, env).numpy()
            if not np.array_equal(eager, expected):
                raise ValueError("Bit-extraction reference disagrees with CPU PyTorch")
            named = dict(zip(meta["graph_inputs"], arrays, strict=True))
            for kernel in meta["kernels"]:
                for argument in kernel["args"]:
                    if argument["name"] in named:
                        argument.pop("seed", None)
                        argument["data"] = named[argument["name"]].tolist()
            meta["input_pattern"] = "Exactly representable inputs; extract native 64-bit result bits before host serialization"
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
            np.savez(folder / "expected.npz", **{meta["graph_outputs"][0]: expected})
            validate_fixture(folder, spec, meta)
            ptx_by_entry = {}
            operations = []
            for kernel in meta["kernels"]:
                ir = folder / f"{kernel['entry']}.ir.json"
                source = load_kernel(ir)
                operations.extend(stmt.op.name for stmt in source.kernel.body.iter() if isinstance(stmt, Assign))
                compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                          capture_output=True, text=True, check=True, timeout=60)
                ptx = folder / f"{kernel['entry']}.ours.ptx"
                ptx.write_text(compiled.stdout)
                subprocess.run(["ptxas", "-arch=sm_89", str(ptx), "-o", str(ptx.with_suffix(".cubin"))],
                               capture_output=True, text=True, check=True, timeout=35)
                ptx_by_entry[kernel["entry"]] = compiled.stdout
            row = {"case": name, "ir_operations": sorted(set(operations)), "status": "compiled_not_executed"}
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, ptx_by_entry)
                time.sleep(1)
            results.append(row)
            (ROOT / "review/i64-output-bit-results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
    return int(any(row["status"] not in ("pass", "compiled_not_executed") for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
