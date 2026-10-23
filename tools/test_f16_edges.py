#!/usr/bin/env python3
"""Exercise f16 numerical edges using fresh-Emmy fixtures with explicit inputs.

Kernel IR is copied unchanged from tests/extended; only input data and NumPy
expectations vary. Runner use is opt-in and only uploads our compiler's PTX.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from expressions_extended import numpy_reference
from test_extended import ROOT, check_runner


def edge_inputs(operation, shape):
    count = math.prod(shape)
    if operation in ("sum", "mean", "row_sum"):
        x = np.zeros(shape, np.float16)
        rows = x.reshape(-1, shape[-1])
        # At the first XOR step lane 0 receives lane 16. A half accumulator
        # loses the +1 next to 2048; the IR's f32 accumulator must retain it.
        for index, row in enumerate(rows):
            row[0], row[16], row[1] = 2048, (1, 0.5, 2)[index % 3], -2048
        return [x]
    if operation == "scale_shift":
        # Find finite half values where rounding after multiply differs from
        # retaining f32 until the last store. This catches missing intermediates.
        values = np.arange(0x7C00, dtype=np.uint16).view(np.float16)
        values = values[values < 8]
        stepwise = values * np.float16(-1.25) + np.float16(0.5)
        deferred = (values.astype(np.float32) * np.float32(-1.25) + np.float32(0.5)).astype(np.float16)
        candidates = values[stepwise != deferred]
        if not len(candidates):
            raise ValueError("Could not construct an intermediate-rounding test")
        return [np.resize(candidates, count).reshape(shape)]
    tiny = np.nextafter(np.float16(0), np.float16(1))
    values = np.array([tiny, -tiny, np.finfo(np.float16).tiny, -np.finfo(np.float16).tiny,
                       0, -0.0, 0.33325, -0.33325, 1.0009765625, -1.0009765625,
                       2048, -2048, 65504, -65504], dtype=np.float16)
    x = np.resize(values, count).reshape(shape)
    y = np.resize(np.array([1, 0.5, -1, -0.5], np.float16), count).reshape(shape)
    return [x, y]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    operations = ("add", "subtract", "multiply", "scale_shift", "relu_mul", "sum", "mean", "row_sum")
    results = []
    for operation in operations:
        suffix = "3x129" if operation == "row_sum" else "129"
        original = ROOT / "tests/extended" / f"ext_{operation}_f16_{suffix}"
        folder = ROOT / "tests/f16-edges" / operation
        folder.mkdir(parents=True, exist_ok=True)
        meta = json.loads((original / "meta.json").read_text())
        shape = (3, 129) if operation == "row_sum" else (129,)
        arrays = edge_inputs(operation, shape)
        inputs = dict(zip(meta["graph_inputs"], arrays, strict=True))
        meta.update(id=f"f16_edge_{operation}", rtol=0, atol=0,
                    input_pattern="Explicit f16 rounding/subnormal/cancellation values")
        outputs = {}
        for kernel in meta["kernels"]:
            for argument in kernel["args"]:
                if argument["name"] in inputs:
                    argument.pop("seed", None)
                    argument["data"] = inputs[argument["name"]].ravel().tolist()
                if argument.get("output"):
                    outputs[argument["name"]] = argument["shape"]
        expected = numpy_reference(*arrays, operation=operation, dtype="f16")
        np.savez(folder / "expected.npz", **{
            name: value.reshape(outputs[name])
            for name, value in zip(meta["graph_outputs"], expected, strict=True)
        })
        (folder / "meta.json").write_text(json.dumps(meta, indent=2))
        for path in original.glob("*.ir.*"):
            shutil.copyfile(path, folder / path.name)
        shutil.copyfile(original / "expression.py", folder / "expression.py")
        ptx_by_entry = {}
        for kernel in meta["kernels"]:
            compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"),
                                       str(folder / f"{kernel['entry']}.ir.json")],
                                      capture_output=True, text=True, check=True, timeout=60)
            ptx = folder / f"{kernel['entry']}.ours.ptx"
            ptx.write_text(compiled.stdout)
            ptx_by_entry[kernel["entry"]] = compiled.stdout
            subprocess.run(["ptxas", "-arch=sm_89", str(ptx), "-o", str(ptx.with_suffix(".cubin"))],
                           capture_output=True, text=True, check=True, timeout=35)
        row = {"operation": operation, "status": "compiled_not_executed"}
        if args.runner:
            row["status"], row["details"] = check_runner(folder, meta, ptx_by_entry)
            time.sleep(1)
        results.append(row)
        report = ROOT / "review/f16-edge-results.json"
        report.write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    return int(any(row["status"] not in ("pass", "compiled_not_executed") for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
