#!/usr/bin/env python3
"""Strict wide-number/overflow tests using unchanged fresh-Emmy pointwise IR."""

import argparse
import json
import shutil
import subprocess
import sys
import time

import numpy as np

from expressions_extended import numpy_reference
from test_extended import ROOT, check_runner


def inputs(dtype, operation):
    if dtype == "f64":
        x = np.array([1 + 2**-40, -1 - 2**-40, 2**40 + .25, -2**40 - .25,
                      np.nextafter(0., 1.), -np.nextafter(0., 1.), 0., -0.], np.float64)
        y = np.array([0., 2**-40, 3., -.5, 3., -.5, 1., -1.], np.float64)
    elif dtype == "i32":
        bits, dt = 32, np.int32
        low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        sentinel = 2**24 + 1
        x = np.array([low, high, sentinel, -sentinel, 0, -1, 1, sentinel], dt)
        y = np.array([-1, 1, 3, -3, 7, -7, 0, 0], dt)
    else:
        # The runner's explicit-input path rejects INT64_MAX (apparently after
        # float conversion). Use exactly representable inputs that create low
        # bits above 2**53 and wrapping overflow INSIDE the generated GPU code.
        x = np.array([2**30 + 1, 2**60, -2**60, 2**62, -2**62, 0, -1, 1], np.int64)
        y = np.array([2**30 + 3, 3, -3, 2**62, -2**62, 7, -7, 0], np.int64)
        assert all(int(float(value)) == int(value) for value in (*x, *y))
    arrays = [np.resize(x, 129), np.resize(y, 129)]
    return arrays[:1] if operation == "scale_shift" else arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--dtypes", nargs="+", choices=("f64", "i32", "i64"), default=["f64", "i32", "i64"])
    parser.add_argument("--report", default="numeric-edge-results.json")
    args = parser.parse_args()
    results = []
    for dtype in args.dtypes:
        for operation in ("add", "subtract", "multiply", "relu_mul", "scale_shift"):
            original = ROOT / "tests/extended" / f"ext_{operation}_{dtype}_129"
            folder = ROOT / "tests/numeric-edges" / f"{operation}_{dtype}"
            folder.mkdir(parents=True, exist_ok=True)
            meta = json.loads((original / "meta.json").read_text())
            arrays = inputs(dtype, operation)
            named_inputs = dict(zip(meta["graph_inputs"], arrays, strict=True))
            meta.update(id=f"edge_{operation}_{dtype}", rtol=0, atol=0,
                        input_pattern="Exact wide integer, wrapping overflow, or f64 low-bit/subnormal values")
            outputs = {}
            for kernel in meta["kernels"]:
                for argument in kernel["args"]:
                    if argument["name"] in named_inputs:
                        argument.pop("seed", None)
                        argument["data"] = named_inputs[argument["name"]].tolist()
                    if argument.get("output"):
                        outputs[argument["name"]] = argument["shape"]
            expected = numpy_reference(*arrays, operation=operation, dtype=dtype)
            np.savez(folder / "expected.npz", **{
                name: value.reshape(outputs[name])
                for name, value in zip(meta["graph_outputs"], expected, strict=True)
            })
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
            for source in original.glob("*.ir.*"):
                shutil.copyfile(source, folder / source.name)
            shutil.copyfile(original / "expression.py", folder / "expression.py")
            ptx_by_entry = {}
            for kernel in meta["kernels"]:
                compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"),
                                           str(folder / f"{kernel['entry']}.ir.json")],
                                          capture_output=True, text=True, check=True, timeout=60)
                path = folder / f"{kernel['entry']}.ours.ptx"
                path.write_text(compiled.stdout)
                ptx_by_entry[kernel["entry"]] = compiled.stdout
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               capture_output=True, text=True, check=True, timeout=35)
            row = {"dtype": dtype, "operation": operation, "status": "compiled_not_executed"}
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, ptx_by_entry)
                time.sleep(1)
            results.append(row)
            (ROOT / "review" / args.report).write_text(json.dumps(results, indent=2))
            print(json.dumps(row), flush=True)
    return int(any(row["status"] not in ("pass", "compiled_not_executed") for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
