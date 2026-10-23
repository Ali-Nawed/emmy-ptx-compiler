#!/usr/bin/env python3
"""Explicit reduction inputs on unchanged, genuinely Emmy-generated boundary IR.

Expression expectations retain the independent NumPy calculation and tolerance.
Differences caused by the upstream schedule/state precision remain failures.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time

import numpy as np

from expressions_extended import numpy_reference
from ir_reference import DTYPES, save_reference
from test_extended import ROOT, check_runner, compare_outputs


def input_values(dtype, shape, pattern):
    x = np.zeros(shape, DTYPES[dtype])
    rows = x.reshape(-1, shape[-1])
    if pattern == "negative":
        x[...] = -(np.arange(math.prod(shape)).reshape(shape) % 7 + 1)
    elif pattern == "cancellation":
        magnitude = {"f16": 2048, "f32": 2 ** 24, "f64": 2 ** 40,
                     "i32": 2 ** 24, "i64": 2 ** 40}[dtype]
        for row in rows:
            # The first XOR step pairs lane 0 with lane 16. A narrow
            # accumulator can lose +1 before lane 1 cancels the large value.
            row[0], row[16], row[1] = magnitude, 1, -magnitude
    elif pattern == "large":
        x.fill(2 ** 30 + 1 if dtype.startswith("i") else 64)
    elif pattern != "zeros":
        raise ValueError(pattern)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    results = []
    output = ROOT / "tests/reduction-inputs"
    output.mkdir(parents=True, exist_ok=True)
    for dtype in DTYPES:
        cohort = "reduce_f32_w128" if dtype == "f32" else f"dtype_{dtype}"
        for operation in ("sum", "row_sum"):
            suffix = "3x257" if operation == "row_sum" else "257"
            original = ROOT / "tests/boundaries" / cohort / f"ext_{operation}_{dtype}_{suffix}"
            for pattern in ("zeros", "negative", "cancellation", "large"):
                name = f"{dtype}_{operation}_{pattern}"
                folder = output / name
                folder.mkdir(parents=True, exist_ok=True)
                meta = json.loads((original / "meta.json").read_text())
                argument = next(arg for arg in meta["kernels"][0]["args"] if arg["name"] == "x")
                x = input_values(dtype, argument["shape"], pattern)
                for kernel in meta["kernels"]:
                    for arg in kernel["args"]:
                        if arg["name"] == "x":
                            arg.pop("seed", None)
                            arg["data"] = x.ravel().tolist()
                meta.update(id=name, input_pattern=pattern)
                (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
                sources = {}
                for path in original.glob("*.ir.*"):
                    shutil.copyfile(path, folder / path.name)
                    sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                shutil.copyfile(original / "expression.py", folder / "expression.py")
                (folder / "input-provenance.json").write_text(json.dumps({
                    "original_fixture": str(original.relative_to(ROOT)), "unchanged_ir_sha256": sources,
                    "pattern": pattern, "dtype": dtype, "shape": list(x.shape)}, indent=2) + "\n")
                expected = numpy_reference(x, operation=operation, dtype=dtype)
                np.savez(folder / "expected.npz", **dict(zip(meta["graph_outputs"], expected, strict=True)))
                ir_reference = save_reference(folder)
                status, details = compare_outputs(ir_reference, dict(zip(meta["graph_outputs"], expected, strict=True)), meta)
                row = {"id": name, "dtype": dtype, "operation": operation, "pattern": pattern,
                       "ir_vs_expression": {"status": status, "details": details}, "status": "compiled_not_executed"}
                ptx_by_entry = {}
                for kernel in meta["kernels"]:
                    ir = folder / f"{kernel['entry']}.ir.json"
                    compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)],
                                              capture_output=True, text=True, check=True, timeout=60)
                    path = folder / f"{kernel['entry']}.ours.ptx"
                    path.write_text(compiled.stdout)
                    subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                                   capture_output=True, text=True, check=True, timeout=35)
                    ptx_by_entry[kernel["entry"]] = compiled.stdout
                if args.runner:
                    row["status"], row["details"] = check_runner(folder, meta, ptx_by_entry, ir_reference)
                    time.sleep(0.2)
                results.append(row)
                report = {"cases": results, "statuses": dict(Counter(row["status"] for row in results))}
                (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    return int(any(row["status"] not in ("pass", "compiled_not_executed")
                   or row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass"
                   for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
