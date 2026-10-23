#!/usr/bin/env python3
"""Trace the three saved f64/f32-accumulator fixtures numerically on CPU.

This is a deliberately restricted diagnostic for the known public-style sum
schedule, not another compiler IR or a general interpreter. NumPy expectations
are never overwritten. --runner compares our GPU output with the CPU trace.
"""

import argparse
import json
import sys
import time

import numpy as np

from test_extended import ROOT, check_runner
from compile import emit_ptx, load_kernel
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.kernel.ir import TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt import Accum, StridedLoop


def trace_saved_schedule(source, meta, operation):
    kernel = meta["kernels"][0]
    tile = source.kernel.body[0]
    loop = next(stmt for stmt in tile.body if isinstance(stmt, StridedLoop))
    accum = next(stmt for stmt in loop.body if isinstance(stmt, Accum))
    shuffle = next(stmt for stmt in tile.body if isinstance(stmt, WarpShuffle))
    tree = next(stmt for stmt in tile.body if isinstance(stmt, TreeHalve))
    arg = next(arg for arg in kernel["args"] if arg["name"] == "x")
    threads = kernel["block"][0]
    if not (accum.dtype.name == "f32" and accum.op.name == "add"
            and isinstance(loop.start, Var) and loop.start.name == tile.axes[-1].name
            and isinstance(loop.step, Literal) and loop.step.value == threads
            and loop.end is None and shuffle.length == 32
            and tree.length == threads // 32 and tree.inner is None
            and arg["dtype"] == "f64" and "seed" in arg):
        raise ValueError("Diagnostic only supports this specific f32 sum schedule")
    x = np.random.default_rng(arg["seed"]).standard_normal(arg["shape"])
    rows = x.reshape(-1, arg["shape"][-1])
    if kernel["grid"][0] != len(rows):
        raise ValueError("Expected one block per row")
    partial = np.zeros((len(rows), threads), np.float32)
    for lane in range(threads):
        for index in range(lane, rows.shape[1], threads):
            # Explicitly reproduce conversion and rounding in the typed IR.
            partial[:, lane] = partial[:, lane] + rows[:, index].astype(np.float32)
    distance = shuffle.length // 2
    while distance:
        partial = (partial + partial[:, np.arange(threads) ^ distance]).astype(np.float32)
        distance //= 2
    shared = partial[:, ::32].copy()
    half = tree.length // 2
    while half:
        shared[:, :half] = shared[:, :half] + shared[:, half:2 * half]
        half //= 2
    result = shared[:, 0].astype(np.float64)
    if operation == "mean":
        count = next(arg for arg in kernel["args"] if arg["name"] == "mean_keepdim_count")
        result = result / np.float64(count["data"][0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    results = []
    for operation in ("sum", "mean", "row_sum"):
        suffix = "3x129" if operation == "row_sum" else "129"
        folder = ROOT / "tests/extended" / f"ext_{operation}_f64_{suffix}"
        meta = json.loads((folder / "meta.json").read_text())
        source = load_kernel(next(folder.glob("*.ir.json")))
        traced = trace_saved_schedule(source, meta, operation)
        with np.load(folder / "expected.npz") as saved:
            expected = saved[meta["graph_outputs"][0]].ravel()
        row = {"case": folder.name, "input_dtype": "f64", "ir_accumulator_dtype": "f32",
               "numpy_expected": expected.tolist(), "cpu_ir_trace": traced.tolist(),
               "trace_matches_numpy_tolerance": bool(np.allclose(traced, expected, rtol=meta["rtol"], atol=meta["atol"]))}
        if args.runner:
            status, details = check_runner(folder, meta, {source.kernel.name: emit_ptx(source)})
            row.update(gpu_status=status, gpu_details=details)
            examples = details.get("examples", [])
            differences = int(np.count_nonzero(traced != expected))
            row["gpu_mismatches_match_cpu_ir_trace"] = bool(
                status == "wrong_result" and differences == details.get("mismatch_count")
                and len(examples) == differences
                and all(example["got"] == traced[example["index"]] for example in examples))
            time.sleep(1)
        results.append(row)
        (ROOT / "review/reduction-precision-diagnosis.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
    return int(any(row.get("gpu_mismatches_match_cpu_ir_trace", True) is False for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
