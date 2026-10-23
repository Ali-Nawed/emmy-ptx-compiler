#!/usr/bin/env python3
"""Generate synthetic variants of public Kernel IR and test our compiler.

This changes typed Emmy nodes, not serialized body strings. These fixtures test
specific assumptions; they are not predictions of Emmy's hidden schedules.
Without --runner, only compilation and local ptxas checks are performed.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import load_kernel
from emmy.compiler.ir.stmt import StridedLoop


def generate(case, template, shape, constants=None, pattern=False):
    folder = ROOT / "tests/generated" / case
    folder.mkdir(parents=True, exist_ok=True)
    original = ROOT / "problems" / template
    source = load_kernel(next(original.glob("*.ir.json")))
    meta = json.loads((original / "meta.json").read_text())
    launch = meta["kernels"][0]
    tile = source.kernel.body[0]
    axes = list(tile.axes)
    if template.startswith("p"):
        unroll = 2 if template.startswith("p00") else 4
        assert shape[0] % unroll == 0
        axes[0] = replace(axes[0], extent=shape[0] // unroll)
        tile = replace(tile, axes=tuple(axes))
        launch["grid"] = [math.ceil(tile.n_elements / launch["block"][0]), 1, 1]
    else:
        if len(shape) == 2:
            axes[0] = replace(axes[0], extent=shape[0])
        statements = tuple(
            replace(stmt, axis=replace(stmt.axis, extent=shape[-1]))
            if isinstance(stmt, StridedLoop) else stmt for stmt in tile.body
        )
        tile = replace(tile, axes=tuple(axes), body=statements)
        launch["grid"] = [tile.n_elements // launch["block"][0], 1, 1]
    source.graph.nodes[source.node_name].op = replace(source.kernel, body=(tile,))
    arrays = {}
    shapes = {}
    for position, arg in enumerate(launch["args"]):
        if arg.get("output"):
            arg["shape"] = (shape if template.startswith("p") else
                            [shape[0]] if len(shape) == 2 else [1])
        elif "data" in arg:
            if constants is not None:
                arg["data"] = [constants[arg["name"]]]
        else:
            arg["shape"] = shape
            arg["seed"] = 17 + position
            array = np.random.default_rng(arg["seed"]).standard_normal(shape).astype(np.float32)
            if pattern:
                array = np.resize(np.array([-3, -0.0, 0.0, 0.125, 2, -0.5], np.float32), math.prod(shape)).reshape(shape)
                arg.pop("seed", None)
                arg["data"] = array.ravel().tolist()
            arrays[arg["name"]] = array
        shapes[arg["name"]] = arg["shape"]
    # Scalar constants must remain consistent between graph and launch metadata.
    raw = source.graph.to_dict()
    for name, node in raw["nodes"].items():
        for tensor in node["outputs"]:
            if tensor["name"] in shapes:
                tensor["shape"] = shapes[tensor["name"]]
        if constants is not None and name in constants:
            node["op_fields"]["value"] = constants[name]
    if template.startswith("p00"):
        answer = arrays["x0"] + arrays["x1"]
    elif template.startswith("p02"):
        answer = arrays["x"] * np.float32(constants["mul_c1"]) + np.float32(constants["add_c1"])
    elif template.startswith("p03"):
        answer = np.maximum(arrays["x"], np.float32(0)) * arrays["y"]
    else:
        x = arrays["x"].astype(np.float64)
        answer = np.asarray(x.sum(axis=-1) if len(shape) == 2 else x.sum(), np.float32)
    output = next(arg for arg in launch["args"] if arg.get("output"))
    np.savez(folder / "expected.npz", **{output["name"]: answer.reshape(output["shape"])})
    meta["id"], meta["code"] = case, f"Synthetic typed-IR variant of {template}, shape={shape}"
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    ir = folder / f"{source.kernel.name}.ir.json"
    ir.write_text(json.dumps(raw, indent=2, default=str))
    return folder, ir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true", help="also test our generated PTX on the hosted runner")
    args = parser.parse_args()
    cases = [(f"add_{n}", "p00_scalar_add", [n], None, False) for n in (2, 130, 514, 1026)]
    cases += [(f"scale_{n}", "p02_scale_shift", [n], {"mul_c1": -1.25, "add_c1": 0.3}, False)
              for n in (12, 1028)]
    cases += [(f"relu_{n}", "p03_relu_mul", [n], None, True) for n in (20, 1036)]
    cases += [(f"sum_{n}", "r01_array_sum", [n], None, False) for n in (1, 31, 127, 129, 1003)]
    cases += [("rows_3x129", "r04_row_sum", [3, 129], None, False),
              ("rows_7x513", "r04_row_sum", [7, 513], None, False)]
    results = []
    for case in cases:
        folder, ir = generate(*case)
        compiled = subprocess.run([sys.executable, str(ROOT / "compile.py"), str(ir)], capture_output=True, text=True)
        ptx = folder / "ours.ptx"
        ptx.write_text(compiled.stdout)
        row = {"case": case[0], "compile_ok": compiled.returncode == 0}
        if compiled.returncode:
            row["error"] = compiled.stderr
        else:
            assembled = subprocess.run(["ptxas", "-arch=sm_89", str(ptx), "-o", str(folder / "ours.cubin")], capture_output=True, text=True)
            row["assemble_ok"] = assembled.returncode == 0
            if assembled.returncode:
                row["error"] = assembled.stderr
            elif args.runner:
                # Always invoke OUR compiler; never submit external/reference PTX.
                checked = subprocess.run([sys.executable, str(ROOT / "submit.py"), str(folder), "--compiler", str(ROOT / "compile.py")], capture_output=True, text=True)
                row.update(correct=checked.returncode == 0, output=checked.stdout + checked.stderr)
                # One case at a time; this is a courtesy delay, not a published quota.
                time.sleep(1)
        results.append(row)
        print(json.dumps(row), flush=True)
    report = ROOT / "review/custom-test-results.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, indent=2))
    return int(any(not r.get("compile_ok") or not r.get("assemble_ok")
                   or (args.runner and not r.get("correct")) for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
