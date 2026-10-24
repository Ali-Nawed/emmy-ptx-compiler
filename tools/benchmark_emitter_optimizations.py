#!/usr/bin/env python3
"""Hosted before/after samples of OUR compiler only, after correctness checks.

The saved baseline is explicitly ours.ptx, never the Emmy/nvcc variants.
Uncontrolled GPU clocks and small kernels make these indicative samples.
"""

import hashlib
import json
from statistics import median
import time

import numpy as np

from test_extended import ROOT, compare_outputs
from compile import emit_ptx, load_kernel
from ir_reference import simulate
from submit import RUNNER, post

CASES = ("p01_vector_add", "p02_scale_shift", "p03_relu_mul", "r04_row_sum",
         "row_sum_f32_17x1025", "r01_array_sum")


def main():
    output = ROOT / "review/emitter-optimizations"
    validation = json.loads((output / "validation.json").read_text())
    current_hash = hashlib.sha256((ROOT / "compile.py").read_bytes()).hexdigest()
    count = validation["summary"]["cases"]
    if (validation["compiler_sha256"] != current_hash or count < 475
            or validation["summary"]["gpu_vs_ir"] != {"pass": count}
            or validation["summary"]["unexpected_predictions"]):
        raise ValueError("Complete the full correctness matrix with this compiler first")
    baseline_root = ROOT / "review/emmy-ptx-comparison"
    baseline = json.loads((baseline_root / "results.json").read_text())
    rows = []
    for index, name in enumerate(CASES):
        original = next(row for row in baseline["cases"] if row["case"] == name)
        fixture = ROOT / original["fixture"]
        meta = json.loads((fixture / "meta.json").read_text())
        if len(meta["kernels"]) != 1:
            raise ValueError("Benchmark supports one-kernel cases only")
        kernel = meta["kernels"][0]
        variants = {"before": (baseline_root / name / "ours.ptx").read_text(),
                    "after": emit_ptx(load_kernel(fixture / f"{kernel['entry']}.ir.json"))}
        compiled = {}
        for variant, source in variants.items():
            result = post(RUNNER, "/compile", {"ptx": source})
            if not result.get("ok"):
                raise RuntimeError(result)
            compiled[variant] = result["cubin_id"]
        ir_reference = simulate(fixture)
        samples = []
        for repeat in range(3):
            order = ("before", "after") if (repeat + index) % 2 == 0 else ("after", "before")
            for variant in order:
                need = sum(np.prod(arg["shape"]).item() for arg in kernel["args"] if arg.get("output"))
                for attempt in range(60):
                    result = post(RUNNER, "/run", {"cubin_id": compiled[variant], "entry": kernel["entry"],
                        "grid": kernel["grid"], "block": kernel["block"], "smem_bytes": kernel["smem_bytes"],
                        "zero_outputs": kernel["zero_outputs"], "args": kernel["args"],
                        "warmup": 10, "iters": 100, "max_return": max(4096, need)})
                    if result.get("stage") != "busy":
                        break
                    time.sleep(min(2 + attempt, 10))
                if not result.get("ok") or any(value.get("truncated") for value in result["outputs"].values()):
                    raise RuntimeError(result)
                produced = {name: value["values"] for name, value in result["outputs"].items()}
                if compare_outputs(produced, ir_reference, meta)[0] != "pass":
                    raise ValueError("Benchmark output differs from the IR")
                with np.load(fixture / "expected.npz") as expected:
                    if compare_outputs(produced, expected, meta)[0] != "pass":
                        raise ValueError("Benchmark output differs from the expression")
                samples.append({"variant": variant, "repeat": repeat, "median_us": result["median_us"]})
        timings = {variant: median(sample["median_us"] for sample in samples if sample["variant"] == variant)
                   for variant in variants}
        row = {"case": name, "fixture": original["fixture"], "median_us": timings,
               "after_over_before": timings["after"] / timings["before"], "samples": samples,
               "ptx_sha256": {variant: hashlib.sha256(source.encode()).hexdigest() for variant, source in variants.items()}}
        rows.append(row)
        (output / "timing-results.json").write_text(json.dumps({"compiler_sha256": current_hash,
            "warmup": 10, "iters": 100, "paired_rounds": 3, "reference_uploaded": False,
            "clocks_locked": False, "cases": rows}, indent=2) + "\n")
        print(json.dumps({"case": name, "median_us": timings, "after_over_before": row["after_over_before"]}), flush=True)


if __name__ == "__main__":
    main()
