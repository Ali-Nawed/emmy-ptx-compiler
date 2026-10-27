#!/usr/bin/env python3
"""Compare saved OUR before/after PTX after correctness validation only."""

import hashlib
import json
from statistics import median
import time

import numpy as np

from test_extended import ROOT, compare_outputs
from compare_emmy_ptx import assemble, ptx_stats
from ir_reference import simulate
from submit import RUNNER, post

NAMES = ("range_exp_f32", "range_log_f32", "f02_row_softmax",
         "ops_softmax_f16_3x129_w128", "ops_softmax_f32_3x257_w256", "range_exp_f64")


def main():
    output = ROOT/"review/native-math"
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py")}
    validation = json.loads((output/"validation.json").read_text())
    edges = json.loads((output/"edges/validation.json").read_text())
    if (validation["source_sha256"] != hashes or len(validation["cases"]) != 169
            or any(row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status") != "pass" for row in validation["cases"])
            or edges["source_sha256"] != hashes or len(edges["cases"]) != 6
            or any(row["status"] != "pass" for row in edges["cases"])):
        raise ValueError("Complete both current-source correctness suites before timing")
    baseline = json.loads((output/"before.json").read_text())
    rows = []
    for index, name in enumerate(NAMES):
        record = next(row for row in baseline["cases"] if row["fixture"].endswith("/"+name))
        fixture = ROOT/record["fixture"]
        meta = json.loads((fixture/"meta.json").read_text())
        if len(meta["kernels"]) != 1:
            raise ValueError("Timing cases must have exactly one kernel")
        descriptor = meta["kernels"][0]
        entry = descriptor["entry"]
        paths = {"before": ROOT/record["kernels"][entry]["ptx"],
                 "after": output/"after"/name/f"{entry}.ours.ptx"}
        statistics, compiled = {}, {}
        for variant, path in paths.items():
            source = path.read_text()
            statistics[variant] = {**ptx_stats(source), **assemble(path)}
            assembled = post(RUNNER, "/compile", {"ptx": source})
            if not assembled.get("ok"):
                raise RuntimeError(assembled)
            compiled[variant] = assembled["cubin_id"]
        ir_reference = simulate(fixture)
        samples = []
        for repeat in range(3):
            order = ("before", "after") if (index+repeat)%2 == 0 else ("after", "before")
            for variant in order:
                request = {"cubin_id": compiled[variant], "entry": entry, "grid": descriptor["grid"],
                           "block": descriptor["block"], "smem_bytes": descriptor["smem_bytes"],
                           "zero_outputs": descriptor["zero_outputs"], "args": descriptor["args"],
                           "warmup": 10, "iters": 100, "max_return": 65536}
                for attempt in range(30):
                    result = post(RUNNER, "/run", request)
                    if result.get("stage") != "busy":
                        break
                    time.sleep(2)
                if not result.get("ok") or any(o.get("truncated") for o in result["outputs"].values()):
                    raise RuntimeError(result)
                produced = {name: value["values"] for name, value in result["outputs"].items()}
                if compare_outputs(produced, ir_reference, meta)[0] != "pass":
                    raise ValueError("Timing output differs from the typed IR")
                with np.load(fixture/"expected.npz") as expected:
                    if compare_outputs(produced, expected, meta)[0] != "pass":
                        raise ValueError("Timing output differs from the expression")
                samples.append({"variant": variant, "repeat": repeat, "median_us": result["median_us"]})
        timings = {variant: median(sample["median_us"] for sample in samples if sample["variant"] == variant) for variant in paths}
        row = {"case": name, "assembly": statistics, "median_us": timings,
               "before_over_after": timings["before"]/timings["after"], "samples": samples,
               "same_ptx": paths["before"].read_text() == paths["after"].read_text()}
        rows.append(row)
        (output/"timings.json").write_text(json.dumps({"source_sha256": hashes,
            "warmup": 10, "iters": 100, "paired_rounds": 3, "clocks_locked": False,
            "only_our_compiler_uploaded": True, "cases": rows}, indent=2)+"\n")
        print(json.dumps({"case": name, "median_us": timings, "before_over_after": row["before_over_after"],
                          "non_nop_sass": {variant: stats["sass_non_nop_count"] for variant, stats in statistics.items()}}), flush=True)


if __name__ == "__main__":
    main()
