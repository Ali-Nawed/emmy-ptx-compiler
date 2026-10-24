#!/usr/bin/env python3
"""Recheck our optimized compiler without overwriting baseline test reports/PTX."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from test_extended import ROOT, check_runner, compare_outputs
from ir_reference import simulate
from compile import emit_ptx, load_kernel

PUBLIC = ("p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
          "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum")


def groups():
    return {
        "mvp": [ROOT / "problems" / name for name in PUBLIC],
        "extended": sorted((ROOT / "tests/extended").glob("ext_*")),
        "boundaries": sorted(path.parent for path in (ROOT / "tests/boundaries").glob("*/ext_*/meta.json")),
        **{name: sorted(path.parent for path in (ROOT / "tests" / folder).glob("*/meta.json"))
           for name, folder in (("inputs", "reduction-inputs"), ("half_edges", "f16-edges"),
                                ("integer_bits", "i64-output-bits"), ("half_large", "extended-f16-large"),
                                ("vectors", "vector-store-cases"), ("wide_vectors", "vector-store-wide"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--groups", nargs="+", choices=list(groups()))
    parser.add_argument("--report-name", default="validation.json")
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name:
        parser.error("--report-name must be a filename")
    output = ROOT / "review/emitter-optimizations"
    output.mkdir(parents=True, exist_ok=True)
    sizes = {"mvp": 8, "extended": 80, "boundaries": 317, "inputs": 40,
             "half_edges": 8, "integer_bits": 8, "half_large": 4, "vectors": 10, "wide_vectors": 5}
    selected = [(group, folder) for group, folders in groups().items()
                if args.groups is None or group in args.groups for folder in folders]
    for group in args.groups or sizes:
        if sum(name == group for name, _ in selected) != sizes[group]:
            parser.error(f"Missing fixtures in {group}; generate them first")
    compiler_hash = hashlib.sha256((ROOT / "compile.py").read_bytes()).hexdigest()
    rows = []
    for group, folder in selected:
        row = {"group": group, "fixture": str(folder.relative_to(ROOT)), "status": "local_failure"}
        try:
            meta = json.loads((folder / "meta.json").read_text())
            ir_reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                cpu_status, cpu_details = compare_outputs(ir_reference, expected, meta)
            row["ir_vs_expression"] = {"status": cpu_status, "details": cpu_details}
            destination = output / "ptx" / group / str(folder.relative_to(ROOT)).replace("/", "__")
            destination.mkdir(parents=True, exist_ok=True)
            compiled = {}
            for kernel in meta["kernels"]:
                ptx = emit_ptx(load_kernel(folder / f"{kernel['entry']}.ir.json"))
                path = destination / f"{kernel['entry']}.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               capture_output=True, text=True, check=True, timeout=35)
                compiled[kernel["entry"]] = ptx
            row["status"] = "compiled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, ir_reference)
                row["matches_cpu_prediction"] = row["status"] == cpu_status
                time.sleep(0.1)
        except Exception as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n" + (exc.stderr or "")
        rows.append(row)
        summary = {"cases": len(rows), "groups": dict(Counter(row["group"] for row in rows)),
                   "statuses": dict(Counter(row["status"] for row in rows)),
                   "gpu_vs_ir": dict(Counter(row.get("details", {}).get("comparisons", {}).get("kernel_ir", {})
                                             .get("status", "not_executed") for row in rows)),
                   "unexpected_predictions": sum(row.get("matches_cpu_prediction") is False for row in rows)}
        (output / args.report_name).write_text(json.dumps({"compiler_sha256": compiler_hash,
            "runner": args.runner, "summary": summary, "cases": rows}, indent=2) + "\n")
        if len(rows) % 10 == 0 or len(rows) == len(selected) or "error" in row:
            print(json.dumps(summary), flush=True)
    return int(any(row["status"] not in ("pass", "compiled_not_executed")
                   or row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass"
                   for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
