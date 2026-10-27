#!/usr/bin/env python3
"""Snapshot and validate our native exp/log change on genuine Emmy fixtures."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from test_extended import ROOT, check_runner, compare_outputs
from compile import emit_ptx, load_kernel
from emmy_fixtures import verify
from ir_reference import simulate
from compare_emmy_ptx import assemble, ptx_stats


OUTPUT = ROOT / "review/native-math"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases():
    public = ("p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
              "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum",
              "f01_rmsnorm", "f03_l2norm", "f02_row_softmax")
    folders = [ROOT / "problems" / name for name in public]
    for family in ("operations", "normalization", "pytorch-only"):
        folders += sorted(path.parent for path in (ROOT / "tests" / family).glob("*/meta.json"))
    return folders


def capture():
    if (OUTPUT / "before.json").exists():
        raise ValueError("Baseline already exists; preserve it instead of overwriting it")
    records = []
    for folder in cases():
        meta = json.loads((folder / "meta.json").read_text())
        destination = OUTPUT / "before" / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        kernels = {}
        for descriptor in meta["kernels"]:
            ir = folder / f"{descriptor['entry']}.ir.json"
            ptx_path = destination / f"{descriptor['entry']}.ours.ptx"
            ptx_path.write_text(emit_ptx(load_kernel(ir)))
            kernels[descriptor["entry"]] = {"ir_sha256": sha(ir), "ptx": str(ptx_path.relative_to(ROOT))}
        records.append({"fixture": str(folder.relative_to(ROOT)), "meta_sha256": sha(folder/"meta.json"), "kernels": kernels})
    report = {"source_sha256": {name: sha(ROOT/name) for name in ("compile.py", "ptx_math.py")}, "cases": records}
    (OUTPUT / "before.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"Captured {len(records)} unchanged-fixture baselines from our compiler.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-before", action="store_true")
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--report-name", default="validation.json")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.capture_before:
        capture()
        return 0
    if Path(args.report_name).name != args.report_name:
        parser.error("report name must be a simple filename")
    baseline = json.loads((OUTPUT / "before.json").read_text())
    rows = []
    for record in baseline["cases"]:
        folder = ROOT / record["fixture"]
        if args.only and folder.name not in args.only:
            continue
        row = {"fixture": record["fixture"], "status": "failure"}
        try:
            if sha(folder/"meta.json") != record["meta_sha256"]:
                raise ValueError("Fixture metadata changed after baseline capture")
            if (folder/"emmy-proof.json").exists():
                verify(folder)
            meta = json.loads((folder/"meta.json").read_text())
            compiled, statistics = {}, {}
            for descriptor in meta["kernels"]:
                entry = descriptor["entry"]
                ir = folder / f"{entry}.ir.json"
                if sha(ir) != record["kernels"][entry]["ir_sha256"]:
                    raise ValueError("IR changed after baseline capture")
                source = emit_ptx(load_kernel(ir))
                destination = OUTPUT / "after" / folder.name
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / f"{entry}.ours.ptx"
                path.write_text(source)
                statistics[entry] = {**ptx_stats(source), **assemble(path)}
                compiled[entry] = source
            row["assembly"] = statistics
            row["status"] = "assembled_not_executed"
            if args.runner:
                reference = simulate(folder)
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
                time.sleep(.3)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
        rows.append(row)
        report = {"source_sha256": {name: sha(ROOT/name) for name in ("compile.py", "ptx_math.py")},
                  "runner": args.runner, "cases": rows, "statuses": dict(Counter(row["status"] for row in rows))}
        (OUTPUT/args.report_name).write_text(json.dumps(report, indent=2)+"\n")
        print(f"{row['status']:26} {folder.name}", flush=True)
        if "error" in row:
            print(row["error"][-800:], flush=True)
    # Known expression-level differences are reported, never converted to passes.
    return int(any(row["status"] not in ("pass", "wrong_result", "assembled_not_executed") or
                   row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass"
                   for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
