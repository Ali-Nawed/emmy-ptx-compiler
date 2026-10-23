#!/usr/bin/env python3
"""Generate boundary/schedule fixtures locally, then optionally test our PTX.

Generation cohorts can run concurrently. Hosted runner cohorts always execute
serially and upload only the PTX produced by compile.py.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BOUNDARIES = (31, 32, 33, 127, 128, 255, 256, 257, 1025)
COHORTS = [
    (f"reduce_f32_w{work}", ["--dtypes", "f32", "--ops", "sum", "row_sum",
      "--sizes", *map(str, BOUNDARIES), "--rows", "1", "3", "17", "--work", str(work)])
    for work in (32, 64, 128, 256)
] + [
    (f"dtype_{dtype}", ["--dtypes", dtype, "--sizes", "31", "33", "257", "1025",
      "--rows", "3", "--work", "128"])
    for dtype in ("f16", "f64", "i32", "i64")
] + [
    ("pointwise_f32", ["--dtypes", "f32", "--ops", "add", "subtract", "multiply",
      "scale_shift", "relu_mul", "--sizes", *map(str, BOUNDARIES)])
]
COHORT_SIZES = {name: 36 if name.startswith("reduce_") else 32 if name.startswith("dtype_") else 45
                for name, _ in COHORTS}


def execute(cohort, output, runner):
    name, selection = cohort
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    local_report = folder / "local-results.json"
    if not local_report.exists():
        local_report = folder / "results.json"  # older fixtures used one filename
    if runner and not local_report.exists():
        raise ValueError(f"Generate {name} locally before running it")
    if runner and len(json.loads(local_report.read_text())["cases"]) != COHORT_SIZES[name]:
        raise ValueError(f"Local generation of {name} is incomplete")
    command = [sys.executable, str(ROOT / "tools/test_extended.py"), *selection,
               "--out", str(folder), "--reuse", "--ir-reference"]
    if runner:
        command.extend(["--runner", "--pace", "0.2"])
    else:
        command.extend(["--report-name", "local-results.json"])
    log = folder / ("runner.log" if runner else "generation.log")
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({"cohort": name, "phase": "runner" if runner else "local",
                      "exit_code": result.returncode, "log": str(log.relative_to(ROOT))}), flush=True)


def summarize(output, runner=True):
    cases = []
    for name, _ in COHORTS:
        folder = output / name
        report = folder / ("results.json" if runner else "local-results.json")
        if not report.exists():
            continue
        saved = json.loads(report.read_text())
        for original in saved["cases"]:
            row = dict(original, cohort=name, requested_work=saved["provenance"]["work"])
            meta_path = folder / row["id"] / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                row["launches"] = [{key: kernel[key] for key in ("entry", "grid", "block", "smem_bytes")}
                                   for kernel in meta["kernels"]]
            cases.append(row)
    summary = {"cases": len(cases), "statuses": dict(Counter(row["status"] for row in cases)),
               "dtypes": dict(Counter(row["dtype"] for row in cases)),
               "operations": dict(Counter(row["operation"] for row in cases)),
               "observed_reduction_blocks": {
                   str(work): dict(Counter(str(launch["block"][0])
                                          for row in cases if row["cohort"] == f"reduce_f32_w{work}"
                                          for launch in row.get("launches", [])))
                   for work in (32, 64, 128, 256)},
               "gpu_vs_ir": dict(Counter(row.get("details", {}).get("comparisons", {}).get("kernel_ir", {})
                                         .get("status", "not_executed") for row in cases)),
               "cpu_ir_vs_expression": dict(Counter(row.get("ir_vs_expression", {}).get("status", "not_checked")
                                                    for row in cases)),
               "failing_cases": [{key: row[key] for key in ("cohort", "id", "status", "error") if key in row}
                                 for row in cases if row["status"] not in ("pass", "compiled_not_executed")]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "results.json").write_text(json.dumps({"summary": summary, "cases": cases}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("generate", "run", "all"), default="generate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "tests/boundaries")
    parser.add_argument("--cohorts", nargs="+", choices=[name for name, _ in COHORTS])
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    args.out.mkdir(parents=True, exist_ok=True)
    cohorts = [cohort for cohort in COHORTS if args.cohorts is None or cohort[0] in args.cohorts]
    if args.phase != "run":
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(execute, cohort, args.out, False) for cohort in cohorts]
            for future in as_completed(futures):
                future.result()
    if args.phase != "generate":
        for cohort in cohorts:
            execute(cohort, args.out, True)
            summarize(args.out)
    cases = summarize(args.out, runner=args.phase != "generate")
    selected = [row for row in cases if row["cohort"] in {name for name, _ in cohorts}]
    return int(len(selected) != sum(COHORT_SIZES[name] for name, _ in cohorts) or any(
        row["status"] not in ("pass", "compiled_not_executed")
        or row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass"
        for row in selected))


if __name__ == "__main__":
    raise SystemExit(main())
