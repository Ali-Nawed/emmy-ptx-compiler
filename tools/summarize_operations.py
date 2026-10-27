#!/usr/bin/env python3
"""Aggregate final operation reports while keeping known expression gaps visible."""

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ("review/operations/f32.json", "review/operations/other-types.json",
           "review/operations/work32.json", "review/operations/work256.json",
           "review/normalization/validation.json", "review/pytorch-only/validation.json")
KNOWN = {"ops_log2_f64_129_w128", "ops_log10_f64_129_w128", "ops_sum_composed_f64_3x129_w128",
         "ops_center_f64_3x1_w128", "ops_center_f64_3x129_w128", "ops_maximum_f64_3x129_w128",
         "ops_softmax_f64_3x129_w128"}


def main():
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py")}
    rows, unexpected, suites = [], [], []
    for report_path in REPORTS:
        report = json.loads((ROOT/report_path).read_text())
        if report.get("source_sha256") != hashes:
            unexpected.append({"report": report_path, "error": "compiler source hash differs; rerun this suite"})
        if report.get("runner") is not True:
            unexpected.append({"report": report_path, "error": "not a hosted run"})
        suites.append({"report": report_path, "cases": len(report["cases"]),
                       "statuses": dict(Counter(row["status"] for row in report["cases"]))})
        for row in report["cases"]:
            name = row.get("case", row.get("fixture"))
            ir_status = row.get("gpu_vs_ir", row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status"))
            accepted_gap = (name in KNOWN and row["status"] == "wrong_result" and
                            row.get("ir_vs_expression") == "wrong_result" and ir_status == "pass")
            if (row["status"] != "pass" and not accepted_gap) or (ir_status is not None and ir_status != "pass"):
                unexpected.append({"case": name, "status": row["status"], "gpu_vs_ir": ir_status})
            rows.append({"case": name, "status": row["status"], "gpu_vs_ir": ir_status,
                         "kind": row.get("kind", "generated_or_public"), "known_expression_gap": accepted_gap})
    summary = {"source_sha256": hashes, "hosted_cases": len(rows),
               "statuses": dict(Counter(row["status"] for row in rows)),
               "numeric_gpu_vs_ir": dict(Counter(row["gpu_vs_ir"] for row in rows if row["gpu_vs_ir"] is not None)),
               "classification_cases": 0,
               "excluded_historical_synthetic_cases": 88,
               "known_expression_gaps": [row["case"] for row in rows if row["known_expression_gap"]],
               "unexpected": unexpected, "suites": suites}
    (ROOT / "review/operations/summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("source_sha256", "suites")}, indent=2))
    return int(bool(unexpected))


if __name__ == "__main__":
    raise SystemExit(main())
