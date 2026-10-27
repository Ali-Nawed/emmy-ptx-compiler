#!/usr/bin/env python3
"""Audit the exact eight MVP cases through unchanged submit.py --compiler."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASES = ("p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
         "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum")


def main():
    output = ROOT / "review/presentation"
    logs = output / "mvp-logs"
    logs.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
              for name in ("compile.py", "ptx_math.py", "ptx_tensor.py")}
    for case in CASES:
        command = [sys.executable, str(ROOT/"submit.py"), case, "--compiler", str(ROOT/"compile.py")]
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
            transcript = result.stdout+result.stderr
            status = "pass" if result.returncode == 0 and " PASS " in transcript else "fail"
            row = {"case": case, "status": status, "exit_code": result.returncode,
                   "command": ["python", "submit.py", case, "--compiler", "compile.py"]}
        except subprocess.TimeoutExpired:
            transcript, row = "MVP runner command timed out\n", {"case": case, "status": "timeout"}
        (logs/f"{case}.log").write_text(transcript)
        rows.append(row)
        report = {"source_sha256": hashes, "path": "unchanged submit.py --compiler -> our PTX -> hosted runner",
                  "cases": rows, "passes": sum(r["status"] == "pass" for r in rows),
                  "required_cases": len(CASES), "all_pass": len(rows) == len(CASES) and all(r["status"] == "pass" for r in rows)}
        (output/"mvp-validation.json").write_text(json.dumps(report,indent=2)+"\n")
        print(f"{status:8} {case}", flush=True)
        if status != "pass":
            print(transcript[-1500:], flush=True)
    return int(not report["all_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
