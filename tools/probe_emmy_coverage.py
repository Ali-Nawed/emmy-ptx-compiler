#!/usr/bin/env python3
"""Distinguish real Emmy scope from compiler gaps; never synthesize missing IR."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from generate import trace
from emmy_fixtures import package, verify
from test_extended import ROOT, check_runner, save_expression
from compile import emit_ptx, load_kernel
from ir_reference import simulate


def specs():
    return [
        ("row_broadcast", "x = torch.randn(3, 129); w = torch.randn(129); x + w", lambda x,w: [x+w]),
        ("column_broadcast", "x = torch.randn(3, 129); b = torch.randn(3, 1); x + b", lambda x,b: [x+b]),
        ("multi_axis_sum", "x = torch.randn(2, 3, 129); x.sum((0, 2))",
         lambda x: [x.astype(np.float64).sum((0,2)).astype(np.float32)]),
        ("two_row_sums", "x = torch.randn(3, 129); y = torch.randn(3, 129); x.sum(-1) + y.sum(-1)",
         lambda x,y: [(x.astype(np.float64).sum(-1)+y.astype(np.float64).sum(-1)).astype(np.float32)]),
        ("where", "x = torch.randn(129); y = torch.randn(129); torch.where(x > 0, x, y)", lambda x,y: [np.where(x>0,x,y)]),
        ("sigmoid", "x = torch.randn(129); torch.sigmoid(x)", lambda x: [1/(1+np.exp(-x))]),
        ("tanh", "x = torch.randn(129); torch.tanh(x)", lambda x: [np.tanh(x)]),
        ("variance_composition", "x = torch.randn(3, 129); (x*x).mean(-1) - x.mean(-1)**2",
         lambda x: [((x.astype(np.float64)**2).mean(-1)-x.astype(np.float64).mean(-1)**2).astype(np.float32)]),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    output = ROOT/"review/native-math"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, code, reference in specs():
        spec = dict(id=f"probe_{name}", code=code, ref=reference, rtol=1e-4, atol=1e-5)
        if name in ("multi_axis_sum", "two_row_sums", "variance_composition"):
            spec["knobs"] = "WORK=t128,REDUCE=coop"
        folder = ROOT/"tests/coverage-probes"/spec["id"]
        row = {"case": name, "status": "upstream_generation_failed", "emmy_generated": False}
        try:
            save_expression(folder, code)
            with tempfile.TemporaryDirectory(prefix="emmy-coverage-") as temporary:
                dump = Path(temporary)/"dump"
                trace(str(Path(sys.executable).parent/"emmy"), spec, dump, "sm_89")
                package(spec, dump, folder)
            verify(folder)
            row.update(emmy_generated=True, status="compiler_unsupported")
            meta = json.loads((folder/"meta.json").read_text())
            compiled = {}
            for descriptor in meta["kernels"]:
                ptx = emit_ptx(load_kernel(folder/f"{descriptor['entry']}.ir.json"))
                path = folder/f"{descriptor['entry']}.ours.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[descriptor["entry"]] = ptx
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, simulate(folder))
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-2500:]
        rows.append(row)
        (output/"coverage-probes.json").write_text(json.dumps({"runner": args.runner, "cases": rows,
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py","ptx_math.py")},
            "statuses": dict(Counter(row["status"] for row in rows))}, indent=2)+"\n")
        print(f"{row['status']:28} {name}", flush=True)
        if "error" in row:
            print(row["error"][-400:], flush=True)
    return 0  # This is an audit, not a passing-coverage claim.


if __name__ == "__main__":
    raise SystemExit(main())
