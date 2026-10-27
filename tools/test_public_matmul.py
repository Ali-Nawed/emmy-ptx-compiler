#!/usr/bin/env python3
"""Validate our compiler on unchanged, organizer-supplied matmul packages."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from test_extended import ROOT, check_runner
from compile import emit_ptx, load_kernel
from ir_reference import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--ir-reference", action="store_true", help="Also interpret the supplied schedule; large scalar matmuls are slow")
    parser.add_argument("--cases", nargs="+", choices=("m01_matmul_512", "m02_matmul_rect"), default=["m01_matmul_512"])
    args = parser.parse_args()
    output = ROOT / "review/shared-matmul"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in args.cases:
        folder = ROOT / "problems" / name
        meta = json.loads((folder / "meta.json").read_text())
        row = {"case": name, "status": "compiler_failure", "ir_reference": args.ir_reference}
        try:
            compiled = {}
            for descriptor in meta["kernels"]:
                ptx = emit_ptx(load_kernel(folder / f"{descriptor['entry']}.ir.json"))
                destination = output / name
                destination.mkdir(parents=True, exist_ok=True)
                path = destination / f"{descriptor['entry']}.ours.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[descriptor["entry"]] = ptx
            row["status"] = "assembled_not_executed"
            if args.runner:
                reference = simulate(folder) if args.ir_reference else None
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-2000:]
        rows.append(row)
        (output / "public-validation.json").write_text(json.dumps({"source_sha256": {
            name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py", "ptx_tensor.py")},
            "runner": args.runner, "only_our_compiler_uploaded": True, "cases": rows}, indent=2)+"\n")
        print(f"{row['status']:28} {name}", flush=True)
        if "error" in row:
            print(row["error"][-800:], flush=True)
    return int(any(row["status"] not in ("pass", "assembled_not_executed") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
