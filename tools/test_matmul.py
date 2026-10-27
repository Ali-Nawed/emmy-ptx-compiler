#!/usr/bin/env python3
"""Generate genuine matmul schedules and inventory their compiler requirements."""

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


def specifications():
    cases = [
        ("scalar", "f32", 16, 16, 16, "WORK=t8x4,TILE=f1x1,STAGE="),
        ("async", "f32", 16, 16, 16, "WORK=t8x4,TILE=f1x1,STAGE=d1/smem-async"),
        ("mma_global", "f16", 16, 8, 16, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,STAGE="),
        ("mma_shared", "f16", 16, 8, 16, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,STAGE=d1/smem-async"),
    ]
    for name, dtype, m, n, k, knobs in cases:
        torch_type = "float16" if dtype == "f16" else "float32"
        yield dict(id=f"matmul_{name}_{dtype}_{m}x{n}x{k}",
            code=f"a = torch.randn({m}, {k}, dtype=torch.{torch_type}); "
                 f"b = torch.randn({k}, {n}, dtype=torch.{torch_type}); a @ b",
            ref=lambda a,b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(a.dtype)],
            knobs=knobs, rtol=2e-3 if dtype == "f16" else 1e-4,
            atol=2e-3 if dtype == "f16" else 1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    output = ROOT / "review/matmul"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specifications():
        folder = ROOT / "tests/matmul" / spec["id"]
        row = {"case": spec["id"], "status": "upstream_generation_failed", "knobs": spec["knobs"]}
        try:
            if not args.reuse:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-matmul-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
            verify(folder)
            meta = json.loads((folder / "meta.json").read_text())
            sources = [load_kernel(folder / f"{d['entry']}.ir.json") for d in meta["kernels"]]
            row["statements"] = dict(Counter(type(s).__name__ for source in sources for s in source.kernel.body.iter()))
            row["status"] = "compiler_unsupported"
            compiled = {}
            for source in sources:
                ptx = emit_ptx(source)
                path = folder / f"{source.kernel.name}.ours.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               capture_output=True, text=True, check=True, timeout=60)
                compiled[source.kernel.name] = ptx
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-3500:]
        rows.append(row)
        (output / "validation.json").write_text(json.dumps({"pipeline": "PyTorch -> Emmy -> unchanged Kernel IR -> our compiler",
            "runner": args.runner, "cases": rows,
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py")}}, indent=2)+"\n")
        print(f"{row['status']:28} {spec['id']}", flush=True)
        if "error" in row:
            print(row["error"][-800:], flush=True)
    return int(any(r["status"] not in ("pass", "assembled_not_executed", "upstream_generation_failed") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
