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
from test_extended import ROOT, check_runner, compare_outputs, save_expression
from compile import emit_ptx, load_kernel
from ir_reference import simulate


def specifications():
    cases = [
        ("scalar", "f32", 16, 16, 16, "WORK=t8x4,TILE=f1x1,STAGE="),
        ("async", "f32", 16, 16, 16, "WORK=t8x4,TILE=f1x1,STAGE=d1/smem-async"),
        ("mma_global", "f16", 16, 8, 16, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,STAGE="),
        ("mma_shared", "f16", 16, 8, 16, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,STAGE=d1/smem-async"),
        ("mma_nosplit", "f16", 16, 8, 16, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_kloop", "f16", 16, 8, 32, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_tails", "f16", 17, 13, 19, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_tiles", "f16", 32, 24, 32, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_warps", "f16", 32, 16, 32, "WORK=w2x2,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_transposed", "f16", 16, 8, 16, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_tiny", "f16", 1, 1, 1, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE="),
        ("mma_fragments", "f16", 32, 16, 32, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f2x2,REDUCE=,STAGE="),
        ("shared_kloop", "f16", 16, 8, 32, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE=d1/smem-async"),
        ("shared_depth2", "f16", 16, 8, 64, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE=d2/smem-async"),
        ("shared_depth3", "f16", 16, 8, 128, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1/k2,REDUCE=,STAGE=d3/smem-async"),
        ("shared_k32", "f16", 16, 8, 64, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1/k2,REDUCE=,STAGE=d1/smem-async"),
        ("shared_k64", "f16", 16, 8, 64, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1/k4,REDUCE=,STAGE=d1/smem-async"),
        ("shared_warps", "f16", 32, 16, 32, "WORK=w2x2,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE=d1/smem-async"),
        ("shared_transposed", "f16", 16, 8, 32, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE=d1/smem-async"),
        ("shared_tails", "f16", 17, 13, 32, "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE=d1/smem-async"),
        ("shared_scalar", "f32", 32, 64, 64, "WORK=t32x16,TILE=f2x2,REDUCE=,STAGE=d1/smem-async"),
    ]
    for name, dtype, m, n, k, knobs in cases:
        torch_type = "float16" if dtype == "f16" else "float32"
        transposed = name in ("mma_transposed", "shared_transposed")
        suite = ("shared" if name == "mma_shared" or name.startswith("shared_") else
                 "scope" if name in ("async", "mma_fragments") else "global")
        yield dict(id=f"matmul_{name}_{dtype}_{m}x{n}x{k}",
            code=f"a = torch.randn({m}, {k}, dtype=torch.{torch_type}); "
                 f"b = torch.randn({n if transposed else k}, {k if transposed else n}, dtype=torch.{torch_type}); a @ {'b.transpose(0, 1)' if transposed else 'b'}",
            ref=lambda a,b,t=transposed: [(a.astype(np.float64) @ (b.T if t else b).astype(np.float64)).astype(a.dtype)],
            path=name, suite=suite, knobs=knobs, rtol=2e-3 if dtype == "f16" else 1e-4,
            atol=2e-3 if dtype == "f16" else 1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--report-name")
    parser.add_argument("--suite", choices=("global", "shared", "scope"), default="global")
    args = parser.parse_args()
    output = ROOT / "review/matmul"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specifications():
        if args.suite != "scope" and spec["suite"] != args.suite:
            continue
        if args.only and spec["id"] not in args.only:
            continue
        folder = ROOT / "tests/matmul" / spec["id"]
        row = {"case": spec["id"], "status": "upstream_generation_failed", "knobs": spec["knobs"]}
        try:
            if not args.reuse:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-matmul-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
            row["status"] = "fixture_provenance_failed"
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
            row["status"] = "reference_failure"
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, expected, meta)
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
                if (row["status"] == "pass" and
                        row["details"]["comparisons"]["kernel_ir"]["status"] != "pass"):
                    row["status"] = "gpu_ir_mismatch"
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-3500:]
        rows.append(row)
        (output / (args.report_name or f"{args.suite}-validation.json")).write_text(json.dumps({"pipeline": "PyTorch -> Emmy -> unchanged Kernel IR -> our compiler",
            "runner": args.runner, "cases": rows,
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py", "ptx_tensor.py")}}, indent=2)+"\n")
        print(f"{row['status']:28} {spec['id']}", flush=True)
        if "error" in row:
            print(row["error"][-800:], flush=True)
    return int(any(r["status"] not in ("pass", "assembled_not_executed", "upstream_generation_failed") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
