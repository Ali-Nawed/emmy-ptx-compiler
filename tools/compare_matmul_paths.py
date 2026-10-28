#!/usr/bin/env python3
"""Compare matched scalar/MMA schedules generated from the same PyTorch input."""

import argparse
import hashlib
import importlib.metadata
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


CODE = ("a = torch.randn(16, 16, dtype=torch.float16); "
        "b = torch.randn(16, 8, dtype=torch.float16); a @ b")
SCHEDULES = {
    "scalar": "WORK=t8x4,TILE=f1x1,REDUCE=,STAGE=",
    "tensor": "WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,STAGE=d1/smem-async",
}


def expected(a, b):
    return [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float16)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("emmy-ml") != "0.3.11":
        parser.error("requires pinned emmy-ml==0.3.11")
    import torch
    a = np.random.default_rng(0).standard_normal((16,16)).astype(np.float16)
    b = np.random.default_rng(1).standard_normal((16,8)).astype(np.float16)
    np.testing.assert_allclose((torch.from_numpy(a) @ torch.from_numpy(b)).numpy(), expected(a,b)[0],
                               rtol=2e-3, atol=2e-3)
    output = ROOT / "review/walkthrough"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, knobs in SCHEDULES.items():
        spec = dict(id=f"matmul_{path}_f16_16x8x16", code=CODE, knobs=knobs,
                    ref=expected, rtol=2e-3, atol=2e-3)
        folder = ROOT / "tests/walkthrough" / spec["id"]
        row = {"path": path, "fixture": str(folder.relative_to(ROOT)), "knobs": knobs,
               "status": "upstream_generation_failed"}
        try:
            if not args.reuse:
                save_expression(folder, CODE)
                with tempfile.TemporaryDirectory(prefix="emmy-matmul-path-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
            proof = verify(folder)
            if proof["code"] != CODE or proof["knobs"] != knobs:
                raise ValueError("saved fixture differs from selected PyTorch source/schedule")
            meta = json.loads((folder / "meta.json").read_text())
            if not meta["kernels"]:
                raise ValueError("no generated runtime kernel")
            row.update(status="compiler_or_assembly_failure", kernels=meta["kernels"])
            compiled = {}
            for descriptor in meta["kernels"]:
                entry = descriptor["entry"]
                ptx = emit_ptx(load_kernel(folder / f"{entry}.ir.json"))
                destination = folder / f"{entry}.ours.ptx"
                destination.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(destination), "-o", str(destination.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[entry] = ptx
            row["status"] = "reference_failure"
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as original:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, original, meta)
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
                if row["status"] == "pass" and row["details"]["comparisons"]["kernel_ir"]["status"] != "pass":
                    row["status"] = "gpu_ir_mismatch"
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-3500:]
        rows.append(row)
        report = {"pipeline": "same PyTorch -> Emmy 0.3.11 with different schedules -> unchanged IR -> our PTX",
                  "runner": args.runner, "code": CODE, "cases": rows,
                  "source_sha256": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                    for n in ("compile.py", "ptx_math.py", "ptx_tensor.py")}}
        (output / "matmul-paths.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"{row['status']:28} {path}", flush=True)
        if "error" in row:
            print(row["error"][-1000:], flush=True)
    return int(any(r["status"] not in ("pass", "assembled_not_executed")
                   or r.get("ir_vs_expression") != "pass" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
