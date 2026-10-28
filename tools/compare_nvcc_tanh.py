#!/usr/bin/env python3
"""LOCAL ONLY: inspect nvcc tanh lowering from saved, source-attested Emmy CUDA.

No runner or networking. Unsupported CUDA source is reported without rewriting
the reference, input IR, expectations or compiler implementation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from emmy_fixtures import verify
from compare_emmy_ptx import ptx_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import emit_ptx, load_kernel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", choices=("f32", "f16"), default=["f32", "f16"])
    parser.add_argument("--out", type=Path, default=ROOT / "review/walkthrough/tanh-nvcc")
    args = parser.parse_args()
    if shutil.which("nvcc") is None:
        parser.error("requires local nvcc")
    args.out.mkdir(parents=True, exist_ok=True)
    version = subprocess.run(["nvcc", "--version"], check=True, text=True, capture_output=True).stdout
    rows = []
    for dtype in args.dtypes:
        folder = ROOT / "tests/activations" / f"tanh_{dtype}_129"
        verify(folder)
        descriptor, = json.loads((folder / "meta.json").read_text())["kernels"]
        entry = descriptor["entry"]
        reference = folder / f"{entry}.ref.cu"
        ir = folder / f"{entry}.ir.json"
        ours = emit_ptx(load_kernel(ir))
        (args.out / f"{dtype}-ours.ptx").write_text(ours)
        cuda = args.out / f"{dtype}-emmy.cu"
        cuda.write_text("#include <cuda_fp16.h>\n"+reference.read_text())
        for mode, flags in (("default", []), ("fast_math", ["--use_fast_math"])):
            ptx = args.out / f"{dtype}-nvcc-{mode}.ptx"
            command = ["nvcc", "--ptx", "--gpu-architecture=compute_89", "-O3", "--std=c++17",
                       *flags, str(cuda), "-o", str(ptx)]
            result = subprocess.run(command, text=True, capture_output=True, timeout=90)
            log = args.out / f"{dtype}-{mode}.log"
            log.write_text(result.stdout+result.stderr)
            row = {"dtype": dtype, "mode": mode, "command": command,
                   "source": str(reference.relative_to(ROOT)), "source_sha256": digest(reference),
                   "ir_sha256": digest(ir), "ours": ptx_stats(ours),
                   "status": "ptx_generated_not_executed" if result.returncode==0 else "nvcc_source_compile_failed",
                   "exit_code": result.returncode, "diagnostics": result.stderr}
            if result.returncode == 0:
                row["reference_ptx"] = ptx_stats(ptx.read_text())
                row["reference_ptx_sha256"] = digest(ptx)
            rows.append(row)
            print(f"{dtype:3} {mode:10} {row['status']}", flush=True)
    report = {"nvcc_version": version, "target": "compute_89", "reference_uploaded": False,
              "reference_executed": False, "performance_measured": False, "cases": rows,
              "compiler_sha256": {n: digest(ROOT/n) for n in ("compile.py", "ptx_math.py", "ptx_tensor.py")}}
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2)+"\n")
    # Keep compile failures visible in both the report and command exit status.
    return int(any(row["exit_code"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
