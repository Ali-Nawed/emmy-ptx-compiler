#!/usr/bin/env python3
"""LOCAL ONLY: compare process-level PTX generation and assembly latency.

Our input is scheduled Kernel IR; nvcc's input is saved Emmy CUDA. Both produce
PTX for sm_89. No reference kernel is uploaded or executed. Repeated CLI calls
include process/import overhead; separately report already-imported API costs.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from statistics import median
import subprocess
import sys
import tempfile
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import emit_ptx, load_kernel

CASES = {
    "p00_add": "problems/p00_scalar_add",
    "p02_scale_shift": "problems/p02_scale_shift",
    "r04_row_sum": "problems/r04_row_sum",
    "tanh_f32": "tests/activations/tanh_f32_129",
    "scalar_matmul_f16": "tests/walkthrough/matmul_scalar_f16_16x8x16",
    "tensor_matmul_f16": "tests/walkthrough/matmul_tensor_f16_16x8x16",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(operation):
    start = perf_counter_ns()
    operation()
    return (perf_counter_ns()-start)/1e6


def summary(samples):
    return dict(median_ms=median(samples), min_ms=min(samples), max_ms=max(samples), samples_ms=samples)


def run(command, stdout=None):
    result = subprocess.run(command, stdout=stdout or subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--api-repeats", type=int, default=100)
    parser.add_argument("--only", nargs="+", choices=CASES)
    parser.add_argument("--out", type=Path, default=ROOT / "review/compile-time")
    args = parser.parse_args()
    if min(args.repeats, args.api_repeats) < 1 or args.warmups < 0:
        parser.error("positive repeat counts and nonnegative warmups required")
    for program in ("nvcc", "ptxas"):
        if shutil.which(program) is None:
            parser.error(f"requires local {program}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    cpu_info = Path("/proc/cpuinfo")
    cpu_model = next((line.split(":",1)[1].strip() for line in cpu_info.read_text().splitlines()
                      if line.startswith("model name")), "unknown") if cpu_info.exists() else platform.processor()
    report = {"boundary": "Kernel IR -> our CLI -> PTX versus saved Emmy CUDA -> nvcc -> PTX",
              "python": sys.version, "platform": platform.platform(), "logical_cpus": os.cpu_count(),
              "cpu_model": cpu_model,
              "emmy_version": importlib.metadata.version("emmy-ml"),
              "nvcc_version": subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout,
              "ptxas_version": subprocess.run(["ptxas", "--version"], capture_output=True, text=True, check=True).stdout,
              "repeats": args.repeats, "warmups": args.warmups, "api_repeats": args.api_repeats,
              "input_staging": "Both inputs, output files and our three source modules copied to Linux temporary storage",
              "reference_uploaded": False, "reference_executed": False,
              "notes": ["CLI times include Python startup, Emmy imports, parsing, emission and output writing.",
                        "nvcc input is pre-rendered CUDA; Emmy tracing, scheduling and rendering are excluded.",
                        "CLI and nvcc samples use new processes after untimed warmups; OS caches are not flushed.",
                        "API times exclude imports and process startup; they are not the submission CLI time.",
                        "Assembly is separately measured with ptxas -arch=sm_89 -O3 for both PTX variants.",
                        "CPU clocks/load are uncontrolled; measurements are specific to this local machine."],
              "source_sha256": {n:digest(ROOT/n) for n in ("compile.py", "ptx_math.py", "ptx_tensor.py")},
              "cases": rows}
    with tempfile.TemporaryDirectory(prefix="emmy-compile-time-") as temporary:
        stage = Path(temporary)
        compiler = stage / "compiler"
        compiler.mkdir()
        for name in report["source_sha256"]:
            shutil.copyfile(ROOT/name, compiler/name)
        for name, relative in CASES.items():
            if args.only and name not in args.only:
                continue
            folder = ROOT / relative
            row = {"case": name, "fixture": relative, "status": "failed"}
            try:
                meta = json.loads((folder/"meta.json").read_text())
                descriptor, = meta["kernels"]
                entry = descriptor["entry"]
                local = stage / name
                local.mkdir()
                ir = local / f"{entry}.ir.json"
                shutil.copyfile(folder/ir.name, ir)
                shutil.copyfile(folder/"meta.json", local/"meta.json")
                reference = folder / f"{entry}.ref.cu"
                cuda = local / "reference.cu"
                cuda.write_text("#include <cuda_fp16.h>\n#include <stdint.h>\n"+reference.read_text())
                ours, nvcc = local/"ours.ptx", local/"nvcc.ptx"
                ours_command = [sys.executable, str(compiler/"compile.py"), str(ir)]
                nvcc_command = ["nvcc", "--ptx", "--gpu-architecture=compute_89", "-O3", "--std=c++17",
                                str(cuda), "-o", str(nvcc)]
                def ours_ptx():
                    with ours.open("wb") as output:
                        run(ours_command, stdout=output)
                operations = {"ours": ours_ptx, "nvcc": lambda: run(nvcc_command)}
                def assembly(path):
                    run(["ptxas", "-arch=sm_89", "-O3", str(path), "-o", str(path.with_suffix(".cubin"))])
                for _ in range(args.warmups):
                    for variant, operation in operations.items():
                        operation()
                        assembly(ours if variant=="ours" else nvcc)
                samples = {k:[] for k in ("ours_cli_ptx", "nvcc_ptx", "ours_ptxas", "nvcc_ptxas", "ours_combined", "nvcc_combined")}
                for repeat in range(args.repeats):
                    for variant in (("ours", "nvcc") if repeat%2==0 else ("nvcc", "ours")):
                        generation = measure(operations[variant])
                        assembled = measure(lambda: assembly(ours if variant=="ours" else nvcc))
                        samples["ours_cli_ptx" if variant=="ours" else "nvcc_ptx"].append(generation)
                        samples[f"{variant}_ptxas"].append(assembled)
                        samples[f"{variant}_combined"].append(generation+assembled)
                source = load_kernel(ir)
                # Warm the in-process path before collecting API measurements.
                emit_ptx(source)
                samples["api_parse_and_emit"] = [measure(lambda: emit_ptx(load_kernel(ir))) for _ in range(args.api_repeats)]
                samples["api_emit_only"] = [measure(lambda: emit_ptx(source)) for _ in range(args.api_repeats)]
                row.update(status="measured", entry=entry, commands={"ours":ours_command,"nvcc":nvcc_command},
                           fixture_sha256={ir.name:digest(ir),"meta.json":digest(local/"meta.json"),reference.name:digest(reference)},
                           timings={k:summary(v) for k,v in samples.items()})
                row["nvcc_over_ours_cli"] = row["timings"]["nvcc_ptx"]["median_ms"]/row["timings"]["ours_cli_ptx"]["median_ms"]
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
            (args.out/"results.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps({"case":name,"status":row["status"],
                  "median_ms":{k:round(v["median_ms"],3) for k,v in row.get("timings",{}).items()},
                  "error":row.get("error")}),flush=True)
    return int(any(row["status"]!="measured" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
