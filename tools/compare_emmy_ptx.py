#!/usr/bin/env python3
"""LOCAL ONLY: compare our PTX with nvcc PTX from saved Emmy reference CUDA.

No runner client or networking is imported. Nothing here uploads or executes
reference code. Register counts and SASS are for the same local sm_89 assembler,
not measurements of RTX 5090 performance.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ("p00_scalar_add", "p01_vector_add", "p02_scale_shift", "p03_relu_mul",
          "r01_array_sum", "r02_big_sum", "r03_mean", "r04_row_sum")
FIXTURES = [(name, ROOT / "problems" / name) for name in PUBLIC]
FIXTURES += [(f"sum_f32_1025_w{work}", ROOT / f"tests/boundaries/reduce_f32_w{work}/ext_sum_f32_1025")
             for work in (32, 64, 128, 256)]
FIXTURES += [("row_sum_f32_17x1025", ROOT / "tests/boundaries/reduce_f32_w128/ext_row_sum_f32_17x1025")]
FIXTURES += [(f"scale_shift_{dtype}_1025", ROOT / f"tests/boundaries/dtype_{dtype}/ext_scale_shift_{dtype}_1025")
             for dtype in ("f16", "f64", "i32", "i64")]
FIXTURES += [(f"sum_{dtype}_1025", ROOT / f"tests/boundaries/dtype_{dtype}/ext_sum_{dtype}_1025")
             for dtype in ("f16", "f64")]


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stderr}")
    return result


def ptx_stats(source):
    operations = Counter()
    for line in source.splitlines():
        line = line.split("//", 1)[0].strip().lstrip("{").rstrip("}").strip()
        match = re.fullmatch(r"(?:@!?%\w+\s+)?([a-z][\w.]*)(?:\s+[^;]*)?;", line)
        if match:
            operations[match[1]] += 1
    return {"instruction_count": sum(operations.values()), "opcodes": dict(sorted(operations.items())),
            "vector_memory_instructions": sum(count for opcode, count in operations.items()
                                              if opcode.startswith(("ld.", "st.")) and re.search(r"\.v[248]\.", opcode))}


def sass_stats(source):
    opcodes = Counter(re.findall(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?(?:U?P\d+|U?PT)\s+)?([A-Z][A-Z0-9_.]*)\b", source))
    if not opcodes:
        raise ValueError("Could not parse disassembly")
    return {"sass_instruction_count": sum(opcodes.values()),
            "sass_non_nop_count": sum(opcodes.values()) - opcodes.get("NOP", 0),
            "sass_opcodes": dict(sorted(opcodes.items()))}


def assemble(ptx, no_fma=False):
    cubin = ptx.with_suffix(".cubin")
    command = ["ptxas", "-arch=sm_89", "-O3", "-v"]
    if no_fma:
        command.append("--fmad=false")
    result = run([*command, str(ptx), "-o", str(cubin)])
    ptx.with_suffix(".ptxas.txt").write_text(result.stderr)
    disassembly = run(["cuobjdump", "--dump-sass", str(cubin)]).stdout
    ptx.with_suffix(".sass.txt").write_text(disassembly)
    registers = re.search(r"Used (\d+) registers", result.stderr)
    if not registers:
        raise ValueError("Could not parse physical registers")
    spills = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", result.stderr)
    return {"registers": int(registers[1]),
            "spill_store_bytes": int(spills[1]) if spills else None,
            "spill_load_bytes": int(spills[2]) if spills else None,
            **sass_stats(disassembly)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "review/emmy-ptx-comparison")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for program in ("nvcc", "ptxas", "cuobjdump"):
        if shutil.which(program) is None:
            parser.error(f"Requires local CUDA tool {program}")
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {program: run([program, "--version"]).stdout for program in ("nvcc", "ptxas", "cuobjdump")}
    rows = []
    for name, fixture in FIXTURES[:args.limit]:
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = json.loads((fixture / "meta.json").read_text())
        for kernel in meta["kernels"]:
            entry = kernel["entry"]
            row = {"case": name, "entry": entry, "fixture": str(fixture.relative_to(ROOT)),
                   "grid": kernel["grid"], "block": kernel["block"], "status": "comparison_failure"}
            try:
                ir = fixture / f"{entry}.ir.json"
                reference = fixture / f"{entry}.ref.cu"
                row["source_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                        for path in (ir, reference, ROOT / "compile.py")}
                cuda = folder / f"{entry}.emmy.cu"
                cuda.write_text("#include <cuda_fp16.h>\n#include <stdint.h>\n" + reference.read_text())
                ours = folder / "ours.ptx"
                ours.write_text(run([sys.executable, str(ROOT / "compile.py"), str(ir)]).stdout)
                commands = {}
                for variant, extra in (("emmy", []), ("emmy_no_fma", ["--fmad=false"])):
                    destination = folder / f"{variant}.ptx"
                    command = ["nvcc", "--ptx", "--gpu-architecture=compute_89", "-O3", "--std=c++17",
                               *extra, str(cuda), "-o", str(destination)]
                    commands[variant] = command
                    run(command)
                row["commands"] = commands
                row["variants"] = {}
                for variant in ("ours", "emmy", "emmy_no_fma"):
                    path = folder / f"{variant}.ptx"
                    source = path.read_text()
                    if not re.search(r"\.entry\s+" + re.escape(entry) + r"\s*\(", source):
                        raise ValueError("PTX entry differs from the fixture ABI")
                    row["variants"][variant] = {**ptx_stats(source), **assemble(path, no_fma=variant == "emmy_no_fma")}
                row["assembler_flags"] = {"all": ["-arch=sm_89", "-O3", "-v"],
                                          "emmy_no_fma_additional": ["--fmad=false"]}
                row["status"] = "assembled_local_comparison"
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                row["error"] = str(exc)
            rows.append(row)
            (folder / "comparison.json").write_text(json.dumps(row, indent=2) + "\n")
            (args.out / "results.json").write_text(json.dumps({"target": "sm_89", "tool_versions": versions,
                "reference_uploaded": False, "performance_measured": False, "cases": rows}, indent=2) + "\n")
            print(json.dumps({"case": name, "status": row["status"],
                              "registers": {variant: stats["registers"] for variant, stats in row.get("variants", {}).items()},
                              "error": row.get("error")}), flush=True)
    return int(any(row["status"] != "assembled_local_comparison" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
