#!/usr/bin/env python3
"""Generate real serial-Loop regressions through PyTorch and pinned Emmy."""

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
from ir_reference import simulate
from compile import emit_ptx, load_kernel
from emmy.compiler.ir.stmt import Loop


def expression_reference(*arrays):
    # PyTorch sum promotes signed integer inputs to i64; floats retain dtype.
    sums = [x.sum(-1, dtype=np.int64) if x.dtype.kind == "i" else
            x.astype(np.float64).sum(-1).astype(x.dtype) for x in arrays]
    result = sums[0]
    for term in sums[1:]:
        result = result + term
    return [result]


def specifications():
    settings = [(dtype, width, 128, 2) for dtype in ("f16", "f32", "f64", "i32", "i64")
                for width in (1, 33, 129)]
    settings += [("f32", 257, 32, 2), ("f32", 257, 256, 2), ("f32", 129, 128, 3)]
    names = {"f16": "float16", "f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}
    for dtype, width, work, count in settings:
        shape = (3, width)
        ctor = (f"torch.randint(-3, 4, {shape!r}, dtype=torch.{names[dtype]})" if dtype.startswith("i")
                else f"torch.randn({shape!r}, dtype=torch.{names[dtype]})")
        variables = ("x", "y", "z")[:count]
        code = "; ".join([*(f"{name} = {ctor}" for name in variables),
                          " + ".join(f"{name}.sum(-1)" for name in variables)])
        yield {"id": f"loop_{count}sums_{dtype}_3x{width}_w{work}", "code": code,
               "ref": expression_reference, "knobs": f"WORK=t{work},REDUCE=coop",
               "rtol": 3e-3 if dtype == "f16" else 1e-4,
               "atol": 2e-2 if dtype == "f16" else 1e-5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    report_dir = ROOT / "review/serial-loops"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specifications():
        folder = ROOT / "tests/serial-loops" / spec["id"]
        row = {"case": spec["id"], "status": "upstream_generation_failed"}
        try:
            if args.reuse:
                verify(folder)
            else:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-serial-loop-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
                verify(folder)
            meta = json.loads((folder / "meta.json").read_text())
            compiled = {}
            row.update(status="compiler_failure", serial_loops=0)
            for descriptor in meta["kernels"]:
                source = load_kernel(folder / f"{descriptor['entry']}.ir.json")
                row["serial_loops"] += sum(isinstance(s, Loop) for s in source.kernel.body.iter())
                ptx = emit_ptx(source)
                path = folder / f"{descriptor['entry']}.ours.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[descriptor["entry"]] = ptx
            row["status"] = "reference_failure"
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, expected, meta)
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n" + (exc.stderr or "")[-2500:]
        rows.append(row)
        report = {"pipeline": "PyTorch -> Emmy -> unchanged Kernel IR -> our compiler",
                  "runner": args.runner, "cases": rows,
                  "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                    for name in ("compile.py", "ptx_math.py", "tools/ir_reference.py")},
                  "statuses": dict(Counter(r["status"] for r in rows))}
        (report_dir / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"{row['status']:28} {spec['id']} (Loop nodes: {row.get('serial_loops', '-')})", flush=True)
        if "error" in row:
            print(row["error"][-600:], flush=True)
    # Upstream failures are recorded as out of scope, not silently turned into passes.
    return int(any(r["status"] not in ("pass", "assembled_not_executed", "upstream_generation_failed") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
