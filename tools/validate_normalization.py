#!/usr/bin/env python3
"""Validate public basics/norms, optionally generating fresh PyTorch/Emmy cases.

Only compile.py's output is uploaded when --runner is selected. This is testing,
not a final competition submission. Original expression expectations stay intact.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from generate import trace
from emmy_fixtures import package, verify
from ir_reference import simulate
from test_extended import ROOT, check_runner, compare_outputs, save_expression
from validate_emitter_optimizations import PUBLIC
from compile import emit_ptx, load_kernel


def specs():
    for dtype, torch_type in (("f16", "float16"), ("f32", "float32"), ("f64", "float64")):
        for size in (1, 129):
            for operation in ("sqrt", "reciprocal"):
                def reference(x, operation=operation):
                    one = np.asarray(1, dtype=x.dtype)
                    positive = x * x + one
                    return [np.sqrt(positive) if operation == "sqrt" else np.divide(one, positive)]
                tail = f"torch.{operation}(x * x + 1)"
                yield dict(id=f"norm_{operation}_{dtype}_{size}",
                           code=f"x = torch.randn({size}, dtype=torch.{torch_type}); {tail}",
                           ref=reference, rtol={"f16": 2e-3, "f32": 1e-6, "f64": 1e-12}[dtype], atol=0)
    for size in (1, 129, 1025):
        for operation in ("l2norm", "rmsnorm"):
            def reference(x, operation=operation):
                wide = x.astype(np.float64)
                denominator = (np.sqrt((wide * wide).sum(-1, keepdims=True)) if operation == "l2norm"
                               else np.sqrt((wide * wide).mean(-1, keepdims=True) + 1e-5))
                return [(wide / denominator).astype(x.dtype)]
            tail = ("x / torch.sqrt((x * x).sum(-1, keepdim=True))" if operation == "l2norm"
                    else "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5)")
            yield dict(id=f"norm_{operation}_f32_3x{size}",
                       code=f"x = torch.randn(3, {size}); {tail}", ref=reference,
                       knobs="WORK=t128,REDUCE=coop", rtol=1e-3, atol=1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    output = ROOT / "review/normalization"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    folders = [ROOT / "problems" / name for name in (*PUBLIC, "f03_l2norm", "f01_rmsnorm", "f02_row_softmax")]
    if args.generate:
        if importlib.metadata.version("emmy-ml") != "0.3.11":
            parser.error("requires emmy-ml==0.3.11")
        for spec in specs():
            folder = ROOT / "tests/normalization" / spec["id"]
            save_expression(folder, spec["code"])
            with tempfile.TemporaryDirectory(prefix="emmy-normalization-") as temporary:
                dump = Path(temporary) / "dump"
                trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                package(spec, dump, folder)
            (folder / "fixture-provenance.json").write_text(json.dumps({
                "emmy": importlib.metadata.version("emmy-ml"), "target": "sm_89",
                "torch": importlib.metadata.version("torch"), "numpy": np.__version__,
                "pipeline": "PyTorch expression -> Emmy Kernel IR -> our compile.py"}, indent=2) + "\n")
            print(f"generated {spec['id']}", flush=True)
    folders += sorted(path.parent for path in (ROOT / "tests/normalization").glob("*/meta.json"))
    for folder in folders:
        row = {"fixture": str(folder.relative_to(ROOT)), "status": "failure"}
        try:
            if folder.parent.name != "problems":
                verify(folder)
            meta = json.loads((folder / "meta.json").read_text())
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, expected, meta)
            compiled = {}
            destination = output / folder.name
            destination.mkdir(parents=True, exist_ok=True)
            for kernel in meta["kernels"]:
                ptx = emit_ptx(load_kernel(folder / f"{kernel['entry']}.ir.json"))
                path = destination / f"{kernel['entry']}.ours.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               capture_output=True, text=True, check=True, timeout=35)
                compiled[kernel["entry"]] = ptx
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
                time.sleep(1)
        except Exception as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n" + (exc.stderr or "")
        rows.append(row)
        report = {"compiler_sha256": hashlib.sha256((ROOT / "compile.py").read_bytes()).hexdigest(),
                  "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                    for name in ("compile.py", "ptx_math.py")},
                  "runner": args.runner, "cases": rows, "statuses": dict(Counter(r["status"] for r in rows))}
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{row['status']:26} {folder.name}", flush=True)
    return int(any(row["status"] not in ("pass", "assembled_not_executed")
                   or row.get("ir_vs_expression") != "pass"
                   or row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass"
                   for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
