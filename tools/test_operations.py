#!/usr/bin/env python3
"""Fresh PyTorch -> Emmy -> our compiler coverage for arithmetic combinations.

Preserve expression and typed-IR comparisons separately. --reuse rechecks
existing source-matched fixtures without regenerating them. Only our PTX runs.
"""

import argparse
from collections import Counter
from functools import partial
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
from test_extended import ROOT, check_runner, compare_outputs, save_expression, validate_numpy
from compile import emit_ptx, load_kernel

DTYPES = {"f16": "float16", "f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}
OPS = ("divide", "minmax", "negative_abs", "exp", "exp2", "log", "log2", "log10", "power", "rsqrt",
       "sum_composed", "center", "maximum", "minimum", "product", "softmax")
REDUCTIONS = {"sum_composed", "center", "maximum", "minimum", "product", "softmax"}
INTEGER_OPS = {"minmax", "negative_abs", "power", "maximum", "minimum", "product"}


def reference(*arrays, operation):
    x = arrays[0]
    y = arrays[1] if len(arrays) > 1 else None
    one, quarter, half = (np.asarray(v, dtype=x.dtype) for v in (1, .25, .5))
    positive = x * x + one
    if operation == "divide":
        result = (x - y) / (y*y + one)
    elif operation == "minmax":
        result = np.maximum(x, y) - np.minimum(x, y)
    elif operation == "negative_abs":
        result = -np.abs(x) + y
    elif operation in ("exp", "exp2"):
        scaled = x * quarter
        if operation == "exp2":
            scaled = scaled * np.asarray(np.log(2.), dtype=x.dtype)
        result = np.exp(scaled)
    elif operation in ("log", "log2", "log10"):
        result = np.log(positive)
        if operation != "log":
            result = result / np.asarray(np.log(2. if operation == "log2" else 10.), dtype=x.dtype)
    elif operation == "power":
        result = np.power(x, y*y) if x.dtype.kind == "i" else np.power(positive, y * half)
    elif operation == "rsqrt":
        result = one / np.sqrt(positive)
    elif operation == "sum_composed":
        values = (x - y) / (y*y + one)
        result = values.astype(np.float64).sum(-1).astype(x.dtype)
    elif operation == "center":
        result = (x.astype(np.float64) - x.astype(np.float64).mean(-1, keepdims=True)).astype(x.dtype)
    elif operation in ("maximum", "minimum"):
        result = (np.max if operation == "maximum" else np.min)(x, axis=-1)
    elif operation == "product":
        if x.dtype.kind == "i":
            result = x.astype(np.int64).prod(-1)
        else:
            values = one + x * np.asarray(.01, dtype=x.dtype)
            result = values.astype(np.float64).prod(-1).astype(x.dtype)
    elif operation == "softmax":
        wide = x.astype(np.float64)
        e = np.exp(wide - wide.max(-1, keepdims=True))
        result = (e / e.sum(-1, keepdims=True)).astype(x.dtype)
    else:
        raise ValueError(operation)
    return [np.asarray(result)]


def specs(dtypes, operations, sizes, work):
    for dtype in dtypes:
        for operation in operations:
            if dtype.startswith("i") and operation not in INTEGER_OPS:
                continue
            for size in sizes:
                shape = (3, size) if operation in REDUCTIONS else (size,)
                ctor = f"torch.randn({shape!r}, dtype=torch.{DTYPES[dtype]})"
                if dtype.startswith("i"):
                    ctor = f"torch.randint(-3, 4, {shape!r}, dtype=torch.{DTYPES[dtype]})"
                setup = f"x = {ctor}; "
                if operation in ("divide", "minmax", "negative_abs", "power", "sum_composed"):
                    setup += f"y = {ctor}; "
                tail = {
                    "divide": "(x - y) / (y*y + 1)",
                    "minmax": "torch.maximum(x, y) - torch.minimum(x, y)",
                    "negative_abs": "-torch.abs(x) + y",
                    # The pinned upstream CUDA packaging stage cannot render
                    # direct exp2/log2/log10. Express these combinations at the
                    # PyTorch level; backend tests cover the direct IR op too.
                    "exp": "torch.exp(x * 0.25)", "exp2": "torch.exp(x * 0.25 * 0.6931471805599453)",
                    "log": "torch.log(x*x + 1)", "log2": "torch.log(x*x + 1) / 0.6931471805599453",
                    "log10": "torch.log(x*x + 1) / 2.302585092994046", "power": "torch.pow(x*x + 1, y * 0.5)",
                    "rsqrt": "torch.rsqrt(x*x + 1)",
                    "sum_composed": "((x - y) / (y*y + 1)).sum(-1)",
                    "center": "x - x.mean(-1, keepdim=True)",
                    "maximum": "x.amax(-1)", "minimum": "x.amin(-1)",
                    "product": "x.prod(-1)" if dtype.startswith("i") else "(1 + x * 0.01).prod(-1)",
                    "softmax": "torch.softmax(x, dim=-1)",
                }[operation]
                if dtype.startswith("i") and operation == "power":
                    tail = "torch.pow(x, y*y)"
                rtol, atol = {"f16": (6e-3, 5e-3), "f32": (3e-5, 2e-6), "f64": (1e-11, 1e-12),
                              "i32": (0, 0), "i64": (0, 0)}[dtype]
                spec = dict(id=f"ops_{operation}_{dtype}_{'x'.join(map(str, shape))}_w{work}",
                            code=setup+tail, ref=partial(reference, operation=operation),
                            rtol=rtol, atol=atol, dtype=dtype, operation=operation, shape=list(shape))
                if operation in REDUCTIONS:
                    spec["knobs"] = f"WORK=t{work},REDUCE=coop"
                yield spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--ops", nargs="+", choices=OPS,
                        default=[op for op in OPS if op not in ("minimum", "product")],
                        help="amin/prod are opt-in upstream tracer probes; unsupported traces remain coverage gaps")
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 129])
    parser.add_argument("--work", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--report-name", default="validation.json")
    args = parser.parse_args()
    if min(args.sizes) < 1 or Path(args.report_name).name != args.report_name:
        parser.error("positive sizes and a simple report filename are required")
    if importlib.metadata.version("emmy-ml") != "0.3.11":
        parser.error("requires emmy-ml==0.3.11")
    output = ROOT / "review/operations"
    output.mkdir(parents=True, exist_ok=True)
    provenance = {key: importlib.metadata.version(key) for key in ("emmy-ml", "torch", "numpy")}
    provenance["target"] = "sm_89"
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py")}
    rows = []
    for spec in specs(args.dtypes, args.ops, args.sizes, args.work):
        folder = ROOT / "tests/operations" / spec["id"]
        row = {"case": spec["id"], "status": "reference_failure"}
        try:
            validate_numpy(spec)
            row["status"] = "generation_failure"
            provenance_path = folder / "fixture-provenance.json"
            if args.reuse and provenance_path.exists():
                verify(folder)
                old_meta = json.loads((folder / "meta.json").read_text())
                if old_meta["code"] != spec["code"] or json.loads(provenance_path.read_text()) != provenance:
                    raise ValueError("Saved fixture source/environment changed; regenerate without --reuse")
            else:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-operations-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
                provenance_path.write_text(json.dumps(provenance, indent=2)+"\n")
            row["status"] = "ir_reference_failure"
            meta = json.loads((folder / "meta.json").read_text())
            ir_reference = simulate(folder)
            np.savez(folder / "expected_ir.npz", **ir_reference)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(ir_reference, expected, meta)
            compiled = {}
            row["status"] = "compiler_failure"
            for kernel in meta["kernels"]:
                ptx = emit_ptx(load_kernel(folder / f"{kernel['entry']}.ir.json"))
                path = folder / f"{kernel['entry']}.ours.ptx"
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[kernel["entry"]] = ptx
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, ir_reference)
                row["gpu_vs_ir"] = row["details"].get("comparisons", {}).get("kernel_ir", {}).get("status", "not_compared")
                time.sleep(1)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n" + (exc.stderr or "")[-6000:]
        rows.append(row)
        (output / args.report_name).write_text(json.dumps({"source_sha256": hashes,
            "provenance": provenance, "runner": args.runner, "cases": rows,
            "statuses": dict(Counter(row["status"] for row in rows))}, indent=2)+"\n")
        print(f"{row['status']:26} {row['case']} IR={row.get('ir_vs_expression')} GPU/IR={row.get('gpu_vs_ir')}", flush=True)
        if "error" in row:
            print(row["error"][-900:], flush=True)
    return int(any(row["status"] not in ("pass", "assembled_not_executed")
                   or row.get("gpu_vs_ir", "pass") != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
