#!/usr/bin/env python3
"""Test activation emission only on genuine PyTorch -> Emmy Kernel IR."""

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
from test_pytorch_cases import explicit_inputs
from compile import emit_ptx, load_kernel
from ir_reference import simulate


def reference(x, operation):
    value = x.astype(np.float64)
    if operation == "tanh":
        result = np.tanh(value)
    else:
        t = np.exp(-np.abs(value))
        result = np.where(value >= 0, 1/(1+t), t/(1+t))
    return result.astype(x.dtype)


def specifications():
    for dtype, torch_type in (("f16", "float16"), ("f32", "float32"), ("f64", "float64")):
        rtol = {"f16": 2e-3, "f32": 2e-6, "f64": 2e-12}[dtype]
        for operation in ("sigmoid", "tanh"):
            for size in (1, 33, 129):
                yield dict(id=f"{operation}_{dtype}_{size}",
                    code=f"x = torch.randn({size}, dtype=torch.{torch_type}); torch.{operation}(x)",
                    ref=lambda x,op=operation: [reference(x,op)], rtol=rtol, atol=0.)
            # Explicitly exercise near-zero, subnormal, series-boundary and
            # saturating inputs. Stay within eager PyTorch's finite exp range.
            limit = {"f16": 80., "f32": 80., "f64": 700.}[dtype]
            tiny = np.nextafter(np.array(0., dtype=torch_type), np.array(1., dtype=torch_type)).item()
            magnitudes = [tiny, 1e-12, 1e-6, .01, .12499999, .125, .12500001, .5, 1., 8., 20., limit]
            values = np.array([-v for v in magnitudes[::-1]] + [-0., 0.] + magnitudes, dtype=torch_type)
            yield dict(id=f"{operation}_edges_{dtype}", inputs=[values],
                code=f"x = torch.tensor({values.tolist()}, dtype=torch.{torch_type}); torch.{operation}(x)",
                ref=lambda x,op=operation: [reference(x,op)], rtol=rtol, atol=0.)
        yield dict(id=f"silu_composed_{dtype}",
            code=f"x = torch.randn(129, dtype=torch.{torch_type}); x * torch.sigmoid(x)",
            ref=lambda x: [(x*reference(x,"sigmoid")).astype(x.dtype)], rtol=rtol, atol=0.)
        yield dict(id=f"tanh_sum_{dtype}",
            code=f"x = torch.randn(3, 129, dtype=torch.{torch_type}); torch.tanh(x).sum(-1)",
            ref=lambda x: [reference(x,"tanh").astype(np.float64).sum(-1).astype(x.dtype)],
            knobs="WORK=t128,REDUCE=coop", rtol=rtol, atol=1e-5 if dtype != "f64" else 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    output = ROOT / "review/activations"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specifications():
        folder = ROOT / "tests/activations" / spec["id"]
        row = {"case": spec["id"], "status": "upstream_generation_failed"}
        try:
            if not args.reuse:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-activation-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
                if "inputs" in spec:
                    explicit_inputs(folder, spec)
            verify(folder)
            meta = json.loads((folder / "meta.json").read_text())
            row["status"] = "compiler_failure"
            compiled = {}
            for descriptor in meta["kernels"]:
                source = emit_ptx(load_kernel(folder / f"{descriptor['entry']}.ir.json"))
                path = folder / f"{descriptor['entry']}.ours.ptx"
                path.write_text(source)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[descriptor["entry"]] = source
            row["status"] = "reference_failure"
            expected_ir = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(expected_ir, expected, meta)
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, expected_ir)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-2500:]
        rows.append(row)
        (output / "validation.json").write_text(json.dumps({"pipeline": "PyTorch -> Emmy -> unchanged Kernel IR -> our compiler",
            "runner": args.runner, "cases": rows, "statuses": dict(Counter(r["status"] for r in rows)),
            "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ("compile.py", "ptx_math.py", "tools/ir_reference.py")}}, indent=2)+"\n")
        print(f"{row['status']:28} {spec['id']}", flush=True)
        if "error" in row:
            print(row["error"][-700:], flush=True)
    return int(any(r["status"] not in ("pass", "assembled_not_executed", "upstream_generation_failed") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
