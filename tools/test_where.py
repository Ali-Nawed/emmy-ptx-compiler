#!/usr/bin/env python3
"""Generate comparison/where programs through PyTorch -> pinned Emmy only."""

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


def specifications():
    types = {"f16": "float16", "f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}
    comparisons = [("gt", ">", np.greater), ("ge", ">=", np.greater_equal),
                   ("lt", "<", np.less), ("le", "<=", np.less_equal),
                   ("eq", "==", np.equal), ("ne", "!=", np.not_equal)]
    def spec(name, code, reference):
        return dict(id="where_"+name, code=code, ref=reference, rtol=0., atol=0.)
    for dtype, torch_type in types.items():
        for size in (1, 33, 129):
            ctor = (f"torch.randint(-3, 4, ({size},), dtype=torch.{torch_type})" if dtype.startswith("i")
                    else f"torch.randn({size}, dtype=torch.{torch_type})")
            yield spec(f"gt_{dtype}_{size}", f"x = {ctor}; y = {ctor}; torch.where(x > y, x, y)",
                       lambda x,y: [np.where(x>y,x,y)])
        yield spec(f"scalar_{dtype}", f"x = {ctor}; y = {ctor}; torch.where(x > 0, x, y)",
                   lambda x,y: [np.where(x>0,x,y)])
    ctor = "torch.randn(129)"
    for name, symbol, compare in comparisons[1:]:
        yield spec(f"{name}_f32_129", f"x = {ctor}; y = {ctor}; torch.where(x {symbol} y, x, y)",
                   lambda x,y,compare=compare: [np.where(compare(x,y),x,y)])
    expressions = [
        ("and", "torch.where((x > y) & (x < 1), x, y)", lambda x,y: np.where((x>y)&(x<1),x,y)),
        ("or", "torch.where((x > y) | (x < -1), x, y)", lambda x,y: np.where((x>y)|(x<-1),x,y)),
        ("not", "torch.where(~(x > y), x, y)", lambda x,y: np.where(~(x>y),x,y)),
        ("nested", "torch.where(x > y, torch.where(x > 0, x, y), y)",
         lambda x,y: np.where(x>y,np.where(x>0,x,y),y)),
    ]
    for name, expression, reference in expressions:
        yield spec(name, f"x = {ctor}; y = {ctor}; {expression}", lambda x,y,ref=reference: [ref(x,y)])
    for name, yshape in (("row_broadcast", "(129,)"), ("column_broadcast", "(3, 1)")):
        yield spec(name, f"x = torch.randn(3, 129); y = torch.randn({yshape}); torch.where(x > y, x, y)",
                   lambda x,y: [np.where(x>y,x,y)])
    masked = spec("masked_sum", "x = torch.randn(3, 129); y = torch.randn(3, 129); torch.where(x > y, x, y).sum(-1)",
                  lambda x,y: [np.where(x>y,x,y).astype(np.float64).sum(-1).astype(np.float32)])
    masked.update(knobs="WORK=t128,REDUCE=coop", rtol=1e-4, atol=1e-5)
    yield masked
    # Build adjacent large integers ON DEVICE, return small selected values.
    # This avoids any >2**53 JSON numeric precision loss at the hosted boundary.
    x = np.array([0, 1, -2, -1, 0, 1, -1], dtype=np.int64)
    y = np.array([1, 0, -1, -2, 0, -1, 1], dtype=np.int64)
    for name, symbol, compare in comparisons:
        case = spec(f"wide_{name}_i64", f"x = torch.tensor({x.tolist()}, dtype=torch.int64); "
                    f"y = torch.tensor({y.tolist()}, dtype=torch.int64); "
                    f"torch.where((x + 1152921504606846976) {symbol} (y + 1152921504606846976), x, y)",
                    lambda x,y,compare=compare: [np.where(compare(x+np.int64(2**60),y+np.int64(2**60)),x,y)])
        case["inputs"] = [x,y]
        yield case
    # Make NaNs on device from finite source inputs; selected outputs stay finite.
    x = np.array([0., 1., -1., 2., -0.], dtype=np.float32)
    y = np.array([0., 1., 2., -1., 0.], dtype=np.float32)
    for name, symbol, compare in (comparisons[2], comparisons[4], comparisons[5]):
        def reference(x, y, compare=compare):
            with np.errstate(invalid="ignore", divide="ignore"):
                return [np.where(compare(x/x, y), y, -y)]
        case = spec(f"nan_{name}_f32", f"x = torch.tensor({x.tolist()}, dtype=torch.float32); "
                    f"y = torch.tensor({y.tolist()}, dtype=torch.float32); torch.where((x / x) {symbol} y, y, -y)", reference)
        case["inputs"] = [x,y]
        yield case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    output = ROOT / "review/where"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specifications():
        if args.only and spec["id"] not in args.only:
            continue
        folder = ROOT / "tests/where" / spec["id"]
        row = {"case": spec["id"], "status": "upstream_generation_failed"}
        try:
            if not args.reuse:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-where-") as temporary:
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
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, expected, meta)
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
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
    return int(any(row["status"] not in ("pass", "assembled_not_executed", "upstream_generation_failed") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
