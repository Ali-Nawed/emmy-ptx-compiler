#!/usr/bin/env python3
"""Fresh, source-attested pointwise/reduction combinations for hidden-set gaps.

PyTorch -> pinned Emmy -> unchanged Kernel IR -> our PTX. This finite matrix
complements existing suites; it does not claim exhaustive input coverage.
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

import numpy as np

from generate import trace
from emmy_fixtures import package, verify
from test_extended import ROOT, check_runner, compare_outputs, save_expression
from compile import emit_ptx, load_kernel
from ir_reference import simulate


TYPES = {"f16": "float16", "f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}


def total(x, axis=None, keepdims=False, mean=False):
    """Expression oracle: wide reduction, independent of scheduled IR."""
    integer = x.dtype.kind == "i"
    wide = x.astype(np.int64 if integer and not mean else np.float64)
    result = (wide.mean if mean else wide.sum)(axis=axis, keepdims=keepdims)
    return result.astype(np.int64 if integer and not mean else x.dtype)


def specifications():
    def case(family, dtype, setup, expression, ref, work=None):
        return dict(id=f"hidden_{family}_{dtype}" + (f"_w{work}" if work else ""),
                    family=family, dtype=dtype, code=setup+expression,
                    ref=lambda *arrays: [np.asarray(ref(*arrays))],
                    knobs=f"WORK=t{work},REDUCE=coop" if work else "",
                    rtol={"f16": 6e-3, "f32": 1e-4, "f64": 1e-10, "i32": 0, "i64": 0}[dtype],
                    atol={"f16": 6e-3, "f32": 1e-5, "f64": 1e-10, "i32": 0, "i64": 0}[dtype])

    for position, (dtype, torch_type) in enumerate(TYPES.items()):
        def ctor(shape):
            return (f"torch.randint(-3, 4, {shape!r}, dtype=torch.{torch_type})" if dtype.startswith("i")
                    else f"torch.randn({shape!r}, dtype=torch.{torch_type})")
        setup = f"x = {ctor((2, 3, 33))}; y = {ctor((2, 3, 33))}; "
        yield case("chain_rank3", dtype, setup, "torch.relu(x-y) * (x+y)",
                   lambda x,y: np.maximum(x-y, np.asarray(0, x.dtype)) * (x+y))
        yield case("where_rank3", dtype, setup, "torch.where(x > y, x-y, x+y)",
                   lambda x,y: np.where(x>y, x-y, x+y))
        for name, shape in (("row_broadcast", (129,)), ("column_broadcast", (3, 1))):
            yield case(name, dtype, f"x = {ctor((3, 129))}; y = {ctor(shape)}; ",
                       "torch.relu(x + y) * y", lambda x,y: np.maximum(x+y, np.asarray(0,x.dtype))*y)
        work = (32, 64, 128, 256, 128)[position]
        setup = f"x = {ctor((3, 129))}; y = {ctor((3, 129))}; "
        yield case("masked_row_sum", dtype, setup, "torch.where(x > y, x-y, x+y).sum(-1)",
                   lambda x,y: total(np.where(x>y,x-y,x+y), -1), work)
        yield case("column_sum", dtype, f"x = {ctor((33, 3))}; ", "x.sum(0)",
                   lambda x: total(x, 0), work)
        yield case("all_sum_rank3", dtype, f"x = {ctor((2, 3, 33))}; ", "x.sum()",
                   lambda x: total(x), work)
        yield case("strided_slice", dtype, f"x = {ctor((257,))}; ", "x[::2] + x[1::2].sum()",
                   lambda x: (x[::2] + total(x[1::2])).astype(x.dtype), work)
        if not dtype.startswith("i"):
            yield case("mean_rank3", dtype, f"x = {ctor((2, 3, 129))}; ", "x.mean(-1)",
                       lambda x: total(x, -1, mean=True), work)
            yield case("two_reductions", dtype, setup, "x.sum(-1) + y.sum(-1)",
                       lambda x,y: total(x,-1)+total(y,-1), work)
            yield case("math_composition", dtype, f"x = {ctor((129,))}; y = {ctor((129,))}; ",
                       "torch.log(x*x + 1) + torch.exp(y * 0.125) + torch.pow(x*x + 1, y * 0.125)",
                       lambda x,y: np.log(x*x + np.asarray(1,x.dtype)) +
                       np.exp(y*np.asarray(.125,x.dtype)) +
                       np.power(x*x + np.asarray(1,x.dtype), y*np.asarray(.125,x.dtype)))
    for work, size in ((32,31), (64,32), (128,33), (256,257)):
        yield case("keepdim_center", "f32", f"x = torch.randn(3, {size}); ",
                   "x - x.mean(-1, keepdim=True)",
                   lambda x: (x.astype(np.float64)-x.astype(np.float64).mean(-1,keepdims=True)).astype(x.dtype), work)
    for left in TYPES:
        for right in TYPES:
            if left == right:
                continue
            # PyTorch tensor promotion: floating inputs determine the floating
            # result type; otherwise the wider signed integer type wins.
            common = next(t for t in ("f64", "f32", "f16", "i64", "i32") if t in (left, right))
            def constructor(dtype, shape):
                return (f"torch.randint(-3, 4, {shape!r}, dtype=torch.{TYPES[dtype]})"
                        if dtype.startswith("i") else f"torch.randn({shape!r}, dtype=torch.{TYPES[dtype]})")
            def mixed_ref(x, y, common=common):
                a, b = x.astype(TYPES[common]), y.astype(TYPES[common])
                return [(a+b)*(a-b)]
            yield dict(id=f"hidden_mixed_{left}_{right}", family="mixed_inputs", dtype=f"{left}+{right}",
                       code=f"x = {constructor(left,(3,33))}; y = {constructor(right,(33,))}; (x+y)*(x-y)",
                       ref=mixed_ref, knobs="", rtol=6e-3 if common=="f16" else 1e-5,
                       atol=6e-3 if common=="f16" else 1e-6)
    yield dict(id="hidden_multiple_pointwise_outputs_f32", family="multi_outputs", dtype="f32",
               code="x = torch.randn(3, 33); y = torch.randn(3, 33); (x+y, x-y)",
               ref=lambda x,y: [x+y,x-y], knobs="", rtol=0., atol=0.)
    yield dict(id="hidden_multiple_reduction_outputs_f32", family="multi_outputs", dtype="f32",
               code="x = torch.randn(3, 33); y = torch.randn(3, 33); (x.sum(-1), y.mean(-1))",
               ref=lambda x,y: [total(x,-1),total(y,-1,mean=True)],
               knobs="WORK=t64,REDUCE=coop", rtol=1e-4, atol=1e-5)
    yield case("rank4_broadcast", "f32", "x = torch.randn(2, 3, 1, 33); y = torch.randn(1, 3, 5, 1); ",
               "torch.relu(x+y) * (x-y)", lambda x,y: np.maximum(x+y,np.float32(0))*(x-y))


def validate_reference(spec):
    """Check expression semantics with eager CPU PyTorch before Emmy tracing."""
    import torch
    namespace = {"torch": torch}
    setup, expression = spec["code"].rsplit(";", 1)
    # This is our test source, not input from a kernel or external service.
    exec(setup, namespace)
    value = eval(expression, namespace)
    eager = [v.numpy() for v in value] if isinstance(value, (tuple, list)) else [value.numpy()]
    arrays = [namespace[name].numpy() for name in ("x", "y") if name in namespace]
    expected = spec["ref"](*arrays)
    if len(eager) != len(expected):
        raise ValueError("NumPy reference output count differs from eager PyTorch")
    for actual, wanted in zip(eager, expected, strict=True):
        if actual.dtype != wanted.dtype or actual.shape != wanted.shape:
            raise ValueError("NumPy reference dtype/shape differs from eager PyTorch")
        equal = (np.array_equal(actual, wanted) if wanted.dtype.kind == "i" else
                 np.allclose(actual, wanted, rtol=spec["rtol"], atol=spec["atol"]))
        if not equal:
            raise ValueError("NumPy reference disagrees with eager PyTorch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--report-name", default="validation.json")
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name:
        parser.error("report name must be a filename")
    if importlib.metadata.version("emmy-ml") != "0.3.11":
        parser.error("requires pinned emmy-ml==0.3.11")
    specs = [s for s in specifications() if (not args.only or s["id"] in args.only)
             and (not args.families or s["family"] in args.families)]
    if not specs or (args.only and set(args.only)-{s["id"] for s in specs}):
        parser.error("unknown or empty case selection")
    output = ROOT / "review/mvp-hidden"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        folder = ROOT / "tests/mvp-hidden" / spec["id"]
        row = {key: spec[key] for key in ("id", "family", "dtype", "code", "knobs")}
        row["status"] = "expression_reference_failure"
        try:
            validate_reference(spec)
            row["status"] = "upstream_generation_failed"
            if not args.reuse:
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-hidden-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package(spec, dump, folder)
            proof = verify(folder)
            if proof["code"] != spec["code"] or proof["knobs"] != spec["knobs"]:
                raise ValueError("saved fixture differs from requested PyTorch case")
            meta = json.loads((folder / "meta.json").read_text())
            if not meta["kernels"]:
                raise ValueError("upstream produced no runtime kernel")
            row["status"] = "compiler_failure"
            compiled, nodes, operations = {}, Counter(), Counter()
            for k in meta["kernels"]:
                source = load_kernel(folder / f"{k['entry']}.ir.json")
                for stmt in source.kernel.body.iter():
                    nodes[type(stmt).__name__] += 1
                    if type(stmt).__name__ == "Assign":
                        operations[stmt.op.name] += 1
                ptx = emit_ptx(source)
                path = folder / f"{k['entry']}.ours.ptx"
                path.write_text(ptx)
                row["status"] = "assembly_failure"
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[k["entry"]] = ptx
                row["status"] = "compiler_failure"
            row.update(nodes=dict(nodes), operations=dict(operations), status="reference_failure")
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, expected, meta)
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"] = "runner_request_failure"
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
                if row["status"] == "pass" and row["details"]["comparisons"]["kernel_ir"]["status"] != "pass":
                    row["status"] = "gpu_ir_mismatch"
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n"+(exc.stderr or "")[-4000:]
        rows.append(row)
        report = {"pipeline": "PyTorch -> Emmy 0.3.11 -> unchanged Kernel IR -> our PTX",
                  "runner": args.runner, "cases": rows, "statuses": dict(Counter(r["status"] for r in rows)),
                  "source_sha256": {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                    for n in ("compile.py", "ptx_math.py", "ptx_tensor.py", "tools/test_mvp_hidden_variants.py")}}
        (output / args.report_name).write_text(json.dumps(report, indent=2)+"\n")
        print(f"{row['status']:28} {spec['id']}", flush=True)
        if "error" in row:
            print(row["error"][-500:], flush=True)
    return int(any(r["status"] not in ("pass", "assembled_not_executed", "upstream_generation_failed") for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
