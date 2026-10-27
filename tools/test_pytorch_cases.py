#!/usr/bin/env python3
"""Conversion/range coverage using only real PyTorch -> Emmy Kernel IR.

Shapes, datatypes and explicit range inputs are written in expression.py before
tracing. No Kernel IR node is constructed or changed by this tool.
"""

import argparse
from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from emmy_fixtures import package, verify
from generate import trace
from ir_reference import simulate
from test_extended import ROOT, check_runner, compare_outputs, save_expression, validate_numpy
from compile import emit_ptx, load_kernel

DTYPES = {"f16": "float16", "f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}


def specifications(families):
    if "conversions" in families:
        for source, torch_source in DTYPES.items():
            for target, torch_target in DTYPES.items():
                ctor = (f"torch.randint(-8, 9, (129,), dtype=torch.{torch_source})" if source.startswith("i")
                        else f"torch.randn(129, dtype=torch.{torch_source})")
                yield dict(id=f"convert_{source}_to_{target}", family="conversions", dtype=source, shape=[129],
                           code=f"x = {ctor}; x.to(torch.{torch_target})",
                           ref=lambda x, *unused, target=torch_target: [x.astype(target)], rtol=0, atol=0)
    if "ranges" in families:
        for dtype, torch_type in list(DTYPES.items())[:3]:
            for operation in ("exp", "log", "power"):
                if operation == "exp":
                    limit = {"f16": 10, "f32": 80, "f64": 700}[dtype]
                    x = np.linspace(-limit, limit, 257).astype(torch_type)
                elif operation == "log":
                    limits = {"f16": (-24, 15), "f32": (-149, 127), "f64": (-1074, 1023)}[dtype]
                    x = np.exp2(np.linspace(*limits, 257)).astype(torch_type)
                    x[128] = 1 + np.finfo(torch_type).eps
                else:
                    x = np.resize(np.array([-2., -1., -.5, .1, 1., 2., 10.], dtype=torch_type), 257)
                inputs = [x]
                code = f"x = torch.tensor({x.tolist()!r}, dtype=torch.{torch_type}); "
                if operation == "power":
                    y = np.resize(np.array([-3., -2., -1., 0., 1., 2., 3.], dtype=torch_type), len(x))
                    inputs.append(y)
                    code += f"y = torch.tensor({y.tolist()!r}, dtype=torch.{torch_type}); "
                tail = "torch.pow(x, y)" if operation == "power" else f"torch.{operation}(x)"
                function = np.power if operation == "power" else getattr(np, operation)
                yield dict(id=f"range_{operation}_{dtype}", family="ranges", dtype=dtype, shape=[257],
                           code=code+tail, ref=lambda *arrays, fn=function: [fn(*arrays)], inputs=inputs,
                           rtol={"f16": 2e-3, "f32": 2e-6, "f64": 4e-13}[dtype], atol=0)
    if "native_edges" in families:
        for dtype, numpy_type in (("f16", np.float16), ("f32", np.float32)):
            if dtype == "f32":
                lower = np.float32(-103.97208)
                low_values = [lower]
                for direction in (np.float32(-np.inf), np.float32(np.inf)):
                    value = lower
                    for _ in range(20):
                        value = np.nextafter(value, direction)
                        low_values.append(value)
                upper = np.float32(88.722831725)
                high_values = [upper]
                for _ in range(20):
                    upper = np.nextafter(upper, np.float32(-np.inf))
                    high_values.append(upper)
                exponents = np.array([*low_values, -100, -90, -87.33655, 0, 80, *high_values], numpy_type)
            else:
                exponents = np.array([-17.375, -17.34375, -17.328125, -17.3125, -17.28125,
                                      -16, -10, 0, 10, 11, 11.078125, 11.0859375], numpy_type)
            around_one = [numpy_type(1.)]
            for direction in (numpy_type(0.), numpy_type(2.)):
                value = numpy_type(1.)
                for _ in range(32):
                    value = np.nextafter(value, direction)
                    around_one.append(value)
            logarithms = np.array([*around_one, .5, .75, 1.25, 2,
                                  np.nextafter(numpy_type(0.), numpy_type(1.)), np.finfo(numpy_type).tiny], numpy_type)
            composed = np.array([-10, -1, -.1, -.01, -.001, 0, .001, .01, .1, 1, 10], numpy_type)
            for name, values, tail, fn in (
                    ("exp_edges", exponents, "torch.exp(x)", np.exp),
                    ("log_edges", logarithms, "torch.log(x)", np.log),
                    ("log_exp", composed, "torch.log(torch.exp(x))", lambda x: np.log(np.exp(x)))):
                torch_type = DTYPES[dtype]
                code = f"x = torch.tensor({values.tolist()!r}, dtype=torch.{torch_type}); {tail}"
                yield dict(id=f"native_{name}_{dtype}", family="native_edges", dtype=dtype, shape=list(values.shape),
                           code=code, inputs=[values], ref=lambda *arrays, fn=fn: [fn(*arrays)],
                           rtol=2e-3 if dtype == "f16" else 2e-6,
                           atol=(2e-3 if dtype == "f16" else 1e-6) if name == "log_exp" else 0)


def explicit_inputs(folder, spec):
    """Use the exact tensors from the PyTorch setup; change no IR/launch fields."""
    import torch
    arrays = spec["inputs"]
    expected = spec["ref"](*arrays)
    namespace = {"torch": torch, **dict(zip(("x", "y"), (torch.from_numpy(a) for a in arrays)))}
    eager = eval(spec["code"].rsplit(";", 1)[-1].strip(), {"__builtins__": {}}, namespace).numpy()
    np.testing.assert_allclose(eager, expected[0], rtol=spec["rtol"], atol=spec["atol"])
    meta = json.loads((folder / "meta.json").read_text())
    named = dict(zip(meta["graph_inputs"], arrays, strict=True))
    for descriptor in meta["kernels"]:
        for arg in descriptor["args"]:
            if arg["name"] in named:
                array = named[arg["name"]]
                if list(array.shape) != arg["shape"] or array.dtype != np.dtype(DTYPES[arg["dtype"]]):
                    raise ValueError("PyTorch source input shape/type disagrees with Emmy metadata")
                arg.pop("seed", None)
                arg["data"] = array.ravel().tolist()
    (folder / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
    np.savez(folder / "expected.npz", **dict(zip(meta["graph_outputs"], expected, strict=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=("conversions", "ranges", "native_edges"), default=["conversions", "ranges"])
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    report_dir = ROOT / ("review/native-math/edges" if args.families == ["native_edges"] else "review/pytorch-only")
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specifications(args.families):
        fixture_family = "native-math-edges" if spec["family"] == "native_edges" else "pytorch-only"
        folder = ROOT / "tests" / fixture_family / spec["id"]
        row = {"case": spec["id"], "status": "generation_failure"}
        try:
            if args.reuse:
                verify(folder)
            else:
                if "inputs" not in spec:
                    validate_numpy(spec)
                save_expression(folder, spec["code"])
                with tempfile.TemporaryDirectory(prefix="emmy-source-only-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(str(Path(sys.executable).parent / "emmy"), spec, dump, "sm_89")
                    package_spec = spec
                    if "inputs" in spec:
                        package_spec = {**spec, "ref": lambda *unused: spec["ref"](*spec["inputs"])}
                    package(package_spec, dump, folder)
                if "inputs" in spec:
                    explicit_inputs(folder, spec)
                verify(folder)
            row["status"] = "reference_failure"
            meta = json.loads((folder / "meta.json").read_text())
            reference = simulate(folder)
            with np.load(folder / "expected.npz") as expected:
                row["ir_vs_expression"], row["reference_details"] = compare_outputs(reference, expected, meta)
            row["status"] = "compiler_failure"
            compiled = {}
            for descriptor in meta["kernels"]:
                path = folder / f"{descriptor['entry']}.ours.ptx"
                ptx = emit_ptx(load_kernel(folder / f"{descriptor['entry']}.ir.json"))
                path.write_text(ptx)
                subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                               check=True, capture_output=True, text=True, timeout=60)
                compiled[descriptor["entry"]] = ptx
            row["status"] = "assembled_not_executed"
            if args.runner:
                row["status"], row["details"] = check_runner(folder, meta, compiled, reference)
                time.sleep(1)
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                row["error"] += "\n" + (exc.stderr or "")[-3000:]
        rows.append(row)
        report = {"pipeline": "PyTorch -> Emmy -> unmodified Kernel IR -> our compiler", "runner": args.runner,
                  "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ("compile.py", "ptx_math.py")},
                  "cases": rows, "statuses": dict(Counter(r["status"] for r in rows))}
        (report_dir / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(f"{row['status']:26} {spec['id']}", flush=True)
        if "error" in row:
            print(row["error"][-700:], flush=True)
    return int(any(row["status"] not in ("pass", "assembled_not_executed") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
