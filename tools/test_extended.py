#!/usr/bin/env python3
"""Generate fresh Emmy IR from PyTorch, store NumPy expectations, test our compiler.

CPU generation uses emmy-ml==0.3.11, not the newer ~/emmy checkout. Runner
testing is opt-in and uploads only PTX produced by our own compile.py.
"""

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from expressions_extended import DTYPES, OPS, SIZES, build_specs
from generate import package, trace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submit import RUNNER, post

NP_TYPES = {"f16": np.float16, "f32": np.float32, "f64": np.float64,
            "i32": np.int32, "i64": np.int64}


def validate_numpy(spec):
    """Check reference semantics against eager CPU PyTorch on small typed inputs."""
    import torch
    # The program's random constructors are replaced with known input arrays.
    # Run the final expression with the same x/y inputs as our NumPy reference.
    rng = np.random.default_rng(123)
    shape = tuple(min(d, 7) for d in spec["shape"])
    arrays = [rng.standard_normal(shape).astype(NP_TYPES[spec["dtype"]]) for _ in range(2)]
    # Include meaningful integer values, not just truncated near-zero normals.
    if spec["dtype"].startswith("i"):
        arrays = [rng.integers(-8, 9, shape).astype(NP_TYPES[spec["dtype"]]) for _ in range(2)]
    env = {"torch": torch, "x": torch.from_numpy(arrays[0]), "y": torch.from_numpy(arrays[1])}
    eager = eval(spec["code"].rsplit(";", 1)[-1].strip(), {"__builtins__": {}}, env).numpy()
    expected = spec["ref"](*arrays)[0]
    if eager.dtype != expected.dtype or eager.shape != expected.shape:
        raise ValueError(f"Reference type/shape mismatch: torch={eager.dtype}/{eager.shape}, numpy={expected.dtype}/{expected.shape}")
    # CPU torch reductions have their own finite-precision summation order.
    if not np.allclose(eager, expected, rtol=spec["rtol"], atol=spec["atol"], equal_nan=False):
        raise ValueError("NumPy reference disagrees with CPU PyTorch")


def save_expression(folder, code):
    """Keep the source of each fixture easy to open and run in a notebook."""
    folder.mkdir(parents=True, exist_ok=True)
    setup, expression = code.rsplit(";", 1)
    source = "import torch\n\n" + setup.replace("; ", "\n").strip() + "\n"
    source += "result = " + expression.strip() + "\n"
    (folder / "expression.py").write_text(source)


def validate_fixture(folder, spec, meta):
    """Catch packaging/type inconsistencies before blaming our compiler."""
    arguments = {arg["name"]: arg for kernel in meta["kernels"] for arg in kernel["args"]}
    for name in meta["graph_inputs"]:
        if arguments[name]["dtype"] != spec["dtype"] or arguments[name]["shape"] != spec["shape"]:
            raise ValueError(f"Generated input {name} does not match requested dtype/shape")
    with np.load(folder / "expected.npz") as saved:
        if set(saved.files) != set(meta["graph_outputs"]):
            raise ValueError("Expected output names differ from graph outputs")
        for name in saved.files:
            arg = arguments[name]
            expected = saved[name]
            if expected.dtype != np.dtype(NP_TYPES[arg["dtype"]]) or expected.size != math.prod(arg["shape"]):
                raise ValueError(f"NumPy expectation for {name} disagrees with generated output dtype/size")


def compare_outputs(produced, reference, meta):
    """Compare one set of results without changing the reference's precision."""
    for name, expected in reference.items():
        try:
            got = np.asarray(produced[name], dtype=expected.dtype).reshape(expected.shape)
        except (OverflowError, ValueError) as exc:
            return "runner_output_conversion_failure", {"output": name, "dtype": str(expected.dtype),
                                                        "error": str(exc), "values": produced[name][:5]}
        equal = (got == expected if np.issubdtype(expected.dtype, np.integer)
                 else np.isclose(got, expected, rtol=meta["rtol"], atol=meta["atol"]))
        if not np.all(equal):
            unequal = np.flatnonzero(~equal.ravel())
            examples = [{"index": int(index), "got": got.ravel()[index].item(),
                         "expected": expected.ravel()[index].item()} for index in unequal[:5]]
            return "wrong_result", {"output": name, "dtype": str(expected.dtype),
                                    "mismatch_count": len(unequal), "examples": examples}
    return "pass", {}


def check_runner(folder, meta, ptx_by_entry, ir_reference=None):
    """Preserve f64 and integer types; chain multi-kernel graphs as metadata says."""
    produced = {}
    timings = []
    for kernel in meta["kernels"]:
        assembled = post(RUNNER, "/compile", {"ptx": ptx_by_entry[kernel["entry"]]})
        if not assembled.get("ok"):
            return "runner_assembly_failure", assembled
        arguments = []
        for original in kernel["args"]:
            arg = dict(original)
            if arg.pop("intermediate", False):
                arg.pop("seed", None)
                arg["data"] = produced[arg["name"]]
            arguments.append(arg)
        returned = sum(math.prod(arg["shape"]) for arg in arguments if arg.get("output"))
        for attempt in range(60):
            result = post(RUNNER, "/run", {
                "cubin_id": assembled["cubin_id"], "entry": kernel["entry"],
                "grid": kernel["grid"], "block": kernel["block"],
                "smem_bytes": kernel["smem_bytes"], "zero_outputs": kernel["zero_outputs"],
                "args": arguments, "warmup": 0, "iters": 1, "max_return": max(4096, returned),
            })
            if result.get("stage") != "busy":
                break
            time.sleep(min(2 + attempt, 10))
        if not result.get("ok"):
            return "runner_execution_failure", result
        for name, output in result["outputs"].items():
            if output.get("truncated"):
                return "truncated_output", {"output": name}
            produced[name] = output["values"]
        timings.append({"entry": kernel["entry"], "median_us": result.get("median_us"),
                        "registers": assembled.get("registers")})
    with np.load(folder / "expected.npz") as reference:
        status, details = compare_outputs(produced, reference, meta)
    if ir_reference is not None:
        ir_status, ir_details = compare_outputs(produced, ir_reference, meta)
        return status, {"kernels": timings, "comparisons": {
            "expression": {"status": status, "details": details},
            "kernel_ir": {"status": ir_status, "details": ir_details}}}
    return (status, {"kernels": timings}) if status == "pass" else (status, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(OPS))
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 129])
    parser.add_argument("--rows", nargs="+", type=int, default=[3])
    parser.add_argument("--full-sizes", action="store_true", help="use the full boundary/large-size list")
    parser.add_argument("--work", type=int, choices=(32, 64, 128, 256), default=128)
    parser.add_argument("--runner", action="store_true")
    parser.add_argument("--pace", type=float, default=1, help="seconds between serial hosted cases")
    parser.add_argument("--ir-reference", action="store_true", help="save independent NumPy IR expectations and report both comparisons")
    parser.add_argument("--generate-only", action="store_true", help="generate fixtures without invoking our compiler")
    parser.add_argument("--reuse", action="store_true", help="reuse matching generated fixtures when iterating on the compiler")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=ROOT / "tests/extended")
    parser.add_argument("--report-name", default="results.json", help="report filename within --out")
    args = parser.parse_args()
    if Path(args.report_name).name != args.report_name:
        parser.error("--report-name must be a filename")
    if args.pace < 0:
        parser.error("--pace cannot be negative")
    if args.generate_only and args.runner:
        parser.error("--generate-only and --runner cannot be combined")
    if importlib.metadata.version("emmy-ml") != "0.3.11":
        parser.error("Run with the competition .venv: requires emmy-ml==0.3.11")
    sizes = SIZES if args.full_sizes else args.sizes
    if min((*sizes, *args.rows)) < 1:
        parser.error("All dimensions must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    specs = build_specs(args.dtypes, sizes, args.ops, args.rows, args.work)
    if args.limit is not None:
        specs = specs[:args.limit]
    (args.out / "matrix.json").write_text(json.dumps(
        [{key: value for key, value in spec.items() if key != "ref"} for spec in specs], indent=2))
    provenance = {"emmy": importlib.metadata.version("emmy-ml"),
                  "torch": importlib.metadata.version("torch"), "numpy": np.__version__,
                  "target": "sm_89", "work": args.work, "runner": args.runner}
    rows = []
    emmy = str(Path(sys.executable).parent / "emmy")
    for spec in specs:
        row = {key: spec[key] for key in ("id", "code", "dtype", "operation", "shape")}
        row["status"] = "reference_failure"
        try:
            validate_numpy(spec)
            row["status"] = "generation_failure"
            folder = args.out / spec["id"]
            save_expression(folder, spec["code"])
            provenance_path = folder / "fixture-provenance.json"
            if args.reuse and provenance_path.exists():
                old_provenance = json.loads(provenance_path.read_text())
                old_meta = json.loads((folder / "meta.json").read_text())
                if (any(old_provenance[key] != provenance[key] for key in
                        ("emmy", "torch", "numpy", "target", "work"))
                        or old_meta.get("code") != spec["code"]
                        or not (folder / "expected.npz").exists()
                        or any(not (folder / f"{k['entry']}.ir.json").exists() for k in old_meta["kernels"])):
                    raise ValueError("Existing fixture does not match the selected source/environment; rerun without --reuse")
            else:
                with tempfile.TemporaryDirectory(prefix="emmy-extended-") as temporary:
                    dump = Path(temporary) / "dump"
                    trace(emmy, spec, dump, "sm_89")
                    row["status"] = "packaging_failure"
                    package(spec, dump, folder)
            (folder / "fixture-provenance.json").write_text(json.dumps(provenance, indent=2))
            meta = json.loads((folder / "meta.json").read_text())
            row["status"] = "fixture_validation_failure"
            validate_fixture(folder, spec, meta)
            ir_reference = None
            if args.ir_reference:
                from ir_reference import save_reference
                row["status"] = "ir_reference_failure"
                ir_reference = save_reference(folder)
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(ir_reference, expected, meta)
                row["ir_vs_expression"] = {"status": status, "details": details}
            if args.generate_only:
                row["status"] = "generated_not_tested"
                rows.append(row)
                (args.out / args.report_name).write_text(json.dumps({"provenance": provenance, "cases": rows}, indent=2))
                print(f"{row['status']:32} {spec['id']}", flush=True)
                continue
            ptx_by_entry = {}
            row["status"] = "compiler_unsupported_or_failure"
            for kernel in meta["kernels"]:
                compilation = subprocess.run([sys.executable, str(ROOT / "compile.py"),
                                               str(folder / f"{kernel['entry']}.ir.json")],
                                              capture_output=True, text=True, timeout=60)
                if compilation.returncode:
                    raise RuntimeError(compilation.stderr.strip())
                ptx = folder / f"{kernel['entry']}.ours.ptx"
                ptx.write_text(compilation.stdout)
                ptx_by_entry[kernel["entry"]] = compilation.stdout
                row["status"] = "local_assembly_failure"
                subprocess.run(["ptxas", "-arch=sm_89", str(ptx), "-o", str(ptx.with_suffix(".cubin"))],
                               capture_output=True, text=True, check=True, timeout=35)
                row["status"] = "compiler_unsupported_or_failure"
            row["status"] = "compiled_not_executed"
            if args.runner:
                row["status"] = "runner_request_failure"
                row["status"], row["details"] = check_runner(folder, meta, ptx_by_entry, ir_reference=ir_reference)
                time.sleep(args.pace)  # Courtesy pacing, not a stated server quota.
        except subprocess.CalledProcessError as exc:
            row["error"] = (exc.stderr or str(exc))[-6000:]
        except (Exception, SystemExit) as exc:
            row["error"] = str(exc)[-6000:]
        rows.append(row)
        # Save after every case so interrupted runs retain their findings.
        (args.out / args.report_name).write_text(json.dumps({"provenance": provenance, "cases": rows}, indent=2))
        print(f"{row['status']:32} {spec['id']}", flush=True)
        if "error" in row:
            print(row["error"][-600:], flush=True)
    return int(any(
        row["status"] not in ("pass", "compiled_not_executed", "generated_not_tested")
        or row.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass"
        for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
