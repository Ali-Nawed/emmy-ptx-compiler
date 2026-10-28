#!/usr/bin/env python3
"""Generate and trace the walkthrough's real PyTorch -> Emmy -> PTX example."""

import argparse
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
from compile import PTXEmitter, emit_ptx, load_kernel
from ir_reference import simulate


def expected(x):
    # PyTorch rounds the tanh result to f16 before the multiplication.
    return [np.tanh(x.astype(np.float64)).astype(np.float16) * np.float16(1.5)]


SPEC = dict(id="tanh_scale_f16_128",
            code="x = torch.randn(128, dtype=torch.float16); torch.tanh(x) * 1.5",
            ref=expected, rtol=2e-3, atol=0.)


class RecordedEmitter(PTXEmitter):
    """Record existing emission decisions; never alter or synthesize input IR."""

    def __init__(self, source):
        super().__init__(source)
        self.events = []

    def statement(self, stmt):
        start = len(self.lines)
        before = self.values.copy()
        super().statement(stmt)
        self.events.append({"node": type(stmt).__name__, "start": start, "end": len(self.lines),
                            "instructions": self.lines[start:],
                            "bindings": {name: {"register": register, "dtype": self.value_dtypes.get(name)}
                                         for name, register in self.values.items()
                                         if before.get(name) != register}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("emmy-ml") != "0.3.11":
        parser.error("requires pinned emmy-ml==0.3.11")
    import torch
    x = np.random.default_rng(0).standard_normal(128).astype(np.float16)
    np.testing.assert_allclose((torch.tanh(torch.from_numpy(x))*1.5).numpy(), expected(x)[0],
                               rtol=SPEC["rtol"], atol=SPEC["atol"])
    folder = ROOT / "tests/walkthrough" / SPEC["id"]
    if not args.reuse:
        save_expression(folder, SPEC["code"])
        with tempfile.TemporaryDirectory(prefix="emmy-walkthrough-") as temporary:
            dump = Path(temporary) / "dump"
            trace(str(Path(sys.executable).parent / "emmy"), SPEC, dump, "sm_89")
            package(SPEC, dump, folder)
    proof = verify(folder)
    if proof["code"] != SPEC["code"] or proof["knobs"]:
        raise ValueError("fixture source/settings differ from walkthrough")
    meta = json.loads((folder / "meta.json").read_text())
    descriptor, = meta["kernels"]
    entry = descriptor["entry"]
    source = load_kernel(folder / f"{entry}.ir.json")
    emitter = RecordedEmitter(source)
    ptx = emitter.compile()
    # The recorder must produce exactly the submission compiler's output.
    if ptx != emit_ptx(source):
        raise ValueError("recording changed compiler output")
    ir_lines = (folder / f"{entry}.ir.txt").read_text().splitlines()
    if len(ir_lines) != len(emitter.events)+1:
        raise ValueError("walkthrough expects one flat Tile body")
    prefix = ptx.splitlines().index(emitter.lines[0])
    for number, (event, ir_line) in enumerate(zip(emitter.events, ir_lines[1:], strict=True), 2):
        event.update(ir_line=number, ir=ir_line.strip(),
                     ptx_lines=[prefix+event.pop("start")+1, prefix+event.pop("end")])
    path = folder / f"{entry}.ours.ptx"
    path.write_text(ptx)
    subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                   check=True, capture_output=True, text=True, timeout=60)
    ir_reference = simulate(folder)
    with np.load(folder / "expected.npz") as reference:
        ir_status, ir_details = compare_outputs(ir_reference, reference, meta)
    report = {"pipeline": "PyTorch -> Emmy 0.3.11 -> unchanged IR -> our compiler",
              "status": "assembled_not_executed", "ir_vs_expression": ir_status,
              "ir_comparison_details": ir_details, "metadata": descriptor,
              "pointers": emitter.pointers, "register_counts": emitter.counts,
              "statements": emitter.events,
              "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                for name in ("compile.py", "ptx_math.py", "ptx_tensor.py")},
              "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest()}
    if args.runner:
        report["status"], report["details"] = check_runner(folder, meta, {entry: ptx}, ir_reference)
    output = ROOT / "review/walkthrough"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tanh-scale-emission.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "ir_vs_expression": ir_status,
                      "statement_count": len(emitter.events), "register_counts": emitter.counts}))
    return int(ir_status != "pass" or report["status"] not in ("pass", "assembled_not_executed")
               or report.get("details", {}).get("comparisons", {}).get("kernel_ir", {}).get("status", "pass") != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
