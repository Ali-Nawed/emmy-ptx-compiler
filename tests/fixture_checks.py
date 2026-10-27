"""Shared checks on complete, unmodified Emmy-generated kernel fixtures."""

import json
from pathlib import Path
import subprocess
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from compile import emit_ptx, load_kernel
from emmy_fixtures import verify
from ir_reference import simulate
from test_extended import compare_outputs


def assemble(folder, check, require_proof=False):
    folder = Path(folder)
    if require_proof:
        verify(folder)
    meta = json.loads((folder / "meta.json").read_text())
    emitted = []
    for descriptor in meta["kernels"]:
        source = load_kernel(folder / f"{descriptor['entry']}.ir.json")
        ptx = emit_ptx(source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "kernel.ptx"
            path.write_text(ptx)
            result = subprocess.run(["ptxas", "-arch=sm_89", str(path), "-o", str(path.with_suffix(".cubin"))],
                                    capture_output=True, text=True, timeout=60)
            check.assertEqual(result.returncode, 0, result.stderr)
        emitted.append((source, ptx))
    return emitted
