#!/usr/bin/env python3
"""Compare the new emitter's local SASS with our saved pre-optimization PTX.

Both sides are our compiler. The old reference-comparison artifacts remain
unchanged. This tool neither invokes nvcc nor makes runner requests.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from compile import emit_ptx, load_kernel
from compare_emmy_ptx import assemble, ptx_stats


def main():
    baseline = json.loads((ROOT / "review/emmy-ptx-comparison/results.json").read_text())
    output = ROOT / "review/emitter-optimizations"
    rows = []
    for original in baseline["cases"]:
        fixture = ROOT / original["fixture"]
        ir = fixture / f"{original['entry']}.ir.json"
        if hashlib.sha256(ir.read_bytes()).hexdigest() != original["source_sha256"][ir.name]:
            raise ValueError("Comparison fixture changed")
        destination = output / "assembly" / original["case"]
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "ours.ptx"
        source = emit_ptx(load_kernel(ir))
        path.write_text(source)
        after = {**ptx_stats(source), **assemble(path)}
        if after["spill_load_bytes"] or after["spill_store_bytes"]:
            raise ValueError("Optimization introduced a spill")
        row = {"case": original["case"], "fixture": original["fixture"],
               "before": original["variants"]["ours"], "after": after}
        rows.append(row)
        print(json.dumps({"case": row["case"], "registers": [row["before"]["registers"], after["registers"]],
                          "non_nop_sass": [row["before"]["sass_non_nop_count"], after["sass_non_nop_count"]],
                          "reciprocals": [row["before"]["sass_opcodes"].get("MUFU.RCP", 0),
                                          after["sass_opcodes"].get("MUFU.RCP", 0)]}), flush=True)
    (output / "assembly-results.json").write_text(json.dumps({
        "target": "sm_89", "tool_versions": baseline["tool_versions"],
        "compiler_sha256": hashlib.sha256((ROOT / "compile.py").read_bytes()).hexdigest(),
        "performance_measured": False, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
