"""Package real PyTorch/Emmy dumps and attest their unchanged Kernel IR bodies.

Never construct or rewrite Kernel IR here. The organizer's packager supplies
metadata and only normalizes an empty kernel entry name to its launch name.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from generate import package as organizer_package


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def package(spec, dump, folder):
    dump, folder = Path(dump), Path(folder)
    kernel_stage = next(dump.glob("*_lowering_kernel.json"))
    source_dir = kernel_stage.with_suffix(".kernels")
    result = organizer_package(spec, dump, folder)
    meta = json.loads((folder / "meta.json").read_text())
    records = {}
    for descriptor in meta["kernels"]:
        original = source_dir / f"{descriptor['node']}.json"
        if not original.exists():
            original = source_dir / f"{descriptor['entry']}.json"
        upstream = json.loads(original.read_text())
        packaged = folder / f"{descriptor['entry']}.ir.json"
        # Compare the complete upstream payload after ONLY the organizer's
        # required entry-name normalization. Body/schedule/types are untouched.
        kernels = [node for node in upstream["nodes"].values() if node["op"] == "KernelOp"]
        if len(kernels) != 1:
            raise ValueError("Expected one upstream Emmy KernelOp")
        kernels[0]["op_fields"]["name"] = descriptor["entry"]
        if json.loads(packaged.read_text()) != upstream:
            raise ValueError("Packaged IR differs from Emmy output beyond entry-name normalization")
        records[packaged.name] = {"sha256": digest(packaged), "upstream_dump_sha256": digest(original)}
    proof = {"pipeline": "PyTorch expression -> Emmy -> unchanged Kernel IR -> our compiler",
             "code": spec["code"], "knobs": spec.get("knobs", ""), "target": "sm_89",
             "versions": {name: importlib.metadata.version(name) for name in ("emmy-ml", "torch", "numpy")},
             "entry_name_normalization": "organizer packager only; no body/type/shape/schedule rewriting",
             "kernels": records}
    (folder / "emmy-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
    return result


def verify(folder):
    """Reject altered/new unattested fixtures; don't invent proof for old ones."""
    folder = Path(folder)
    path = folder / "emmy-proof.json"
    if not path.exists():
        raise ValueError(f"Missing Emmy generation proof: {folder}; regenerate from its PyTorch expression")
    proof = json.loads(path.read_text())
    meta = json.loads((folder / "meta.json").read_text())
    if meta.get("code") != proof["code"]:
        raise ValueError("Fixture source differs from its Emmy generation proof")
    expected_files = {f"{kernel['entry']}.ir.json" for kernel in meta["kernels"]}
    if expected_files != set(proof["kernels"]):
        raise ValueError("Fixture kernel set differs from Emmy proof")
    for name, record in proof["kernels"].items():
        if digest(folder / name) != record["sha256"]:
            raise ValueError(f"Kernel IR changed after Emmy generation: {name}")
    return proof
