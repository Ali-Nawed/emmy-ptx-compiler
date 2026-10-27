"""Validate complete tensor-core kernels emitted by pinned Emmy, never fake IR."""

import json
import shutil
import unittest

import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from compile import emit_ptx
from emmy_fixtures import verify
from test_matmul import specifications
from emmy.compiler.ir.kernel.ir import MmaSyncPtx


class TensorLoweringTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_generated_global_matmuls_assemble_and_match_numpy(self):
        count = 0
        for spec in specifications():
            if spec["path"] in ("async", "mma_shared", "mma_fragments"):
                continue
            count += 1
            folder = ROOT / "tests/matmul" / spec["id"]
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)
        self.assertEqual(count, 9)

    def test_native_mma_is_emitted_for_actual_emmy_nodes(self):
        folder = ROOT / "tests/matmul/matmul_mma_tails_f16_17x13x19"
        verify(folder)
        source = load_kernel(next(folder.glob("*.ir.json")))
        mma_count = sum(isinstance(s, MmaSyncPtx) for s in source.kernel.body.iter())
        self.assertGreater(mma_count, 0)
        ptx = emit_ptx(source)
        self.assertEqual(ptx.count("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"), mma_count)
        self.assertIn("ld.global.b16", ptx)
        self.assertNotIn("ldmatrix.sync", ptx)  # This fixture explicitly selects global transport.

    def test_unsupported_shared_transport_and_fragment_families_fail_explicitly(self):
        for case, message in (("matmul_mma_shared_f16_16x8x16", "CpAsyncCopy"),
                              ("matmul_mma_fragments_f16_32x16x32", "single m16n8k16")):
            folder = ROOT / "tests/matmul" / case
            verify(folder)
            source = load_kernel(next(folder.glob("*.ir.json")))
            with self.assertRaisesRegex(NotImplementedError, message):
                emit_ptx(source)


if __name__ == "__main__":
    unittest.main()
