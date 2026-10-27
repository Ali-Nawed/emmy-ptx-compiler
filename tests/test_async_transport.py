"""Native shared-transport tests using only actual Emmy-generated statements."""

import json
import shutil
import unittest

import numpy as np

from fixture_checks import ROOT, assemble, load_kernel, simulate, compare_outputs
from compile import PTXEmitter, emit_ptx
from ir_reference import Block, DTYPES
from test_matmul import specifications
from emmy.compiler.ir.stmt import StridedLoop
from emmy.compiler.ir.kernel.ir import CpAsyncCopy, CpAsyncCommit, CpAsyncWait, LdmatrixLoad


class AsyncTransportTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_generated_shared_examples_assemble_and_match_expression(self):
        count = 0
        for spec in specifications():
            if spec["suite"] != "shared":
                continue
            folder = ROOT / "tests/matmul" / spec["id"]
            if spec["path"] == "shared_tails":
                self.assertFalse((folder / "emmy-proof.json").exists())
                continue  # This exact source/schedule was rejected upstream.
            count += 1
            with self.subTest(case=folder.name):
                assemble(folder, self, require_proof=True)
                meta = json.loads((folder / "meta.json").read_text())
                with np.load(folder / "expected.npz") as expected:
                    status, details = compare_outputs(simulate(folder), expected, meta)
                    self.assertEqual(status, "pass", details)
        self.assertEqual(count, 9)

    def test_copies_are_not_visible_until_the_actual_commit_and_wait(self):
        folder = ROOT / "tests/matmul/matmul_mma_shared_f16_16x8x16"
        source = load_kernel(next(folder.glob("*.ir.json")))
        descriptor = source.metadata
        buffers = {}
        for arg in descriptor["args"]:
            if "seed" in arg:
                buffers[arg["name"]] = np.random.default_rng(arg["seed"]).standard_normal(arg["shape"]).astype(DTYPES[arg["dtype"]])
        initialized = {name: np.ones(value.shape, bool) for name,value in buffers.items()}
        count = descriptor["block"][0]
        block = Block(count, buffers, initialized)
        mask = np.ones(count, bool)
        tile = source.kernel.body[0]
        for axis, coordinate in zip(tile.axes, np.unravel_index(np.arange(count), tile.extents), strict=True):
            block.bind(axis.name, coordinate.astype(np.int64), mask)
        block.bind("lane", np.arange(count, dtype=np.int64)%32, mask)
        block.bind("warp", np.arange(count, dtype=np.int64)//32, mask)
        outer = next(s for s in tile.body if isinstance(s, StridedLoop))
        for stmt in tile.body:
            if stmt is outer:
                break
            block.statement(stmt, mask)
        block.bind(outer.axis.name, block.expression(outer.start,mask), mask)
        copy = next(s for s in outer.body if isinstance(s,CpAsyncCopy))
        load = next(s for s in outer.body.iter() if isinstance(s,LdmatrixLoad) and s.role == "a")
        inner = next(s for s in outer.body if isinstance(s,StridedLoop) and any(isinstance(t,LdmatrixLoad) for t in s.body))
        block.bind(inner.axis.name, block.expression(inner.start,mask), mask)
        block.statement(copy,mask)
        self.assertFalse(np.any(block.initialized[copy.smem]))
        with self.assertRaisesRegex(ValueError, "uninitialized memory"):
            block.statement(load,mask)
        block.statement(next(s for s in outer.body if isinstance(s,CpAsyncCommit)),mask)
        self.assertFalse(np.any(block.initialized[copy.smem]))
        block.statement(next(s for s in outer.body if isinstance(s,CpAsyncWait)),mask)
        self.assertTrue(np.all(block.initialized[copy.smem]))
        block.statement(load,mask)

    def test_native_transport_and_pipeline_waits_are_emitted(self):
        folder = ROOT / "tests/matmul/matmul_shared_depth3_f16_16x8x128"
        ptx = emit_ptx(load_kernel(next(folder.glob("*.ir.json"))))
        self.assertIn("cp.async.cg.shared.global", ptx)
        self.assertIn("cp.async.commit_group;", ptx)
        self.assertIn("cp.async.wait_group 2;", ptx)
        self.assertIn("ldmatrix.sync.aligned.m8n8.x4.shared.b16", ptx)
        self.assertIn("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16", ptx)
        self.assertIn("bar.sync", ptx)

    @unittest.skipUnless(shutil.which("ptxas"), "requires Ubuntu CUDA ptxas")
    def test_public_dynamic_shared_budget_is_used_without_static_duplication(self):
        folder = ROOT / "problems/m01_matmul_512"
        emitted = assemble(folder,self)
        self.assertEqual(len(emitted),1)
        source, ptx = emitted[0]
        emitter = PTXEmitter(source)
        emitter.compile()
        self.assertEqual(emitter.dynamic_shared_offset, source.metadata["smem_bytes"])
        self.assertIn(".extern .shared", ptx)
        self.assertNotIn(".b8 scratch_", ptx)


if __name__ == "__main__":
    unittest.main()
