# First native tensor-core matmul

Update: [native shared staging](17-shared-memory-async-copies.md) was implemented after this first global-transport increment. It adds cp.async, swizzled staged ldmatrix loads, dynamic shared allocation and passing public m01 coverage. The global-only results and remaining gaps below describe the initial increment.

The compiler now translates a real Emmy f16-input, f32-accumulator tensor-core program into native `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX. This is a working global-memory transport baseline, not shared-memory staging or a tuned GEMM implementation.

## Start with a genuine PyTorch expression

```python
a = torch.randn(16, 16, dtype=torch.float16)
b = torch.randn(16, 8, dtype=torch.float16)
result = a @ b
```

Pinned Emmy 0.3.11 generates the kernel. The test selects its schedule at generation time with:

```text
WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,REDUCE=,STAGE=
```

This requests one warp, one tensor-core cell per warp, no split-K partition and no shared-memory staging. Other fixtures intentionally exercise Emmy's default split-K path. We never change the resulting IR or metadata launch geometry. [tools/test_matmul.py](../../tools/test_matmul.py) packages the actual dump and records the unchanged-IR proof.

The earlier scalar f32 16×16×16 fixture already compiled with our ordinary Loop implementation and passes on the hosted GPU. New emission support is needed for Emmy's tensor-core fragment statements.

## Which code implements the tensor-core operations?

[compile.py](../../compile.py) dispatches the four statement kinds to our [ptx_tensor.py](../../ptx_tensor.py) mixin. Include `compile.py`, `ptx_math.py` and `ptx_tensor.py` at the submission repository root. These are our emitters; no CUDA renderer, nvcc, libdevice, cuBLAS or external compiler translates the input IR.

| Actual Emmy node | Our handling |
| --- | --- |
| RegFragment | Allocate A/B packed b32 registers or C f32 registers; initialize C to zero |
| LdmatrixLoad with staged=False | Gather the operand's half elements from global memory into the specified per-lane fragment layout |
| MmaSyncPtx | Emit one native warp MMA instruction using those register tuples |
| RegStore | Map each lane's four f32 accumulator elements to output coordinates, convert to the declared output dtype and store |

One warp has 32 threads. Together, they produce a 16×8 output tile: 128 elements, four per lane. Each lane holds four packed b32 A registers (eight half elements), two packed b32 B registers (four half elements), and four f32 accumulator registers. Those are the instruction's fragments, not independent scalar matmuls per thread.

The lane mapping follows NVIDIA's [PTX matrix-fragment specification](https://docs.nvidia.com/cuda/parallel-thread-execution/). A lane group is lane//4 and its within-group position is lane%4. C elements occupy rows group/group+8 and the two columns 2*position/2*position+1. A/B gathers pack the corresponding half elements in the instruction's required order.

Despite the Emmy node's name, an unstaged LdmatrixLoad does not emit `ldmatrix`: that instruction reads shared memory. The global path uses `ld.global.b16` and packs half pairs into b32 registers. B may be stored row-major or transposed; `b_trans` changes the gather addresses so the MMA instruction still receives its required column-oriented B fragment.

The compiler's existing runtime StridedLoop drives additional K chunks. Fragment accumulator registers remain bound across those iterations. They are separate from ordinary scalar SSA bindings and do not get reinitialized on each K iteration.

## Masked edges and split-K

M/N edge loads follow Emmy's supplied clamp guards. Coordinates outside a partial output tile read a valid edge element, and the matching RegStore guards prevent writing invalid C elements.

K edges must instead load zero. Duplicating a K element would add an extra product to the result. The emitter initializes each half to zero and predicates its global load on the supplied k_zero bound. It still executes the MMA with all participating lanes.

The default unstaged schedule for the first small matmul chose a split-K decomposition: a tensor-core kernel writes f32 partials to a rank-three intermediate buffer, then an ordinary reduction kernel combines them and stores f16 output. The runner harness chains those kernels using the original metadata. This path passed as well as the explicitly unsplit schedule.

Fragment stores support trailing row/column axes, including leading split-K dimensions. M/N guards predicate each element's store. Fused epilogues, atomic fragment stores, alternative fragment layouts and arrayed fragment families remain explicitly unsupported.

## Independent checks and current results

[tools/tensor_reference.py](../../tools/tensor_reference.py) extends our bounded NumPy IR interpreter independently of the emitter. It represents fragments as numeric element arrays, reconstructs each active warp's A/B/C matrices, computes the matrix product plus C, rounds the result to f32 at the MMA boundary, and applies the actual IR's stores/guards. It checks complete active warps and uninitialized/out-of-bounds accesses through the existing interpreter.

This is a numerical oracle, not a bit-exact model of tensor-core internal summation order. The original NumPy expression reference is separate: a double-precision matmul rounded to the expression's output dtype. Both comparisons use the fixture's declared tolerance. No reference is replaced to manufacture a pass.

[The supported global-transport report](../../review/matmul/global-validation.json) contains nine passing hosted cases: one scalar f32 baseline and eight native tensor-core programs. They cover split/unsplit K, multiple K chunks, irregular 17×13×19 dimensions, multiple output tiles, four cooperating warps, transposed B and a 1×1×1 edge case. All nine match both the original expression and the independent typed-IR reference.

All 43 local test methods passed, including genuine-fixture provenance/assembly, fragment instruction selection and explicit rejection of the generated shared-transport/array-family cases. All 11 existing public pointwise/reduction/normalization GPU cases passed in [the regression report](../../review/native-math/tensor-regression.json). The full historical regression matrix was not rerun for this increment.

Local sm_89 assembly of the single-warp example contains one HMMA instruction, uses 40 physical registers and reports no spills. [Assembly statistics](../../review/matmul/single-warp-assembly.txt) and [disassembly](../../review/matmul/single-warp.sass) are saved. These are local sm_89 observations, not the hosted RTX 5090's final sm_120 machine code or a performance claim.

Run from the repo in the pinned Ubuntu environment:

```bash
python tools/test_matmul.py --runner
python tools/test_matmul.py --reuse --runner
python -m unittest discover -s tests -p 'test_*.py'
```

The default suite selects the supported global-transport subset. `--suite scope` also inventories the unsupported/generated and upstream-rejected probes; it is not a passing regression suite. `--only <case-id>` can isolate a source case, and `--report-name` preserves an exploratory report separately.

Only our compiler PTX is uploaded. The initial probe was committed and pushed before this implementation; initial and extended scope reports remain in review/matmul. All original competition files remain unchanged.

## What is still missing?

- Emmy successfully generates the shared-memory tensor-core probe. It needs CpAsyncCopy/Commit/Wait, swizzled shared addressing and staged ldmatrix loads. This is an in-scope compiler gap and the next transport step.
- The larger f2×f2 output tile generates arrayed RegFragment families and references such as `_a[_r0]`. These are valid Emmy IR, but this first implementation accepts only single fragments. They remain an explicit compiler gap; their generated example is retained and rejection is tested.
- The attempted small scalar asynchronous-copy schedule was rejected by the upstream scheduler. That exact attempted configuration is out of scope; this does not mean Emmy cannot generate asynchronous copies. The shared tensor-core probe and public matmuls already prove it can.
- PyTorch's `.T` spelling failed tracing in this version. The equivalent `transpose(0, 1)` expression generated valid unchanged IR and passes. We changed the source expression, not its IR.
- The supplied public f32 matmuls still require asynchronous copies, additional index operations and, for one, atomic accumulation. They are not claimed as supported by these nine tests. Other MMA shapes/types, bf16/fp8, TMA and WGMMA are not implemented here.

The next priority is the genuine shared-memory/cp.async path, followed by arrayed fragment families and broader datatype/shape coverage. No hidden-set or tuned-performance coverage is implied by the single atom's support.
