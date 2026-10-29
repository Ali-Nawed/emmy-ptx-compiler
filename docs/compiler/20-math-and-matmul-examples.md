# Math and matmul examples

Detailed reference for the [compiler walkthrough](../../COMPILER_WALKTHROUGH.md).

## Scalar math: `ptx_math.py`

| Path | Implementation and reasoning |
| --- | --- |
| f16/f32 exp/log | Native instructions are base-two approximations. Split `x = k*ln(2) + r` for exp; use a series near one for log, where its result approaches zero and relative error matters |
| f64 math / floating power | Preserve double precision through the expansion. `exp(y*log(x))` alone misses negative-base/integer-exponent and zero cases, so power needs domain/sign handling |
| Sigmoid / tanh | Evaluate exp on nonpositive arguments for stable activations; use a tanh polynomial near zero. f16 tanh retains the native path above |
| Integer power | Build powers by repeated squaring: `x¹³ = x⁸*x⁴*x`. Handle nonnegative exponents while retaining integer-width behavior |

Math approximations are tested; helper FMA does not fuse arbitrary IR operations. Code: [floating_math](../../ptx_math.py#L249).

## Reductions and fused operations

For `torch.randn(64, 512).sum(-1)`, 128 workers form partial sums, then cooperate:

| Actual IR operation | Emission |
| --- | --- |
| `for a1 in a1_co..512:128` | Runtime counter, comparison, branch and stride increment |
| `WarpShuffle(acc0, length=32)` | `shfl.sync.bfly.b32` plus the merge program |
| `Sync` + `TreeHalve` | Barrier and shared-memory reduction of four warp partials |

Accumulators carry state; loops preserve scope. Extensions include RMSNorm, L2 normalization and online softmax's coupled maximum/sum merge. Code: [counted_loop](../../compile.py#L524), [merge_states](../../compile.py#L489).

## Tensor cores and async copies: `ptx_tensor.py`

### Same PyTorch expression, two Emmy schedules

Both examples use identical f16 inputs and dimensions:

```python
a = torch.randn(16, 16, dtype=torch.float16)
b = torch.randn(16, 8, dtype=torch.float16)
result = a @ b
```

`a @ b` stays unchanged; the fixture generator sets `EMMY_KNOBS` before tracing. The schedule selects the path; f16 alone does not. Our compiler follows the generated IR.

| Path | Emmy schedule used to generate the fixture |
| --- | --- |
| CUDA-core scalar | `WORK=t8x4,TILE=f1x1,REDUCE=,STAGE=` |
| Tensor-core, shared staging | `WORK=w1x1,TILE=mma_m16n8k16_f16_f32/f1x1,STAGE=d1/smem-async` |

### CUDA-core path: each worker runs a dot-product loop

Actual scalar Kernel IR:

```text
Tile[a0, a1] (N=128)
    for a2 in 0..16
        in0 = load a[a0, a2]
        in1 = load b[a2, a1]
        f16 v0 = multiply(in0, in1)
        f32 acc0 <- add(acc0, v0)
    matmul[a0, a1] = acc0
```

Each worker accumulates one output across K. Selected actual PTX, omitting indexing and loop control:

```ptx
ld.global.b16 %h1, [%rd8+0];
cvt.f32.f16 %f2, %h1;
ld.global.b16 %h2, [%rd10+0];
cvt.f32.f16 %f3, %h2;
mul.rn.f32 %f4, %f2, %f3;
cvt.rn.f16.f32 %h3, %f4;
cvt.f32.f16 %f5, %h3;
add.rn.f32 %f6, %f1, %f5;
mov.f32 %f1, %f6;
```

This scalar IR rounds each product to f16 before f32 accumulation. Our emitter uses ordinary floating-point instructions; other CUDA-core implementations can use scalar FMA.

| Difference | This scalar example | This tensor-core example |
| --- | --- | --- |
| Work ownership | 128 active workers, one output each; metadata launches 256 threads with a guard | 32 threads jointly compute 128 outputs, four per lane |
| Compute IR | `Loop`, `Load`, `Assign`, `Accum`, `Write` | `RegFragment`, `LdmatrixLoad`, `MmaSyncPtx`, `RegStore` |
| Arithmetic PTX | Scalar `mul.rn.f32` and `add.rn.f32` in a K loop | Warp-cooperative `mma.sync.m16n8k16` |
| Memory in these schedules | Direct global loads | Shared buffers, `cp.async`, commit/wait, barriers and `ldmatrix` |
| Numeric contract | Explicit f16 product rounding, f32 accumulator | Native f16-input/f32-accumulation MMA semantics |

**`mma.sync` selects tensor cores; `cp.async` is optional transport.** Scalar kernels can stage too, and MMA also has a global-load path. Numeric contracts can differ; both passed IR/NumPy checks within recorded tolerances. Runtime speedup is unproven.

### Tensor-core path: arrange operands, then invoke MMA

Actual tensor-core IR excerpts; copy indices and loops omitted:

```text
RegFragment a:f16 _a0[4] (16x8x16)
RegFragment b:f16 _b0[2] (16x8x16)
RegFragment c:f32 _c0_0[4] (16x8x16)
Smem __half _a_smem[16, 16] align=256
Smem __half _b_smem[16, 8] align=16
LdmatrixLoad _a0 <- _a_smem[((a0_u * 16) + 0), _ki] (x4, ldm=16)
LdmatrixLoad _b0 <- _b_smem[_ki, ((a1_u * 8) + 0)] (x2.trans, ldm=8)
MmaSyncPtx _c0_0 += _a0 @ _b0 (m16n8k16 f16)
```

The staged data path follows:

```text
CpAsyncCopy → shared memory → LdmatrixLoad → MmaSyncPtx → RegStore
```

Selected **actual emitted instructions**, omitting address setup, loop control and output stores:

```ptx
cp.async.cg.shared.global [%r39], [%rd12], 16;
cp.async.cg.shared.global [%r72], [%rd16], 16;
cp.async.commit_group;
cp.async.wait_group 0;
bar.sync 0;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r12, %r13, %r14, %r15}, [%r97];
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r16, %r17}, [%r105];
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1, %f2, %f3, %f4}, {%r12, %r13, %r14, %r15}, {%r16, %r17}, {%f1, %f2, %f3, %f4};
bar.sync 0;
```

| Step | What happens and why |
| --- | --- |
| Two `cp.async` sites | `cp_async_copy()` checks copy size, spaces and alignment; `shared_slab_address()` applies the IR's swizzle. Copies and matrix reads must agree on physical placement |
| Commit → wait → barrier | The main visitor emits the IR's ordering: complete each thread's copies, then coordinate the block before cooperative reads |
| Two `ldmatrix` instructions | `staged_matrix_load()` supplies the packed A/B register layout expected by MMA; B's transpose arranges that operand layout |
| `mma.sync` | `mma_sync()` checks fragment roles/types and emits the warp-cooperative multiply/accumulate; scalar `Assign` lowering cannot supply these operand tuples |
| Final barrier → `RegStore` | Protect buffer reuse; `reg_store()` maps the distributed accumulator back to output coordinates |

`fragments[name]` tracks role, dtype and registers. Per thread: four packed A registers, two B and four f32 C; native m16n8k16 `mma.sync` computes cooperatively across the warp.

**Why `ptx_tensor.py` here?** Lane zero's four accumulator registers belong to `C[0,0]`, `C[0,1]`, `C[8,0]`, `C[8,1]`. Storing all four consecutively would target the wrong row. The helper implements this hardware-specific layout and packing; the main compiler continues to handle parameters, control flow and shared register allocation.

Extensions include split-K, transposed B, multiple warps and buffering.

Code: [fragment allocation](../../ptx_tensor.py#L93), [matrix_load](../../ptx_tensor.py#L107), [mma_sync](../../ptx_tensor.py#L162), [reg_store](../../ptx_tensor.py#L172).

Matched source artifacts: [scalar IR](../../tests/walkthrough/matmul_scalar_f16_16x8x16/k_matmul_7421ac.ir.txt), [scalar PTX](../../tests/walkthrough/matmul_scalar_f16_16x8x16/k_matmul_7421ac.ours.ptx), [tensor-core IR](../../tests/walkthrough/matmul_tensor_f16_16x8x16/k_matmul_7421ac.ir.txt), [tensor-core PTX](../../tests/walkthrough/matmul_tensor_f16_16x8x16/k_matmul_7421ac.ours.ptx). Both passed hosted checks against both references in the [matched-path report](../../review/walkthrough/matmul-paths.json).
