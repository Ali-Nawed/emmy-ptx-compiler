# Compiler walkthrough

We translate Emmy's scheduled Kernel IR directly into PTX, preserving operations, datatypes and synchronization.

## 1. Our development workflow

```mermaid
flowchart LR
    A[Write PyTorch cases] --> B[Emmy generates Kernel IR]
    B --> C[Our compiler emits PTX]
    B --> R[Emmy reference CUDA]
    R --> N[nvcc reference PTX]
    C --> D[Our PTX on hosted GPU]
    C --> E[Compare PTX + check results]
    N --> E
    D --> E
    E --> F[Fix / extend / optimize]
    F --> A
```

Extend the public baseline by varying **PyTorch operations, shapes, datatypes and schedules**. Keep generated IR unchanged; compare GPU results with independent NumPy and typed-IR references.

Compare against Emmy CUDA → nvcc PTX to identify optimizations. Only our PTX runs on the hosted GPU. Retain regressions and benchmark after correctness.

## 2. How the compiler runs

```mermaid
flowchart LR
    A[Load IR + meta.json] --> B[Emmy typed parser]
    B --> C[Parameters + worker coordinates]
    C --> D[Visit statement]
    D --> E[Resolve operands + allocate registers + emit]
    E --> D
    D -->|finished| F[Prepend declarations + return PTX]
```

`load_kernel()` pairs typed IR with metadata's argument order, buffers and launch requirements. `statement()` visits operations and nested bodies; the GPU executes the emitted instructions later.

`PTXEmitter` inherits `ScalarMath` and `TensorMath`, sharing register counters, maps and instructions without an additional IR.

The file split keeps **scalar numerical algorithms** (`ptx_math.py`) and **warp-level matrix layouts** (`ptx_tensor.py`) separately reviewable. Both are our direct emission code, called by `compile.py`.

Code: [load_kernel](compile.py#L40), [statement](compile.py#L621), [compile](compile.py#L778).

## 3. Worked example: every IR statement → our PTX

### PyTorch and actual Emmy output

```python
x = torch.randn(128, dtype=torch.float16)
result = torch.tanh(x) * 1.5
```

**Actual pinned-Emmy output**; `Tile` is line 1:

```text
Tile[a0] (N=64)
    in1__u0 = load x[((a0 * 2) + 0)]
    f16 v0__u0 = tanh(in1__u0)
    in0__u0 = load mul_c1[0]
    f16 v1__u0 = multiply(in0__u0, v0__u0)
    in1__u1 = load x[((a0 * 2) + 1)]
    f16 v0__u1 = tanh(in1__u1)
    in0__u1 = load mul_c1[0]
    f16 v1__u1 = multiply(in0__u1, v0__u1)
    mul[((a0 * 2) + 0)] = (v1__u0, v1__u1)
```

### A. Create parameters from metadata

| `arg_order` | Buffer | Global address register |
| --- | --- | --- |
| `arg_0` | `mul_c1`: one f16 value, 1.5 | `%rd2` |
| `arg_1` | `x`: 128 f16 values | `%rd4` |
| `arg_2` | `mul`: 128 output values | `%rd6` |

The constant is a buffer. All parameters are **64-bit addresses** to 16-bit elements:

```ptx
.visible .entry k_mul_pointwise(
    .param .u64 arg_0,
    .param .u64 arg_1,
    .param .u64 arg_2
)
```

Load `x`'s pointer; the other pointers follow the same pattern:

```ptx
ld.param.u64 %rd3, [arg_1];
cvta.to.global.u64 %rd4, %rd3;
```

### B. Map the Tile to threads and byte addresses

```ptx
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
setp.ge.u32 %p1, %r4, 64;
@%p1 bra DONE;
```

256 threads launch; 64 workers remain active. `values["a0"] = "%r4"`; worker five handles elements ten/eleven. Tile decoding reuses this coordinate, eliminating divide-by-one and outer remainder.

IR line 2 computes element `2*a0`; f16 elements occupy two bytes:

```ptx
mul.lo.s32 %r5, %r4, 2;
add.s32 %r6, %r5, 0;
mov.s32 %r7, %r6;
mul.wide.s32 %rd7, %r7, 2;
add.u64 %rd8, %rd4, %rd7;
```

`%rd8 = x_base + (2*a0)*2`. Constant loads use index zero.

### C. Visit each body statement and bind its result

Arithmetic bindings below are **after f16 rounding**.

| IR line | Corresponding emitted PTX | `values` binding |
| --- | --- | --- |
| 2: load first `x` | `ld.global.b16 %h1, [%rd8+0];` then `cvt.f32.f16 %f1, %h1;` | `in1__u0 → %f1` |
| 3: first tanh | `tanh.approx.f32 %f3, %f1;` + half rounding | `v0__u0 → %f4` |
| 4: load constant | `ld.global.b16 %h3, [%rd10+0];` then `cvt.f32.f16 %f5, %h3;` | `in0__u0 → %f5` |
| 5: first multiply | `mul.rn.f32 %f6, %f5, %f4;` + half rounding | `v1__u0 → %f7` |
| 6: load second `x` | `ld.global.b16 %h5, [%rd12+0];` then `cvt.f32.f16 %f8, %h5;` | `in1__u1 → %f8` |
| 7: second tanh | `tanh.approx.f32 %f10, %f8;` + half rounding | `v0__u1 → %f11` |
| 8: load constant again | `ld.global.b16 %h7, [%rd14+0];` then `cvt.f32.f16 %f12, %h7;` | `in0__u1 → %f12` |
| 9: second multiply | `mul.rn.f32 %f13, %f12, %f11;` + half rounding | `v1__u1 → %f14` |
| 10: grouped Write | Convert `%f7`/`%f14` to half bits, then `st.global.v2.b16` | Store both results |

`value_dtypes` records **f16** despite `%f` working registers. `pointers` holds buffer bases. `reg(kind)` increments `counts[kind]` for fresh names.

### D. Expand a math operation and preserve rounding

`arithmetic()` calls `ptx_math.py`'s `activation()`, then `round_half()`. IR line 3 becomes:

```ptx
tanh.approx.f32 %f3, %f1;
cvt.rn.f16.f32 %h2, %f3;
cvt.f32.f16 %f4, %h2;
```

Our helper emits **native PTX sequences**. Round to half, then widen: multiplication receives the half-rounded intermediate. The same rounding pattern produces `%f7` after line 5.

**Why `ptx_math.py` here?** The IR operation name alone does not determine a safe instruction sequence:

| Example decision | Code responsible | Reason |
| --- | --- | --- |
| This f16 `tanh` uses native approximate tanh | `ScalarMath.activation()` | It met the half-precision checks; keep the short native path |
| An f32 `tanh` needs a different sequence | `activation()` + `native_math32()` | Native tanh failed strict f32 checks; use an exp-based formula and a near-zero polynomial to avoid cancellation |
| Round before multiplying by 1.5 | Main emitter's `round_half()` | The IR requires an f16 intermediate, even though our working register is f32 |

For small x, subtracting nearly equal numbers in the exp-based tanh formula loses accuracy. The polynomial starts `x - x³/3 + 2x⁵/15`; this dtype/range decision motivates the math helper beyond a simple opcode lookup.

**nvcc comparison:** local CUDA 12.0.140 also expands Emmy's f32 `tanhf` into polynomial FMAs plus exp/reciprocal approximations and corrections. Neither default nor tested fast-math PTX uses `tanh.approx.f32`. This supports choosing sequences by accuracy requirements; our algorithm is independently implemented. [Versioned audit and PTX](review/walkthrough/tanh-nvcc/README.md).

### E. Optimize the store and finalize declarations

```ptx
cvt.rn.f16.f32 %h9, %f7;
cvt.rn.f16.f32 %h10, %f14;
st.global.v2.b16 [%rd16], {%h9, %h10};
```

`%rd16 = output_base + 4*a0`. Proved four-byte alignment allows one paired store. `vector_store()` retains scalar fallback when alignment is unknown.

After emitting the body, counters produce the header declarations:

```ptx
.reg .b32 %r<16>;
.reg .b64 %rd<17>;
.reg .f32 %f<15>;
.reg .b16 %h<11>;
.reg .pred %p<2>;
```

Counters start at one; declarations include zero. Gaps such as `%f2` are unused reserved names. **ptxas allocates physical registers**: this hosted kernel used 14. We implement no liveness pass.

Code: [reg](compile.py#L114), [arithmetic](compile.py#L247), [activation](ptx_math.py#L11), [round_half](compile.py#L414), [address](compile.py#L457), [vector_store](compile.py#L352).

## 4. How this extends to other kernels

### Scalar math: `ptx_math.py`

| Path | Implementation and reasoning |
| --- | --- |
| f16/f32 exp/log | Native instructions are base-two approximations. Split `x = k*ln(2) + r` for exp; use a series near one for log, where its result approaches zero and relative error matters |
| f64 math / floating power | Preserve double precision through the expansion. `exp(y*log(x))` alone misses negative-base/integer-exponent and zero cases, so power needs domain/sign handling |
| Sigmoid / tanh | Evaluate exp on nonpositive arguments for stable activations; use a tanh polynomial near zero. f16 tanh retains the native path above |
| Integer power | Build powers by repeated squaring: `x¹³ = x⁸*x⁴*x`. Handle nonnegative exponents while retaining integer-width behavior |

Math approximations are tested; helper FMA does not fuse arbitrary IR operations. Code: [floating_math](ptx_math.py#L249).

### Reductions and fused operations

For `torch.randn(64, 512).sum(-1)`, 128 workers form partial sums, then cooperate:

| Actual IR operation | Emission |
| --- | --- |
| `for a1 in a1_co..512:128` | Runtime counter, comparison, branch and stride increment |
| `WarpShuffle(acc0, length=32)` | `shfl.sync.bfly.b32` plus the merge program |
| `Sync` + `TreeHalve` | Barrier and shared-memory reduction of four warp partials |

Accumulators carry state; loops preserve scope. Extensions include RMSNorm, L2 normalization and online softmax's coupled maximum/sum merge. Code: [counted_loop](compile.py#L524), [merge_states](compile.py#L489).

### Tensor cores and async copies: `ptx_tensor.py`

#### Same PyTorch expression, two Emmy schedules

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

#### CUDA-core path: each worker runs a dot-product loop

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

#### Tensor-core path: arrange operands, then invoke MMA

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

Code: [fragment allocation](ptx_tensor.py#L93), [matrix_load](ptx_tensor.py#L107), [mma_sync](ptx_tensor.py#L162), [reg_store](ptx_tensor.py#L172).

Matched source artifacts: [scalar IR](tests/walkthrough/matmul_scalar_f16_16x8x16/k_matmul_7421ac.ir.txt), [scalar PTX](tests/walkthrough/matmul_scalar_f16_16x8x16/k_matmul_7421ac.ours.ptx), [tensor-core IR](tests/walkthrough/matmul_tensor_f16_16x8x16/k_matmul_7421ac.ir.txt), [tensor-core PTX](tests/walkthrough/matmul_tensor_f16_16x8x16/k_matmul_7421ac.ours.ptx). Both passed hosted checks against both references in the [matched-path report](review/walkthrough/matmul-paths.json).

## 5. Evidence, reproduction and future work

**Worked example: passes both references. MVP: 8/8.** Matmul suites passed 9 global + 9 shared cases; public normalization, softmax and scalar 512×512 matmul also pass.

[Fresh hidden-style MVP variants](docs/compiler/19-mvp-hidden-coverage.md): 71/71 GPU results match IR, 62/71 match NumPy. Nine expression/precision disagreements and five Emmy generation failures remain recorded; hidden completeness is unproven.

```bash
python tools/generate_walkthrough_example.py --runner
python tools/compare_matmul_paths.py --runner
python tools/test_mvp.py
```

[Actual IR](tests/walkthrough/tanh_scale_f16_128/k_mul_pointwise.ir.txt) · [Full emitted PTX](tests/walkthrough/tanh_scale_f16_128/k_mul_pointwise.ours.ptx) · [Per-statement emission trace and GPU result](review/walkthrough/tanh-scale-emission.json).

Future work: atomic accumulation for m02, larger fragment families and epilogues, end-to-end attention, and further optimization guided by correctness checks and measured performance.
