# Compiler walkthrough

## 1. Our development workflow

Start with a naive compiler for the public cases, then **generate more cases to expose missing features, bugs and optimization opportunities**.

```mermaid
flowchart LR
    A[Write PyTorch cases] --> B[Emmy generates Kernel IR]
    B --> C[Our compiler emits PTX]
    B --> R[Emmy emits reference CUDA]
    R --> N[nvcc emits reference PTX]
    C --> D[Assemble + run on GPU]
    C --> E[Compare PTX + check GPU results]
    D --> E
    N --> E
    E --> F[Fix / extend / optimize]
    F --> A
```

Vary operations, sizes, datatypes and schedules **before tracing through pinned Emmy**. Use its unchanged IR; compare GPU output with independent NumPy results and a typed-IR interpreter.

Compare **our PTX against Emmy CUDA → nvcc PTX**: instruction selection, address arithmetic, scalar/vector memory operations and conversions. Use the differences to guide emitter improvements, then validate correctness and measure on the GPU. Only our compiler's PTX goes to the hosted runner; fewer PTX instructions alone do not prove a speedup.

| What the test reveals | Our next step |
| --- | --- |
| Unsupported generated IR | Add the missing PTX emission feature |
| Assembly failure or output mismatch | Trace IR → PTX, isolate the bug and retain a regression case |
| Correct output with redundant or inefficient PTX | Optimize, rerun correctness checks and benchmark |

Examples: strict f32 tanh failures led to a math fix without relaxing tolerance; grouped writes led to aligned vector stores. New cases exercise each change and repeat the loop.

## 2. How the compiler runs

Our compiler translates **scheduled Emmy Kernel IR directly into PTX**, preserving datatypes and launch metadata.

```mermaid
flowchart LR
    A[Load IR + metadata] --> B[Parse with Emmy]
    B --> C[Visit statement]
    C --> D[Resolve operands + allocate virtual registers]
    D --> G[Emit typed PTX]
    G --> C
    C -->|finished| E[Prepend register declarations + return PTX]
```

`statement()` dispatches operations and recursively visits loop bodies. Python generates instructions; the GPU executes them later.

Code: [load_kernel](compile.py#L40) → [compile](compile.py#L778) → [statement](compile.py#L621).

### Register allocation and tracking

| Mechanism | What it tracks |
| --- | --- |
| `counts` + `reg(kind)` | A counter per register bank; `reg("f")` returns a fresh `%f1`, `%f2`, … |
| `values` | IR name → current PTX register or immediate, e.g. `v0__u0` → `%f3` |
| `value_dtypes` | IR name → numeric type, so operands receive the correct conversions |
| `pointers` / `shared` | Buffer name → global/shared address register |

`data_reg(dtype)` selects the bank: f32 uses `%f`, f64 `%d`, i32 `%r`, i64 `%l`, and predicates `%p`. Addresses use `%rd`; f16 uses rounded f32 working values and `%h` for half bits.

For `v0 = add(a, b)`, look up `a`/`b`, allocate a result, emit `add`, then bind `v0` to that register and type. Loop scopes save/restore bindings; accumulators update their carried registers.

**Emit the body first, then prepend declarations from the counters.** Three allocated f32 names produce `.reg .f32 %f<4>;`: names start at one, leaving `%f0` unused. These are virtual names; **ptxas performs physical register allocation**. We do not run liveness analysis.

Code: [reg](compile.py#L114), [data_reg](compile.py#L133), [declarations](compile.py#L826).

## 3. Features: PyTorch → IR → PTX

### Pointwise: two elements per worker

```python
result = torch.randn(128) + torch.randn(128)
```

| Actual IR excerpt | Corresponding PTX excerpt |
| --- | --- |
| `f32 v0__u0 = add(in0__u0, in1__u0)` | `add.rn.f32 %f3, %f1, %f2;` |
| `add[((a0 * 2) + 0)] = (v0__u0, v0__u1)` | `st.global.v2.f32 [%rd16], {%f3, %f6};` |

The `N=64` tile covers 128 elements. Each worker computes two adds, then writes both together when alignment permits. Loads, guards and the second add are omitted above.

**Beyond the baseline:** f16/f32/f64/i32/i64, conversions, composed arithmetic, comparisons and `where`.

Code: [arithmetic](compile.py#L247), [vector_store](compile.py#L352).

### Reductions: partial sums → one result

```python
result = torch.randn(64, 512).sum(-1)
```

| Generated IR | PTX mapping |
| --- | --- |
| `for a1 in a1_co..512:128` | Compare/branch/increment: per-thread partials |
| `WarpShuffle(acc0, length=32)` | `shfl.sync.bfly.b32` + add: warp sums |
| `Sync` + `TreeHalve` | `bar.sync` + shared-memory tree: final sum |

**Beyond sum:** sequential loops, varied reduction widths, RMSNorm, L2 normalization and online softmax. Accumulator types, identities and coupled-state merges remain explicit.

Code: [counted_loop](compile.py#L524), [merge_states](compile.py#L489).

### Tensor cores and asynchronous copies

```python
a = torch.randn(16, 16, dtype=torch.float16)
b = torch.randn(16, 8, dtype=torch.float16)
result = a @ b
```

| Generated IR node | Native PTX mapping |
| --- | --- |
| `CpAsyncCopy`, commit/wait | `cp.async` + completion groups: stage into shared memory |
| `LdmatrixLoad` | `ldmatrix.sync.aligned`: load warp fragments |
| `MmaSyncPtx` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |

**Beyond basic matmul:** guarded edges, split-K, transposed B, multiple warps, swizzles and double/triple buffering. The public scalar matmul also supports dynamic shared memory.

Code: [tensor emission](ptx_tensor.py).

## 4. Verified coverage

**MVP: 8/8 · Local tests: 47 · Generated matmul: 9 global + 9 shared**, passing both references. Public fused examples and 512×512 scalar matmul also pass. Hidden coverage remains unproven.

[Fresh hidden-style MVP variants](docs/compiler/19-mvp-hidden-coverage.md): 71/71 GPU results match the IR; 62/71 match NumPy. Nine generated-program/precision disagreements and five Emmy generation failures remain recorded.

```bash
python tools/test_mvp.py
python tools/test_matmul.py --suite shared --runner
```

## 5. Future work

Atomic accumulation for m02; arrayed fragments and fused tensor-core epilogues; end-to-end attention; broader datatypes and profiling-led optimization.
