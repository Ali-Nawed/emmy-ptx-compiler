# Local nvcc tanh comparison

This audit compiles saved, source-attested Emmy reference CUDA into PTX locally. Reference code/PTX is neither uploaded nor executed. It measures instruction selection, not runtime or accuracy.

## What the installed toolchain emits

Toolchain: **CUDA 12.0.140**, target `compute_89`, `nvcc --ptx -O3 --std=c++17`.

The f32 source calls ordinary `tanhf` three times in its unrolled body.

| Local compilation | Observed PTX |
| --- | --- |
| Default | Polynomial FMA branch and exp/reciprocal-based branch with correction; no `tanh.approx.f32` |
| `--use_fast_math` | Same broad expansion and static math-op counts, with FTZ modifiers on additional floating operations; no `tanh.approx.f32` |

Across the three unrolled tanh calls, both variants contain 18 FMA instructions, three `ex2.approx.ftz.f32` instructions and three `rcp.approx.ftz.f32` instructions. These are static counts including alternative branches, not instructions executed by every call.

Selected default-mode instructions, with intervening control flow/operations omitted:

```ptx
fma.rn.f32 %f27, %f26, %f24, %f25;
ex2.approx.ftz.f32 %f17, %f16;
rcp.approx.ftz.f32 %f20, %f18;
fma.rn.f32 %f22, %f20, %f21, %f19;
```

The observation supports the reasoning behind our math helper: an accuracy-oriented tanh can require a sequence combining ordinary arithmetic and approximate native primitives. Our implementation uses its own exp-based formula and near-zero polynomial; it does not invoke or copy nvcc's translation, and this audit does not establish equal accuracy or performance.

## Half-precision limitation

The saved f16 reference calls `htanh`, which these CUDA 12.0 headers do not provide. Both local nvcc modes reject it as undefined. We retained the failure instead of rewriting the source. Consequently, this audit establishes no nvcc lowering result for that half reference. Our own f16 PTX uses native `tanh.approx.f32` and half rounding and has separate hosted validation.

The saved f64 activation CUDA is not used for a double-precision compiler comparison: it narrows intermediates to float and calls `tanhf`. That is already a producer/rendering choice, and cannot be attributed to nvcc's lowering of a true double `tanh` call.

## Reproduce

In the pinned Ubuntu environment with local CUDA tools:

```bash
# Successful f32 comparison.
python tools/compare_nvcc_tanh.py --dtypes f32

# Include the known f16 CUDA-source compilation failures; exits nonzero.
python tools/compare_nvcc_tanh.py
```

[comparison.json](comparison.json) records tool versions, commands, source hashes, instruction counts and diagnostics. PTX and logs are alongside it. Findings are specific to this version, target, source API and flags; do not generalize to every nvcc release.

NVIDIA documents `tanh.approx.f32`'s approximate accuracy in the [PTX reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh). The [CUDA math guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html) distinguishes standard math APIs and fast intrinsics; version-specific mappings should be checked against actual generated PTX.
