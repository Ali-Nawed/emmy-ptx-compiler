# Emmy/nvcc versus our direct PTX

The boundary tests were completed first: all 357 new hosted cases match the
independent typed-IR checker; 333 match the original NumPy expressions. The 24
expression failures remain recorded: 12 f64 boundary reductions and 12 explicit
cancellation/large-integer cases reproduce the IR's precision and reduction order.
See review/boundary-validation-results.json and tests/BOUNDARIES.md.

## Method and limits

Our path is Kernel IR -> compile.py -> PTX. The comparison path is saved Emmy
reference CUDA -> local nvcc -> PTX. The compiler does not call nvcc. The local
comparison tool has no runner client or networking. Reference code was never
uploaded or executed; only our compiler's PTX was used in GPU tests.

19 kernels cover all eight public MVP problems, four work widths for a 1025-value
sum, a 17x1025 row sum, four other pointwise datatypes, and f16/f64 sums. All three
variants of all 19 kernels assemble without spills.

Local tools: CUDA 12.0.140, sm_89, ptxas -O3. Variants: ours, default nvcc -O3,
and nvcc with --fmad=false at BOTH the nvcc and ptxas stages. The second flag
matters: inline half PTX was still contracted by ptxas when only the front-end
option was disabled. Disabling both gives separate HMUL2/HADD2 instructions.

Each review/emmy-ptx-comparison/<case>/ folder contains ours.ptx, emmy.ptx,
emmy_no_fma.ptx, reference CUDA with headers, ptxas logs, SASS and comparison.json.
The root results.json records versions, commands, source hashes and launch shapes.

The table's SASS counts exclude NOP padding but include static loop bodies,
control flow and slow paths. They are not executed instruction counts. These
local comparisons do not establish speedup, equal runtime, occupancy or RTX 5090
performance. The hosted toolchain can allocate registers differently. Reference
timing on the hosted GPU was not performed, consistent with our upload policy.

## Recommended optimization order

1. Simplify Tile coordinate decoding. Row kernels emit div.u32/rem.u32, including
   an unnecessary outer-axis modulo. r04_row_sum assembles three MUFU.RCP
   instructions plus conversion/multiply/correction code; the reference has none.
   Both use 17 registers, but ours has 152 non-NOP static instructions versus 96.
   The 17x1025 row case shows the same counts. Use shifts/masks for power-of-two
   divisors, remove divide by one, and omit the outer modulo after proving the
   worker is within the Tile. This preserves floating-point reduction order.

2. Preserve aligned grouped stores. p01's two-value Write becomes two scalar
   STG.E instructions in ours versus one STG.E.64 in the reference. p02/p03 have
   four scalar stores versus one STG.E.128. Keep scalar fallback when alignment
   or contiguity is unproved. The reference p01 loads remain scalar; its name
   alone does not imply vector loads. Handle grouped Load nodes when the actual
   IR provides their guarantees.

3. Reuse addresses and hoist invariant row bases. p01 has five IMAD.WIDE address
   calculations versus three; p02 has five versus two. Compute the row base and
   starting pointer once, then advance by step * element_bytes in strided loops.
   Preserve bounds, index ranges and accumulation order.

4. Remove worker guards when metadata proves exact Tile coverage. Public p02
   has an unnecessary bounds comparison/early exit in ours; the reference omits
   it. Partial Tiles and scalar work still require guards.

5. Use native half operations with the same rounding boundaries. f16 scale/shift
   has 30 non-NOP instructions in ours versus 22 in the noncontracted native
   half reference (21 with contraction). Our widened working values generate
   repeated conversions. Keep half rounding after each typed operation and rerun
   the rounding-sensitive f16 fixtures.

6. Consider native ReLU/max and FMA with explicit numerical semantics. p03 has
   four FSETP/FSEL pairs versus four FMNMX instructions. Check NaN/signed-zero
   behavior before changing its spelling. Default p02 uses four FFMA instructions
   versus our four FMUL plus four FADD; contraction changes rounding. Even the
   noncontracted reference has fewer instructions (30 versus our 41), so address
   and store improvements are available while preserving the arithmetic.

7. Measure reduction unrolling tradeoffs. Several scalar reference reductions
   use 37 registers versus our 14-16 with larger unrolled bodies. Extra registers
   can buy fewer loop iterations or more overlap; neither register count nor
   static instruction count identifies the faster kernel by itself.

PTX text alone can mislead: p02 has 12 global loads in our PTX versus six in the
reference, but BOTH assemble six LDG.E instructions because ptxas removes duplicate
loads. The row reduction's root load is also sunk below its writer guard by the
assembler. None of the 57 variants spills, so a general register reuse pass is
lower priority than the concrete index/store differences.

Next implementation: Tile coordinate simplification, then grouped stores. Keep
floating operations and the reduction schedule unchanged, rerun numerical checks,
then measure our before/after kernels on the hosted runner. No compiler
optimization was applied during this comparison.

Reproduce in Ubuntu: `python tools/compare_emmy_ptx.py`.

NVIDIA references: [PTX vector memory instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
and [nvcc contraction options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#fmad-true-false-fmad).

## Full static comparison

Each triple is ours / default Emmy-nvcc / noncontracted Emmy-nvcc.

| Kernel | Physical registers | Static SASS, excluding NOP |
| --- | --- | --- |
| p00_scalar_add | 18 / 14 / 14 | 25 / 21 / 21 |
| p01_vector_add | 18 / 14 / 14 | 25 / 19 / 19 |
| p02_scale_shift | 23 / 16 / 20 | 41 / 26 / 30 |
| p03_relu_mul | 22 / 18 / 18 | 47 / 29 / 29 |
| r01_array_sum | 16 / 37 / 37 | 102 / 158 / 158 |
| r02_big_sum | 16 / 37 / 37 | 102 / 158 / 158 |
| r03_mean | 16 / 37 / 37 | 216 / 272 / 272 |
| r04_row_sum | 17 / 17 / 17 | 152 / 96 / 96 |
| sum_f32_1025_w32 | 14 / 37 / 37 | 76 / 139 / 139 |
| sum_f32_1025_w64 | 16 / 37 / 37 | 93 / 150 / 150 |
| sum_f32_1025_w128 | 16 / 37 / 37 | 102 / 158 / 158 |
| sum_f32_1025_w256 | 16 / 37 / 37 | 111 / 164 / 164 |
| row_sum_f32_17x1025 | 17 / 17 / 17 | 152 / 96 / 96 |
| scale_shift_f16_1025 | 14 / 12 / 12 | 30 / 21 / 22 |
| scale_shift_f64_1025 | 14 / 16 / 14 | 22 / 25 / 26 |
| scale_shift_i32_1025 | 14 / 14 / 14 | 21 / 21 / 21 |
| scale_shift_i64_1025 | 16 / 16 / 16 | 26 / 26 / 26 |
| sum_f16_1025 | 18 / 26 / 26 | 114 / 187 / 187 |
| sum_f64_1025 | 23 / 40 / 40 | 110 / 188 / 188 |
