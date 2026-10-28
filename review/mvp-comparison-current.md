# Remaining MVP code-generation gaps

Fresh local comparison of the current compiler using CUDA 12.0.140, `sm_89`, and `ptxas -O3`. The reference is saved Emmy CUDA compiled with nvcc; the table uses `--fmad=false` at both reference compilation and assembly to compare separate multiply/add operations. All eight cases assemble in all three variants; no spills were reported. No reference kernel was uploaded or executed.

| MVP case | Our non-NOP static SASS instructions | Emmy/nvcc, FMA disabled |
| --- | ---: | ---: |
| p00 scalar add | 24 | 21 |
| p01 vector add | 24 | 19 |
| p02 scale/shift | 38 | 30 |
| p03 ReLU/multiply | 44 | 29 |
| r01 array sum | 102 | 158 |
| r02 big sum | 102 | 158 |
| r03 mean | 216 | 272 |
| r04 row sum | 105 | 96 |

These counts include static loop bodies and slow paths, not dynamic instruction counts or timing measurements. Fewer instructions or registers alone do not establish faster execution.

## Candidates, in implementation order

1. **Eliminate an ordinary Tile's worker guard only when launch coverage is exact.** `compile.py` currently emits it for every ordinary Tile. Public p01/p02/p03 have exact coverage, and their reference kernels omit it. Require `grid.x * block.x == tile.n_elements`; preserve the guard for p00, which launches 256 threads for 64 workers, and all partial tiles. Test exact and partial coverage before hosted MVP validation.
2. **Reuse a contiguous group's base address.** p02 currently assembles five `IMAD.WIDE` instructions versus two in the reference; p03 has nine versus three. `address()` independently scales each index. Compute a common base plus constant byte offsets where justified, with explicit overflow/range handling. This arithmetic is introduced by our emitter after Emmy's Loop IR, so earlier invariant handling does not remove all of it. Rerun odd-size, multidimensional and datatype regressions.
3. **Investigate native ReLU/max.** p03 uses four compare/select pairs while the reference uses four `FMNMX` instructions. Validate NaN, signed-zero and typed rounding behavior before replacing our sequence. A smaller spelling is not enough to prove equivalence.

Coordinate shifts/masks and aligned grouped stores are already implemented. Arbitrary FMA contraction, changed reduction order, and general unrolling are lower priority because they need numerical or runtime tradeoff evaluation. The MVP already passes correctness; these are optimization candidates, not newly demonstrated correctness bugs.

Reproduce without overwriting historical evidence:

```bash
python tools/compare_emmy_ptx.py --limit 8 --out /tmp/emmy-mvp-current-comparison
```

The output contains PTX, SASS, register/spill logs, commands and source hashes. Only our compiler's generated PTX should be used for subsequent hosted tests.
