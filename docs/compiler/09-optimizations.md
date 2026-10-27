# Implemented optimizations and their safety conditions

Implementation: `unsigned_divide`, `unsigned_remainder`, `tile_coordinates`, `index_divisor` and `vector_store` in [compile.py](../../compile.py).

These are local choices during direct PTX emission. The architecture remains parse -> visit -> emit. There is no additional optimizing IR or pass manager.

## Tile coordinate simplification

For positive divisors and unsigned, nonnegative worker indices:

| Before | After | Reason |
| --- | --- | --- |
| Divide by one | Reuse the input | Identity |
| Remainder by one | Zero immediate | Only possible remainder |
| Divide by 2^k | Shift right by k | Unsigned binary arithmetic |
| Remainder by 2^k | AND with 2^k - 1 | Keep the low k bits |
| Outermost coordinate remainder | Omit | Tile bounds already put quotient within its extent |

For `(17,128)`, decoding becomes a shift and an AND. Non-power-of-two division/remainder retains ordinary PTX instructions. We preserve the bounds guard; it is part of the proof that omitting the outer remainder is safe.

These helpers apply only to Tile decoding. We have not added a general algebra simplifier for all index expressions.

## Aligned grouped stores

The IR already groups consecutive values in a `Write`. We preserve that information with a vector store if two or four values fit in 16 bytes and both the base and byte offset meet its alignment requirement.

The conservative divisor proof includes every row-major stride. If alignment is unknown, the helper returns false and the existing scalar store loop is used. No runtime alignment branch or speculative vector access is emitted.

Four f32 stores can become `st.global.v4.f32`; two f64 stores can become `st.global.v2.f64`. Four f64 stores fall back under our current 16-byte limit. Half conversions remain before the memory instruction, preserving stored values.

Fewer memory instructions do not imply a fourfold bandwidth improvement. Neighboring threads' scalar accesses can already be coalesced, and a grouped store can keep more values live until the store.

## What we measured for the earlier optimization change

The recorded comparison in [review/emitter-optimizations](../../review/emitter-optimizations) assembled both versions with the same local sm_89 toolchain:

- Row-sum static non-NOP instructions fell from 152 to 105; physical registers went from 17 to 16.
- Vector-add instructions fell from 25 to 24; physical registers went from 18 to 16.
- Scale-shift instructions fell from 41 to 38, while physical registers increased from 23 to 24.
- The inspected kernels had no spills. Hosted timings were close and inconsistent; they did not establish a substantial speedup.

These are results for those recorded versions and cases, not a benchmark of every current or future kernel. Static instruction count is evidence about generated code, not a substitute for execution timings.

## Deliberately not implemented

- Liveness analysis, manual register reuse or physical register allocation.
- General loop-invariant code motion or address hoisting.
- Grouped vector loads.
- Contraction of arbitrary IR multiply/add pairs, sqrt/reciprocal fusion or substitution of existing operations with faster approximations. Our separate exp/log/power expansions do use explicit FMA internally for their handwritten math.
- Reduction-order changes, bounds-check elimination or schedule retuning.

Emmy may already move some invariant expressions in its earlier IR stages. Our emitter still introduces byte-address instructions itself, so some future local improvements could remain possible. We have not claimed that all address math is already hoisted.

Next: [testing and coverage](10-testing-and-coverage.md).
