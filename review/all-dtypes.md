# Datatype lowering and validation

The submission compiler now handles f16, f32, f64, i32 and i64 data for the current
pointwise and reduction statements. It translates typed Kernel IR directly to
our own PTX; reference CUDA is never compiled or uploaded by these tests.

## Implementation

- Loads, stores and address strides follow buffer dtypes; pointers remain u64.
- f64 arithmetic uses f64 registers; i32/i64 use native signed integer operations,
  including wrapping low-word multiplication. They do not travel through f32.
- f16 values use b16 memory and rounded f32 working values, preserving rounding
  at each typed operation boundary.
- Copies and mixed input/state/output types use explicit conversions. Local
  assembly checks cover all 25 conversion pairs among the five supported types.
- Reduction state follows the dtype declared by Accum, WarpShuffle and TreeHalve.
  A 64-bit shuffle splits the value into two 32-bit words and reconstructs it.
  Shared storage uses the appropriate type, width and alignment.
- Integer bitwise-and and arithmetic right-shift support the generated i64
  validation expressions. A 64-bit shift count is clamped before narrowing.

## Strict hosted matrix

Fixtures come from PyTorch expressions through pinned Emmy 0.3.11. Independent
NumPy expected arrays and tolerances are unchanged. Sizes are 1 and 129; row sums
use three rows. There are eight operations per dtype.

| dtype | passes | strict failures |
| --- | ---: | ---: |
| f16 | 16 | 0 |
| f32 | 16 | 0 |
| f64 | 13 | 3 |
| i32 | 16 | 0 |
| i64 | 16 | 0 |
| total | 77 | 3 |

All 80 compile locally and assemble for sm_89. These are custom visible tests,
not the hidden evaluation set. Timing samples are correctness-run observations,
not controlled performance measurements.

## Why three f64 reductions fail

Emmy 0.3.11's kernel type-stamping pass uses f32 state for accumulating reductions,
including these f64 inputs. The compiler honors that declared state. For sum,
mean and row_sum at width 129, this loses precision relative to NumPy f64.

For example, the sum expectation is 7.388847032206026; the GPU returns
7.388847351074219. The restricted CPU trace in
`tools/diagnose_reduction_precision.py` reproduces each thread's f32 additions,
warp exchanges, shared tree and final conversion. Its values match every GPU
mismatch in all three cases. Evidence is in
`review/reduction-precision-diagnosis.json`. No tolerance was relaxed and no
fixture IR was rewritten to make these cases pass.

This upstream state choice also threatens wide integer reductions; the small
integer inputs that pass the default matrix cannot expose that loss. Explicitly
typed f64/i64 state is additionally tested with local backend assembly unit
cases, distinct from genuine generated fixtures and GPU correctness tests.

## Numeric edges and the runner interface

Five f64 and five i32 pointwise edge cases pass on the hosted GPU with exact
numeric comparisons. They include f64 low bits/subnormals and integer wraparound.

Large raw i64 input/output testing exposes a hosted interface limitation:
INT64_MAX inputs can trigger `Python int too large to convert to C long`, while
otherwise valid returned values lose low bits consistently with an intermediate
floating conversion. For example, 2**60 + 3 is returned as 2**60. Some rounded
outputs become 2**63 and cannot be converted back to int64 by the checker.
The checker records these failures rather than crashing. Reports preserve both
the original input failures and the subsequent raw-output failures. The inferred
serialization cause has not been confirmed from the hosted server's source.

Eight fresh PyTorch -> Emmy i64 tests extract low 8 bits or bits 32 through 47
on the GPU after add, subtract, multiply or scale/shift. Inputs are exactly
representable by the hosted transport, but arithmetic results exercise low bits
above 2**53 and overflow. Returned values fit in 16 bits. All eight pass exact
NumPy and CPU PyTorch expectations; report: `review/i64-output-bit-results.json`.
These validate native arithmetic while leaving the raw-output failures visible.

Current remaining compiler gaps include lexical scoping, broader reduction
schedules and unsupported statements/operations beyond the current MVP family.
