# Pointwise/reduction coverage beyond the public MVP

**Eight public passes establish the submission baseline, not hidden-set completeness.** Hidden-style testing must start from PyTorch expressions and use pinned Emmy's actual output. A finite matrix can cover operations, structural patterns and boundaries; it cannot enumerate every expression or input value.

## What was audited

The existing extended suite samples all five numeric datatypes, arithmetic, sum/mean/row sum, widths around warp/block boundaries, four cooperative work sizes, conversion pairs, comparisons, numeric `where`, activations and scalar loops. Reports are historical evidence for their recorded compiler versions, not proof that every combination has been tested on the current compiler.

A fresh replay of 154 existing source-attested non-matmul fixtures found no emission failures. Their metadata contained rank-one and rank-two inputs, motivating new rank-three and composition cases. This replay checks emission only; it is not another 154 GPU passes. Older extended/boundary suites predate the strict `emmy-proof.json` check and must be regenerated to obtain that current provenance evidence.

## New systematic PyTorch matrix

Implemented in [test_mvp_hidden_variants.py](../../tools/test_mvp_hidden_variants.py). Each successful trace stores `expression.py`, the unchanged Emmy IR, `meta.json`, independent `expected.npz` and `emmy-proof.json` under `tests/mvp-hidden/`.

| Family | Variants | What it exercises |
| --- | ---: | --- |
| Rank-three composed arithmetic | 5 datatypes | Subtract → ReLU → multiply, with an independent add branch |
| Rank-three numeric `where` | 5 datatypes | Comparison, two arithmetic branches and selection |
| Row/column broadcast compositions | 10 | Broadcast indexing, ReLU and multiply |
| Masked row sum | 5 | Comparisons/arithmetic feeding a reduction |
| Non-final-axis sum | 5 | Reduce axis zero rather than the last axis |
| Rank-three full sum | 5 attempted | Multiple reduced axes under the selected cooperative schedules |
| Strided slice plus sum | 5 | Non-contiguous PyTorch indexing and reduction broadcast |
| Rank-three mean | 3 floating types | Extra unreduced axes and typed mean |
| Two reductions combined | 3 floating types | Independent reduction state and final arithmetic |
| Log/exp/power composition | 3 floating types | Several math operations composed in one expression |
| Keepdim mean/centering | 4 work sizes | Widths 31/32/33/257 and work sizes 32/64/128/256 |
| Mixed-datatype broadcast arithmetic | 20 ordered, unequal type pairs | Type promotion and conversions across all five numeric datatypes |
| Multiple outputs | 2 f32 graphs | Two pointwise outputs; independent sum and mean outputs |
| Rank-four broadcast composition | 1 f32 graph | Two broadcast dimensions and composed arithmetic |

This is a selected interaction matrix, not the full datatype × size × operation × schedule product. Its other widths include 129 and 257. Existing suites provide additional sizes and explicit-input stress cases.

## Fresh hosted results

| Outcome | Cases |
| --- | ---: |
| Attempted PyTorch cases | 76 |
| Emmy-generated, attested cases assembled and executed | 71 |
| GPU matches independent typed-IR interpreter | 71/71 |
| GPU matches original NumPy expression | 62/71 |
| Expression mismatches retained | 9 |
| Emmy generation failures under selected settings | 5 |

The compiler implementation was unchanged. The [combined summary](../../review/mvp-hidden/summary.json) covers the original [53-case run](../../review/mvp-hidden/validation.json) and [23 additional interaction cases](../../review/mvp-hidden/expanded.json), recording compiler hashes and both comparisons. All GPU uploads contained our compiler's PTX.

### Findings that must remain visible

- **Five slicing mismatches:** `x[::2] + x[1::2].sum()` generates loads at `x[a1]` and `x[a0 + 1]`, omitting the step of two. Both our GPU results and the independent IR interpreter follow those generated indices. The original PyTorch/NumPy results differ. See the [f32 generated IR](../../tests/mvp-hidden/hidden_strided_slice_f32_w64/k_slice_reduce_266606.ir.txt).
- **Four f64 reduction mismatches:** masked row sum, axis-zero sum, rank-three mean and two combined reductions contain f32 accumulation state. GPU output matches that IR, while strict f64 expression expectations fail. These are not expression-level passes.
- **Five rank-three full-sum generation failures:** Emmy reports missing output-coordinate extents for the selected cooperative settings. There is no valid generated kernel to test for those cases. This establishes rejection of those settings, not rejection of every possible schedule for the expression. [Generation errors](../../review/mvp-hidden/generation-failures.json) retain the producer diagnostics.

No missing PTX handler or GPU/IR disagreement was exposed by the 71 generated cases. That conclusion is limited to this matrix. Expectations, tolerances, IR bodies, schedules and datatypes were not changed to make the results pass. Before hosted execution, eager PyTorch caught an i32 NumPy oracle promotion error in the slicing test; the reference was corrected to match PyTorch's i32 result, and the fixture was regenerated. Its genuine slicing mismatch remains a failure.

## Reproduce in Ubuntu

Use the pinned Python environment from the [compiler run guide](../../COMPILER_README.md), including CPU PyTorch, with `ptxas` on `PATH`.

```bash
# Fresh PyTorch -> Emmy fixtures, local assembly and independent IR reference.
python tools/test_mvp_hidden_variants.py --report-name local.json

# Reuse source-attested fixtures; execute only our emitted PTX on the runner.
python tools/test_mvp_hidden_variants.py --reuse --runner

# Regenerate and test a selected case.
python tools/test_mvp_hidden_variants.py --only hidden_column_sum_f32_w64 --runner

# Focus on mixed inputs and multiple outputs.
python tools/test_mvp_hidden_variants.py --families mixed_inputs multi_outputs --runner
```

Each case first checks its independent NumPy expression oracle against eager CPU PyTorch. On reuse, absent proofs are recorded as generation failures; the original producer diagnostics are retained separately. Local assembly is not a GPU pass. The hosted command deliberately returns nonzero for the retained expression mismatches.

## Remaining coverage work

Expand expression-tree combinations, datatype/work-size/width interactions, ranks beyond four, and multiple-output datatypes beyond f32. Regenerate older boundary fixtures with strict provenance. Keep unusual numeric inputs separate from structural coverage; NaN/Inf transport and comparison limitations require explicit handling rather than ordinary random-input claims.

Only classify a feature as unavailable from the producer after testing the relevant pinned-Emmy expression/settings. Do not fill a gap by constructing or editing Kernel IR. Any genuine compiler failure found by a new trace becomes a regression before optimization work resumes.
