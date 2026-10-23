# Boundary, schedule and input-pattern tests

Latest hosted results: all 357 new cases match the typed IR. The generated matrix
has 305/317 expression passes and 12 f64 reduction failures. The explicit-input
matrix has 28/40 expression passes and 12 cancellation/large-integer failures.
All 24 failing expression results match the independent CPU IR prediction.
No compiler or assembly failures occurred. Expectations and tolerances remain
unchanged. Compact report: review/boundary-validation-results.json.

This is an additional matrix, separate from tests/extended's original 80 cases.
All shape/schedule fixtures are generated from PyTorch through Emmy 0.3.11.
No serialized IR is edited to create schedule variants. Only our compiler's PTX
is uploaded to the hosted runner.

## Coverage

317 generated cases:

- 144 f32 reduction cases: sum and row sum at widths 31, 32, 33, 127, 128, 255,
  256, 257 and 1025; requested work widths 32, 64, 128 and 256; row counts 1, 3
  and 17 for row sums. Observed launch dimensions are recorded in the report.
- 128 additional datatype cases: f16, f64, i32 and i64; all eight current
  operations; widths 31, 33, 257 and 1025, with three rows for row sums.
- 45 f32 pointwise cases: five operations at all nine boundary widths.

40 explicit-input cases reuse unchanged IR from the width-257 generated cases:
all five datatypes, sum/three-row sum, and zeros, negative values, cancellation
and large values. Metadata contains the explicit inputs. input-provenance.json
records the original fixture and hashes of the copied IR files.

The cancellation pattern places a large value in lane 0, +1 in lane 16 and the
negative large value in lane 1. This deliberately exposes an XOR reduction that
rounds away +1 before cancellation. Integer large-value cases use 2**30 + 1,
which exposes f32 accumulator loss while keeping the returned sums below 2**53
to avoid the already documented large-integer transport issue.

## Commands in Ubuntu

```bash
# Generate, interpret and assemble locally; generation cohorts can run in parallel.
python tools/test_boundaries.py --phase generate

# Hosted requests run serially. Known expression precision failures give exit 1.
python tools/test_boundaries.py --phase run

# Or do both phases in order.
python tools/test_boundaries.py --phase all

# Focus on one cohort while iterating.
python tools/test_boundaries.py --phase run --cohorts reduce_f32_w32

# Explicit input patterns (generate boundary fixtures first).
python tools/test_reduction_inputs.py --runner
```

Reports: tests/boundaries/results.json and summary.json aggregate the selected
cohorts; each cohort keeps results.json for its hosted run and local-results.json
for local-only checks. The first run's older fixtures may have only results.json;
the harness accepts these too. Generation/runner logs retain CLI diagnostics.
The explicit-input report is tests/reduction-inputs/results.json.

Both expectation sets remain separate: expected.npz is the independent NumPy
expression result; expected_ir.npz simulates the typed IR and reduction order.
The top-level status continues to reflect the original expression. Each hosted
case also records details.comparisons.kernel_ir. Neither the IR expectation nor
the comparison tolerances replace the original correctness criterion.

Precision failures must remain visible, including nonassociative f32 cancellation
and Emmy's f32 state for f64/integer reductions. A PTX-versus-IR pass does not
establish a competition pass. This matrix does not reproduce the hidden set.
