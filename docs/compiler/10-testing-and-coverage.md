# Testing, evidence and unsupported features

Implementation: [tests](../../tests), [tools/ir_reference.py](../../tools/ir_reference.py), [tools/test_extended.py](../../tools/test_extended.py), and [tools/validate_normalization.py](../../tools/validate_normalization.py).

## Three different checks

| Check | What it tells us | What it cannot establish |
| --- | --- | --- |
| Local `ptxas -arch=sm_89` | PTX syntax, typing and target compatibility | Runtime correctness |
| Independent NumPy Kernel IR interpreter | Expected typed operations and scheduled reduction results | Actual GPU execution behavior |
| Hosted compile/run plus full output comparison | Our generated PTX runs correctly for those inputs and launch | All hidden tests or optimal performance |

The interpreter uses Emmy's parser, but does not import the emitter or upstream CUDA renderer. Its new sqrt/reciprocal arithmetic uses NumPy and models our f16 working/rounding policy.

## Keep two references

`expected.npz` is the original expression-level NumPy expectation. The IR interpreter independently evaluates the emitted input program's declared types and reduction order. When saved separately, that result is `expected_ir.npz`; the normalization validator computes the same reference in memory.

An IR-reference match is useful for separating upstream precision choices from our translation errors. It does not turn an expression-level failure into a competition pass. In earlier 480-case regression results, all matched the typed IR reference, but 27 still differed from their original-expression expectations due to known precision/order issues. Those references remain unchanged.

## Reproduce in Ubuntu

Activate the pinned environment, then run from the repository root:

```bash
python -m unittest discover -s tests -v
python tools/validate_normalization.py --generate --runner
```

The second command generates 18 new fixtures from PyTorch expressions through Emmy, assembles our compiler's PTX locally, and tests those fixtures plus the eight public MVP cases, two public normalization cases and public row softmax on the runner. Without `--runner`, it assembles and computes references without GPU execution. Once generated, omit `--generate` to reuse the saved fixtures.

Generated cases cover sqrt/reciprocal on f16/f32/f64 at sizes 1 and 129, plus f32 L2/RMS-style normalization of three rows with widths 1, 129 and 1025. The public RMSNorm additionally exercises weights. Fixture folders preserve source expressions, IR, metadata, NumPy expectations and provenance. Kernel tests must use upstream-generated IR; the old hand-typed backend tests have been retired. Fresh regeneration records `emmy-proof.json`, and strict reuse verifies its hashes.

`review/normalization/validation.json` records statuses and the compiler hash. `review/normalization/<case>/` preserves our emitted PTX. The validator uploads only the output of our own `compile.py`; it never submits reference CUDA/PTX. These are test requests, not final competition submission.

## Earlier normalization baseline

On October 24, 2026, all 25 local unittest methods passed, including local sm_89 assembly checks. All 28 cases in the hosted normalization validation passed against both the original NumPy expression expectations and the typed-IR interpreter: eight public MVP problems, two public normalization problems, and 18 fresh generated cases. The report identifies compiler SHA-256 `3ea74b1a42b0f2ea524dc72434a0d25d8dfa803070d679c46b88cfc049d36993`.

That local count is historical and included tests later retired under the strict source-generation policy. The current revised suite has 29 passing methods, all kernel tests use real Emmy output, and the current hosted generation-only results are described in [operation expansion](11-operation-expansion.md) and [the required pipeline](12-required-test-pipeline.md).

The hosted checks used one execution per kernel with no warmup. Their timings are correctness-run observations, not a performance benchmark or evidence that the new instructions are optimal.

Signed-zero/NaN behavior also has explicit CPU oracle assertions. Ordinary hosted comparisons reject NaNs even on both sides, so those assertions should not be described as exceptional-value GPU validation. Current generated hosted cases use ordinary finite random inputs.

## Scope today

See [the native-math validation and coverage audit](13-native-exp-log-and-coverage.md) for its historical counts. Ordinary scalar-accumulating `Loop`, numeric `where`, sigmoid and tanh now have implementations and generated regressions; see [Loop](06-control-flow.md#ordinary-loop), [boolean comparisons and where](14-boolean-comparisons-and-where.md) and [activations](15-sigmoid-and-tanh.md). The activation report retains an additional f64 reduction precision difference rather than counting it as a pass. Upstream generation failures are out of scope under the pinned-producer rule; implemented handlers alone do not establish coverage.

The supported path includes ordinary single-Tile pointwise work, add/multiply/max/min reductions, normalization and the supported online softmax program. Floating data has f16/f32/f64 paths; signed integers have i32/i64 arithmetic where the operation permits it. sqrt/reciprocal/exp/log reject integer types. Integer power accepts nonnegative exponents. [Operation expansion](11-operation-expansion.md) records the newer implementation and its separate validation.

Still missing are array-state Loop programs, boolean memory/output ABI, boolean-valued where, atomic writes and general swizzled ordinary stores. `Let`/`Init`, ordinary scalar-accumulating Loop, internal boolean/numeric selection and integer-index ternaries now have handlers, but this does not imply support for arbitrary pure expression trees. Native async copies, selected swizzled shared slabs and staged matrix loads are documented in [shared staging](17-shared-memory-async-copies.md). Auxiliary/rasterized Tile schedules, general multidimensional launches, warp barriers and nested shared reduction trees are also unsupported. Matmul and attention remain incomplete.

The [first tensor-core matmul subset](16-first-tensor-core-matmul.md) supports genuine f16/f32 m16n8k16 MMA with global fragment loads, guarded stores and split-K chaining. It now also supports shared staging and the public m01 matmul. Arrayed fragment families and m02's atomic writes remain unsupported; this is not a claim of complete matmul support.

Refer to the current result JSON for observed successes rather than inferring support from a category name. The hidden evaluation set is unavailable, so our own generated fixtures help explore its likely variations without proving hidden-set coverage.
