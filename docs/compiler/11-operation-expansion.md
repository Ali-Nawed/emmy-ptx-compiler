# Expanding operation coverage: decisions and evidence

This is the implementation record for the arithmetic/reduction expansion started October 24, 2026. Results are added as checks complete. The implementation remains our direct typed emitter; no optimization-pass pipeline or external IR-to-code compiler is introduced.

**Current test policy:** every kernel must originate from a PyTorch expression compiled by Emmy. The former synthetic backend tests are retired and excluded from the results below. See [the required pipeline](12-required-test-pipeline.md).

## 1. Treat arithmetic and reduction combination separately

`(x-y).sum()` uses subtraction to form each element and addition to combine elements. Subtraction and division do not become unordered tree combiners: changing their order changes the program. `compile.py` now accepts the explicitly supported associative combiner families add, multiply, maximum and minimum, along with their reduction aliases.

The loop seed comes from the IR operation's identity; the update uses the actual combiner instead of hardcoded addition. Product therefore starts at one. Maximum/minimum preserve Emmy's finite floating +/-1e30 sentinels, even though these are not mathematical infinities. For integer state, an out-of-range sentinel is clamped to the integer datatype's bound. That supplies a representable neutral integer value instead of emitting an invalid immediate or converting wide state through f32. The independent NumPy interpreter implements the same documented typed-identity policy separately.

The supplied `Accum.base` remains the left operand when present. This matters for online softmax: the prior sum is rescaled by an exponential before the update. Shuffles and shared trees still execute the IR's supplied merge program, preserving its assignment order and state snapshots.

## 2. Maximum/minimum, signs and NaNs

Signed integer extrema use native PTX max/min. Floating extrema use comparison/select, choosing the right operand for ties, and explicitly propagate a NaN from either input. This follows NumPy maximum/minimum behavior, including differently signed zero, rather than accidentally adopting number-preferring `fmax` behavior. NaN payload bits are not promised.

Negation and absolute value use typed native instructions. `rsqrt` is lowered internally as sqrt then reciprocal, with a single final half rounding for the one IR operation. The existing separate `sqrt` and `reciprocal` nodes still retain two operation boundaries.

## 3. Exponentials and logarithms without narrowing f64

The all-double f16/f32 implementation below describes the earlier baseline. Standalone f16/f32 exp/log now use [native PTX instructions with accuracy corrections](13-native-exp-log-and-coverage.md). The f64 and floating-power paths retain this double implementation.

[ptx_math.py](../../ptx_math.py) contains our own PTX emission helpers. They use no CUDA renderer, nvcc, libdevice or external math implementation. The helpers are mixed into `PTXEmitter` and use its register allocator and output buffer. A submission must include this module beside `compile.py`.

PTX exposes f32 approximate `ex2`/`lg2`, but there is no equivalent native f64 instruction. Rather than silently converting doubles to float, this first baseline computes floating exp/log in f64, then narrows at the IR operation boundary for f32/f16. `exp_fast` currently takes the same accuracy-oriented path as `exp`; its name does not force us to use an approximation instruction.

For exp, range reduction forms `x=k*ln(2)+r` with `|r|` approximately at most `ln(2)/2`. A degree-18 Taylor polynomial approximates `exp(r)`. Split high/low ln(2) constants keep the residual accurate for large x. Two normal power-of-two factors reconstruct the scale so subnormal outputs are rounded on the final multiply. Explicit guards handle NaNs, infinities and values safely beyond the overflow/underflow transitions.

For log, binary exponent/mantissa extraction forms `x=m*2^e`. Positive subnormals are first scaled by 2^54. Bringing m near one allows `z=(m-1)/(m+1)` with `|z|<0.172`; the odd series `2*(z+z^3/3+...+z^27/27)` approximates log(m). Then add e*ln(2). Zero, negative inputs, infinity and NaN have explicit result fixups. log2/log10 divide this result by ln(2)/ln(10); exp2 scales its input by ln(2) before exp.

These are finite polynomial approximations with double arithmetic, not a claim of correctly rounded libm or guaranteed accuracy over all inputs. Internal polynomial evaluation explicitly uses FMA; we still do not contract arbitrary user IR multiply/add pairs. Numerical range, exceptional values and performance need independent testing. Accuracy-oriented code can be much larger and slower than a tuned f32 path.

Primary instruction reference: NVIDIA's [PTX ISA](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html), particularly floating comparisons/conversions, FMA, ex2 and lg2.

## 4. Powers

The floating `power`/`pow`/`float_power` path computes exp(y*log(abs(x))) using our own helpers, then fixes the real-domain and sign cases. Finite negative bases require integral exponents; odd exponents restore a negative sign, including negative zero. Explicit overrides cover exponent zero, base one, and magnitude-one bases with infinite exponents.

This handles variable floating exponents independently of whether Emmy rewrites `x**2` as multiplication. A direct `square` node also lowers to one multiplication. General real power can magnify approximation error, especially for large exponents or bases very close to one; it is tested separately rather than inferred from exp/log support.

Integer power uses exponentiation by squaring for nonnegative integer exponents. It starts with result one, conditionally multiplies for the low exponent bit, squares the base and logically shifts the exponent until zero. All multiplies retain the declared 32/64-bit width, including overflow wraparound. `0**0` is one. Negative integer exponents are outside the integer-power input domain, as in NumPy/PyTorch; the oracle rejects them and we make no GPU result guarantee for invalid-domain inputs.

## 5. Binding and expression support

`Let` handles literals, typed variable copies and integer index expressions; `Init` handles explicit typed seeds. These unblock the literal `one` in public online softmax. Index `TernaryExpr` uses a predicate and select. This is not yet a general boolean-tensor/where implementation, general branch-join analysis, or support for arbitrary expression nodes.

A same-width 32/64-bit `bitcast` uses a raw move before any numeric operand conversion. Its former synthetic classification tests are retired; any replacement must come through PyTorch and Emmy. A direct f16 bitcast remains unsupported.

## 6. Fresh generated combination tests

[tools/test_operations.py](../../tools/test_operations.py) creates PyTorch source, traces with pinned Emmy 0.3.11, packages Kernel IR/metadata and independent NumPy expression outputs, then compiles with our emitter. It keeps a second typed-IR reference and reports both comparisons. Original expression expectations are never replaced to make a test pass.

Successful generated families cover subtraction/division, pointwise min/max, negation/absolute value, exp and base-changing expressions, log and base-changing expressions, variable powers, rsqrt, arithmetic before summation, mean broadcasting, maximum reductions and online softmax. Floating operations use f16/f32/f64; integer-applicable families use i32/i64. Sizes and cooperative widths are configurable. Amin/prod are optional generation probes that currently fail upstream; they are not active coverage successes.

Examples, from the repo root in the pinned Ubuntu environment:

```bash
python tools/test_operations.py --dtypes f32 --sizes 33 129 --runner
python tools/test_operations.py --dtypes f16 f64 i32 i64 --sizes 1 129 --runner
python tools/test_operations.py --dtypes f32 --ops maximum softmax sum_composed --sizes 257 --work 32 --runner
python tools/test_pytorch_cases.py --runner
```

Reports go to `review/operations/`; fixtures and our generated PTX go to `tests/operations/`. Only our compiler's PTX is uploaded. Trace failures, compiler failures, IR/expression mismatches and GPU/IR mismatches remain distinct evidence. No final competition submission is performed.

## 7. Upstream generation limits and replacement tests

The initial probes are preserved in `review/operations/initial-upstream-failures.json`. The pinned upstream CUDA packaging stage rejects direct exp2/log2/log10 nodes. Our fresh PyTorch tests therefore express these using exp/log and explicit base-change constants at the source-expression level; we do not edit their serialized Kernel IR. Their NumPy references follow those actual operation boundaries. Direct exp2/log2/log10 coverage formerly supplied by hand-typed IR has been withdrawn.

The pinned Torch tracer also rejects amin/prod reductions. They remain opt-in tracer probes in `test_operations.py`, not default generated tests. We do not claim those combiners were generated through PyTorch. Their handlers remain implemented, but the retired synthetic schedules are not evidence of valid end-to-end support. Fresh generated maximum and softmax cases exercise the real upstream schedules at different block widths.

`tools/test_pytorch_cases.py` replaces the fabricated math/conversion tests with 34 actual PyTorch traces: 25 datatype conversion pairs and nine exp/log/power range cases. Range cases contain 257 values each, with tensor values and datatypes specified in the PyTorch setup. The resulting IR is compared with Emmy's dump and fingerprinted. Expected values are computed independently with NumPy and checked against eager PyTorch.

The hand-typed special-value classification and forced-wide-reduction fixtures were also retired. CPU-only arithmetic checks do not replace missing GPU coverage. No new IR is manufactured to cover a failed upstream trace.

## 8. File boundaries

Competition-owned files are left unchanged, including the root README, TASK/MVP specifications, supplied problems, `submit.py`, and the original fixture generator. Implementation changes live in our `compile.py` and `ptx_math.py`; explanatory Markdown lives under our `docs/compiler/`; test tools, generated fixtures and result reports live in their own paths. Testing a supplied problem reads it without rewriting its IR, metadata or expected outputs.

## 9. Results and remaining numerical limits

The October 24, 2026 hosted checks cover:

| Suite | Cases | Original-expression / backend checks | GPU versus typed IR |
| --- | ---: | --- | --- |
| Fresh f32 expressions, sizes 33 and 129 | 28 | 28 passed | 28 passed |
| Fresh f16/f64/i32/i64 expressions, sizes 1 and 129 | 72 | 65 passed; 7 f64 differences below | 72 passed |
| Fresh composed-sum/max/softmax, width 257, blocks 32 and 256 | 6 | 6 passed | 6 passed |
| Real PyTorch conversion and range expressions | 34 | 34 passed | 34 passed |
| Eight public MVPs, public L2/RMS/softmax, 18 prior normalization fixtures | 29 | 29 passed | 29 passed |

Current total: 169 generated/public hosted cases, 162 passing their original-expression expectations and seven retained expression-level differences. All 169 match the typed-IR interpreter. The previous 88 hand-typed cases are excluded; the old 223-case aggregate is historical and must not be presented as active coverage.

The revised local unittest suite passed all 29 methods. Tests now load actual Emmy nodes instead of constructing or rewriting them. New policy checks reject IR constructors/rewrites in active test code and altered fresh fixture hashes. The original competition README, TASK/MVP, runner, generator and supplied problem folders remain unchanged.

Run `python tools/summarize_operations.py` after the suites to collect `review/operations/summary.json`. It verifies that all reports refer to the current `compile.py` and `ptx_math.py` hashes, requires hosted execution, and rejects new failures outside the seven explicitly retained expression gaps. Historical initial probe/fixture-error reports are excluded from the final totals but preserved for diagnosis.

The seven retained differences are `log2_f64_129`, `log10_f64_129`, `sum_composed_f64_3x129`, `center_f64_3x1`, `center_f64_3x129`, `maximum_f64_3x129`, and `softmax_f64_3x129` (all work128). Their exact fixture names and mismatching examples remain in `review/operations/other-types.json`.

The generated base-log expressions contain f32 reciprocal/final multiply nodes even though their log nodes are f64. The center expression of width one multiplies the f64 input in an explicitly f32 node before subtracting it, so its mathematical zero becomes a tiny nonzero result. The larger reductions have f32 state/merge or normalization arithmetic. The compiler preserves those declared types; it does not silently rewrite the supplied IR to meet a different expression-level precision. The replacement genuine PyTorch-generated f64 exp/log/power range cases passed their tighter numeric comparisons.

Floating max/min still inherit the IR's finite +/-1e30 seed sentinels. A maximum over values all below -1e30, for example, can therefore return the sentinel instead of an input value. That is an upstream IR/program limitation under the preserve-declared-identity policy, not proof that all floating extrema inputs are covered. Generated finite random tests do not exercise this magnitude range.

The range tests use rtol 4e-13 for direct f64 math, 2e-6 for f32 and 2e-3 for f16, with atol zero. Passing those finite samples does not establish universal error bounds, correct rounding or parity with a tuned GPU math library. General boolean-tensor/where expressions and many other activations remain unsupported. Performance of the new double-based math is not benchmarked here; correctness-run timings have no warmup and should not be used as performance claims.
