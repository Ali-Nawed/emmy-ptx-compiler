# Native exp/log and the remaining coverage gaps

Update: ordinary scalar-accumulating `Loop`, numeric `where`, sigmoid and tanh have since been implemented. See [Loop emission](06-control-flow.md#ordinary-loop), [boolean comparisons and where](14-boolean-comparisons-and-where.md) and [activations](15-sigmoid-and-tanh.md). The observations below describe the audit at the native-math revision.

The standalone f16/f32 exp/log path now uses PTX's f32 special-function instructions instead of computing every result in software double precision. This changes emission inside our existing translator; it adds no intermediate IR or optimization passes. Double exp/log and floating power retain their previous implementation.

## Read the code

In [ptx_math.py](../../ptx_math.py), `floating_math()` selects `native_math32()` for f16/f32 exp/exp_fast/exp2/log/log2/log10. Half inputs already have f32 working registers; `compile.py`'s `arithmetic()` retains the final half rounding at the IR operation boundary. The compiler still follows Emmy's declared types and statement order.

For exp, a naive `ex2(x*log2(e))` lost accuracy at large magnitudes. The emitted code instead computes:

```text
k = clamp(round(x*log2(e)), -126, 127)
r = x - k*ln2_high - k*ln2_low    # two explicit FMA instructions
exp(x) ≈ ex2.approx(r*log2(e)) * ex2.approx(k)
```

Clamping keeps the scale normal and finite; the final multiply supplies subnormal results or overflow. A predicated minimum-subnormal fix handles the first f32 input with a nonzero exponential: permitted hardware approximation error otherwise rounded this particular boundary to zero. There is no `.ftz` instruction modifier.

For log, `lg2.approx.f32` supplies the main result. Its absolute error is too large relative to a logarithm approaching zero. Within [0.75, 1.25], we instead evaluate `z=(x-1)/(x+1)` and `2*z*(1+z²/3+z⁴/5+z⁶/7+z⁸/9)`. The emitted polynomial uses explicit FMA. Results are converted to the requested logarithm base. This is a small arithmetic expansion inside the emitter, not a new PTX instruction.

These are approximations, not a claim of correctly rounded math or universal input coverage. NVIDIA documents the instruction forms and their error/special-value behavior in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). Arbitrary user multiply/add statements are still not contracted into FMA.

## Validation and reproduction

Every new fixture follows PyTorch → pinned Emmy 0.3.11 → unchanged packaged Kernel IR → our compiler → our PTX. Generated fixtures include dump-checked provenance. Older genuine snapshots retain their existing provenance; it is not fabricated retroactively. Only our compiler PTX is uploaded to the hosted GPU.

- [Baseline regression report](../../review/native-math/validation.json): 169 cases, 162 original-expression passes and seven retained f64 precision differences. All 169 GPU results match the typed-IR reference.
- [New edge report](../../review/native-math/edges/validation.json): six passing f16/f32 cases, covering exp underflow/overflow boundaries, log near one, small positive inputs and log(exp(x)). Both independent expression and typed-IR comparisons pass.
- Together: 175 regression cases, 168 expression passes and seven known f64 expression differences; all 175 match the typed IR. The seven differences remain failures against the expression, not competition passes.
- All 32 local tests pass. They check native instruction selection, genuine fixture provenance/assembly and byte-identical f64 exp/log/power PTX versus the saved baseline, alongside the existing compiler tests.

The edge fixtures were produced by changing tensor values in the PyTorch source. No datatypes, schedules or operations were patched into the resulting IR. The reports retain source hashes. Initial failed probes are preserved beside the final reports to explain why the range reduction, near-one correction and underflow fix exist.

```bash
python tools/test_pytorch_cases.py --families native_edges --runner
python tools/test_native_math.py --runner
python tools/probe_emmy_coverage.py --runner
python -m unittest discover -s tests -p 'test_*.py'
python tools/benchmark_native_math.py
```

The benchmark compares saved before/after output from our own compiler, checks correctness again, and runs three paired rounds with alternating order, ten warmups and 100 iterations. [timings.json](../../review/native-math/timings.json) contains individual samples and local sm_89 assembly statistics. Hosted clocks are not locked, so small-kernel timing ratios are observations rather than guarantees.

| Case | Before / after median (µs) | Observed speed ratio |
| --- | --- | --- |
| f32 exp range | 4.768 / 2.976 | 1.60× |
| f32 log range | 4.800 / 2.944 | 1.63× |
| Public row softmax | 19.104 / 4.768 | 4.01× |
| f16 softmax, 3×129 | 17.056 / 4.736 | 3.60× |
| f32 softmax, 3×257 | 29.312 / 4.736 | 6.19× |
| Unchanged f64 exp control | 4.768 / 4.768 | 1.00× |

The locally assembled exp-range kernel shrank from 65 to 32 non-NOP SASS instructions; public softmax shrank from 1085 to 523. These are local sm_89 statistics, not the hosted sm_120 JIT's final machine code.

## Do we cover every case for our implemented kernel families?

No. Category support does not imply that every expression composition or schedule works. [The actual Emmy probes](../../review/native-math/coverage-probes.json) found:

| PyTorch case | Result | Implication |
| --- | --- | --- |
| Row broadcasting | Pass | This generated layout works |
| Column broadcasting | Pass | This generated layout works |
| Variance-style composition | Pass | Multiple reductions can work in some schedules |
| Two row sums in one expression | Unsupported `Loop` | In-scope control-flow gap |
| `where` | Unsupported bool buffer dtype | In-scope boolean/selection gap |
| Sigmoid | Unsupported sigmoid arithmetic | In-scope arithmetic gap |
| Tanh | Unsupported tanh arithmetic | In-scope arithmetic gap |
| Attempted multi-axis sum schedule | Emmy generation failed | This attempted case is out of scope |

The eight probes are separate from the 175 regression cases. A failed multi-axis schedule does not prove Emmy cannot produce any multi-axis reduction. Under the user's scope rule, expressions the pinned producer cannot generate do not require fabricated IR tests or compiler work. Earlier upstream-rejected direct operations are excluded on the same basis.

The next useful compiler change is ordinary `Loop` emission for the genuine two-row-sum fixture, followed by boolean selection and the generated sigmoid/tanh operations. Each needs its own generated regression cases. Broader shape, datatype, scheduling and exceptional-value combinations remain unproven; the earlier 480-case matrix was not rerun as part of this math change. Hidden-set coverage cannot be established from these local fixtures.

Original competition files and supplied problems are unchanged. These hosted calls are tests, not a final competition submission.
