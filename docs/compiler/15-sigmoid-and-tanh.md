# Sigmoid and tanh: native instructions and accuracy

Both operations now emit PTX directly from Emmy's existing typed Assign nodes. The dispatch is in `arithmetic()` in [compile.py](../../compile.py); the implementation is `activation()` in [ptx_math.py](../../ptx_math.py). No extra IR or optimization pass is introduced.

## Which native instructions do we use?

| Operation / datatype | Emitted implementation |
| --- | --- |
| tanh, f16 | `tanh.approx.f32` on the widened input; final half rounding |
| tanh, f32 | Native `ex2.approx.f32` through our exp helper, plus arithmetic and a small-input polynomial |
| sigmoid, f16/f32 | Native `ex2.approx.f32` through our exp helper, plus arithmetic; final half rounding for f16 |
| tanh/sigmoid, f64 | Our existing full-double exp helper plus double arithmetic; a small-input polynomial for tanh |

PTX has no standalone sigmoid instruction and no native f64 tanh/exp equivalent. We do not convert f64 inputs to f32 to use a cheaper instruction. The f16 tanh implementation uses the f32 instruction because the emitter already stores half values in exactly widened f32 working registers; it does not currently use a packed half tanh instruction.

NVIDIA documents `tanh.approx.f32` as requiring sm_75 or newer, so it is available for our sm_89 target. Its documented maximum relative error is 2^-11; supported subnormal inputs and signed-zero behavior are described in the [PTX instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions-tanh). Native approximate instructions still need numerical validation against the accuracy we require.

## Why f32 tanh does not use the tanh instruction directly

Our initial implementation used `tanh.approx.f32` for both f16 and f32. The f16 tests passed their preselected tolerance. Strict f32 tests did not: for example, tanh(1) returned approximately 0.76160097 instead of 0.76159418 on the hosted runner. The [initial results](../../review/activations/initial-native-tanh.json) retain those failures. We kept the original tolerances and changed the f32 implementation.

For f32/f64, let a=abs(x) and t=exp(-2*a). Away from zero:

```text
tanh(x) = sign(x) * (1-t)/(1+t)
```

The exponential's argument is nonpositive. This avoids the overflow caused by forming exp(2*x) for large positive x. Near zero, however, subtracting t from one loses precision. For abs(x)<0.125 we instead evaluate the odd tanh Taylor polynomial using explicit FMA:

```text
f32: x * (1 - x²/3 + 2*x⁴/15 - 17*x⁶/315)
f64: extend the same series through degree 15
```

The final multiplication and sign-bit selection retain tiny inputs and negative zero. This is an accuracy-oriented emission expansion, not a claim of correctly rounded tanh for every input. The f32 path still uses hardware exponentiation; it does not invoke CUDA, libdevice or another compiler.

## Stable sigmoid

With t=exp(-abs(x)), emit:

```text
if x >= 0: t_result = 1/(1+t)
else:      t_result = t/(1+t)
```

The choice is implemented with a predicate and `selp`; both quotients use `div.rn`. We avoid exp(-x) overflowing for large negative x, and avoid subtracting a nearly-one result from one to compute the negative tail. For half inputs, the internal arithmetic remains f32 until the original sigmoid operation's final half rounding. We do not add rounding boundaries between instructions inside that one operation.

## Generated tests and the retained f64 reduction difference

[tools/test_activations.py](../../tools/test_activations.py) creates 30 cases through PyTorch → pinned Emmy 0.3.11 → unchanged Kernel IR → our compiler. Each fresh fixture receives a dump-checked proof. The matrix includes both activations at sizes 1/33/129 for f16/f32/f64, explicit edge inputs, an x*sigmoid(x) composition and a tanh followed by a row sum.

Explicit tensors include signed zeros, the smallest positive subnormal, near-zero values, both sides of the polynomial threshold and saturating magnitudes. The sigmoid edge range stays within eager PyTorch's finite-exponential range: up to magnitude 80 for f16/f32 and 700 for f64. This permits an independent PyTorch check of the NumPy expectation without accepting differences from eager sigmoid's own overflow behavior. It is not coverage of every overflow transition or all nonfinite inputs.

The [final report](../../review/activations/validation.json) records 29 original-expression passes and one f64 reduction difference. All 30 GPU results match the independent typed-IR reference. The remaining expression difference is `tanh_sum_f64`: Emmy generates f64 tanh but an f32 accumulator, shared buffer and reduction steps, then copies the sum back to f64. We preserve those types. We do not change its expected values, loosen tolerances, rewrite the IR or count it as a competition pass.

All 40 local test methods passed, including instruction selection, native versus full-double paths, fixture provenance and local sm_89 assembly. All 11 public GPU regression cases also passed with this compiler revision.

The interpreter's activation values come from NumPy operations, not our PTX expansion. Expression expectations are computed separately in double precision and rounded to each PyTorch operation's output type. Comparisons use rtol 2e-3/2e-6/2e-12 for f16/f32/f64; standalone activations and edge cases use zero absolute tolerance. Composed sums have a separately declared absolute tolerance, unchanged between the initial and final runs.

Run in the pinned Ubuntu environment:

```bash
python tools/test_activations.py --runner
python tools/test_activations.py --reuse --runner
python -m unittest discover -s tests -p 'test_*.py'
```

The activation command currently exits with a failure status because the f64 expression difference is retained. Read the expression and typed-IR statuses separately. Local tests explicitly record that producer precision difference while requiring the other examples to match their expression reference. The signed-zero unit check validates the independent reference; ordinary hosted numeric comparisons alone do not establish bitwise signed-zero output coverage.

Only our compiler PTX is uploaded to the hosted runner. The original competition generator, specifications, supplied problems and runner remain unchanged. These are tests, not a final competition submission. Current public regression results are in [the public report](../../review/native-math/activation-regression.json); the full historical regression matrix was not rerun for this addition.

The next feature investigation is genuine Emmy-generated matmul/tensor-core and copy-engine IR, following the organizer's manual-review guidance. Cases pinned Emmy cannot generate remain out of scope; examples and tests must come from actual producer output.
