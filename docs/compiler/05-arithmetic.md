# Arithmetic operations

Implementation: the `Assign` branch of `statement()`, `arithmetic`, `operand`, `literal` and `round_half` in [compile.py](../../compile.py).

## How Assign is translated

For an IR statement such as `f32 v = multiply(x, y)`, the emitter determines the declared dtype, converts the two input bindings to it, emits arithmetic into a new register, then records `values['v']` and `value_dtypes['v']`.

Inside a reduction merge, an omitted type defaults to the combine type. Elsewhere the fallback is f32. Explicit declared types take precedence.

## Supported operations

Boolean comparison and numeric `where` assignments take a separate path before numeric operand conversion. See [boolean comparisons and where](14-boolean-comparisons-and-where.md) for type promotion, predicate logic and NaN semantics.

`T` below is `f32` or `f64` for floating work, and `s32` or `s64` for signed integer arithmetic. f16 uses the f32 working path followed by half rounding.

| Emmy operation | Floating PTX | Integer PTX / handling |
| --- | --- | --- |
| `add(a,b)` | `add.rn.T` | `add.T` |
| `subtract(a,b)` | `sub.rn.T` | `sub.T` |
| `multiply(a,b)` | `mul.rn.T` | `mul.lo.T` |
| `divide(a,b)` | `div.rn.T` | `div.T` |
| `sqrt(a)` | `sqrt.rn.T` | Rejected |
| `reciprocal(a)` | `rcp.rn.T` | Rejected |
| `relu(a)` | Comparison and selection | Comparison and selection |
| `copy(a)` | `mov.T` after operand conversion | Same |
| `bitwise_and(a,b)` | Rejected | `and.b32` or `and.b64` |
| `right_shift(a,b)` | Rejected | `shr.s32` or `shr.s64` |
| `negative(a)`, `abs(a)` | Native `neg` / `abs` | Native signed `neg` / `abs` |
| `square(a)` | Multiply a by itself | Low-width multiply |
| `maximum(a,b)`, `minimum(a,b)` | Compare/select with explicit NaN propagation | Native signed max/min |
| `exp`, `exp_fast`, `exp2` | f16/f32: native f32 ex2 with range reduction; f64: our double polynomial | Rejected |
| `log`, `log2`, `log10` | f16/f32: native f32 lg2 with near-one correction; f64: our double series | Rejected |
| `power(a,b)` / `pow(a,b)` | Our exp/log expansion with domain/sign handling | Exponentiation by squaring; nonnegative exponents |
| `rsqrt(a)` | Internal sqrt + reciprocal, one final half rounding | Rejected |
| `sigmoid(a)`, `tanh(a)` | Typed native-instruction/math expansion; see [activation reasoning](15-sigmoid-and-tanh.md) | Rejected |
| `bitcast(a)` | Same-width raw move to/from integer data | Same-width raw move to/from floating data |

Integer multiplication keeps the low result bits; it does not become floating multiplication. Signed integer division truncates toward zero. Inputs with undefined integer division behavior should not be used as correctness evidence.

## New: square root and reciprocal

The branch is intentionally small:

```python
elif operation in ("sqrt", "reciprocal") and not integer and len(operands) == 1:
    opcode = "sqrt" if operation == "sqrt" else "rcp"
    self.emit(f"{opcode}.rn.{kind} {result}, {operands[0]};")
```

For a normalization denominator:

```ptx
sqrt.rn.f32 %f10, %f9;
rcp.rn.f32 %f11, %f10;
```

`.rn` means round to nearest, ties to even. These sqrt/reciprocal instructions use neither `.approx` nor `.ftz`. Exp/log have a separate approximate native path described in [native math](13-native-exp-log-and-coverage.md). The f64 path stays f64. The f16 path applies the usual half rounding after each operation.

The two instructions remain separate because `1 / round(sqrt(x))` has an intermediate rounding boundary. Replacing it with `rsqrt.approx` would change that computation. This is deliberate semantics preservation; it is not a performance claim.

Instruction forms and rounding semantics are documented in NVIDIA's [PTX ISA: reciprocal](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#floating-point-instructions-rcp) and [square root](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html#floating-point-instructions-sqrt).

## ReLU and shifts

ReLU compares against zero with `setp.lt`, then uses `selp` to choose zero for negative values. Keeping that comparison/selection also preserves the current handling of negative zero and NaNs; we have not substituted a potentially different `max` form.

Right shift uses a signed shift, preserving the sign bit. The shift count is a 32-bit operand even for i64 data; the i64 path clamps an oversized unsigned count to 64 before narrowing it.

We do not contract arbitrary IR multiply-plus-add pairs into FMA. The handwritten math expansion uses explicit FMA for its own polynomial and range-reduction steps. Exp/log/power are accuracy-oriented approximations, not correctly rounded library replacements. Their derivation, limitations and tests are in [operation expansion](11-operation-expansion.md).

`bitcast` is handled before ordinary operand conversion in `Assign`: converting numerically first would destroy the bits being reinterpreted. It accepts equal-width 32/64-bit values, and does not yet accept half values. Historical synthetic tests used this for encoded classifications; those tests are retired and do not establish current end-to-end coverage. Any replacement must be generated from a PyTorch expression through Emmy. Unknown operation names, unsupported dtypes and wrong arities fail explicitly.

Next: [control flow](06-control-flow.md).
