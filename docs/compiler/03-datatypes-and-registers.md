# Datatypes, conversions, constants and registers

Implementation: `reg`, `element_bytes`, `ptx_type`, `data_reg`, `literal`, `convert`, `operand`, `move`, `round_half` in [compile.py](../../compile.py).

## Storage type versus working type

| IR dtype | Memory bytes | Working register | PTX arithmetic type |
| --- | ---: | --- | --- |
| `f16` | 2 | `%f...` | `f32`, rounded back to half at operation boundaries |
| `f32` | 4 | `%f...` | `f32` |
| `f64` | 8 | `%d...` | `f64` |
| `i32` | 4 | `%r...` | `s32` |
| `i64` | 8 | `%l...` | `s64` |
| internal `bool` | No memory ABI support | `%p...` | `pred` comparisons/logic/selection |

Pointers use `%rd...` 64-bit registers. Predicates use `%p...`. Half storage bits temporarily use `%h...`. Register names are conventions chosen by us; their declarations determine their PTX types.

## Fresh virtual registers

```python
def reg(self, kind):
    self.counts[kind] += 1
    return f"%{kind}{self.counts[kind]}"
```

This creates names, not physical GPU register assignments. `ptxas` later allocates actual registers and may reuse storage for values whose lifetimes do not overlap. We do not maintain liveness intervals or implement a register allocator. An increase in declared virtual registers is not automatically an increase in physical register usage.

## Half precision

A half load reads 16 bits and widens exactly to f32:

```ptx
ld.global.b16 %h1, [address];
cvt.f32.f16 %f1, %h1;
```

`round_half()` narrows an operation result to half, then widens the rounded value back into an f32 working register:

```ptx
cvt.rn.f16.f32 %h2, %f2;
cvt.f32.f16 %f3, %h2;
```

Keeping the value rounded at each boundary prevents a chain of half operations from silently becoming a full-f32 computation until the final store. This remains a computation in f32 followed by half rounding; it is not a claim of native half instruction use or universally identical rounding to every possible native-half implementation.

## Explicit conversions

`value_dtypes` records each bound data value's type. `operand(name, dtype)` asks `convert()` for a value in the consuming operation's declared type. Wide integers and doubles are not routed through f32.

Float-to-integer conversion uses truncation toward zero (`.rzi`). Integer-to-float and narrowing f64-to-f32 use round-to-nearest (`.rn`). Same-type operands are reused. f16-to-f32 already has the correct register representation.

## Constants

`literal()` uses PTX hexadecimal bit encodings for floating immediates: `0f...` for f32 and `0d...` for f64. Integer immediates remain integers. This avoids accidental loss of floating literal precision through decimal formatting.

There are two kinds of constants to distinguish: instruction immediates, such as an accumulator's zero identity, and constant buffers supplied as kernel arguments, such as RMSNorm's epsilon. The latter are still loaded from memory.

Next: [memory](04-memory.md).
