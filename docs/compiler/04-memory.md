# Addresses, loads and stores

Implementation: `address`, `memory_space`, `memory_load`, `memory_store`, `vector_store`, `index_divisor`, and the `Load`, `Write`, `Smem` branches of `statement()` in [compile.py](../../compile.py).

## Elements become byte addresses

IR indices count elements. PTX addresses count bytes. For an f32 buffer with shape `(3, 129)`:

```text
x[row, col]
element_offset = row * 129 + col
byte_offset = element_offset * 4
address = x_base + byte_offset
```

`address()` performs row-major flattening, then uses `mul.wide.s32` to create the 64-bit byte offset and `add.u64` to add the base pointer. A constant element offset is first moved into a register because of the instruction's operand requirements. An empty index is accepted for a buffer containing one element.

The current index arithmetic uses 32-bit signed intermediate calculations. A 64-bit pointer does not make arbitrary enormous index expressions safe; this emitter does not yet widen all index calculations.

## Load

`Load.names` can contain one name or several consecutive names. We calculate the first address, load each element at `offset * element_bytes`, convert to the declared result dtype if needed, and bind the resulting names and types.

```text
a, b = load x[i]
```

currently produces two scalar loads. Grouped loads are not yet vectorized. The input buffer's element dtype controls the memory instruction; `Load.dtype` controls any conversion after loading.

## Write

The output buffer dtype controls the store width. Each input value is converted to that dtype. Plain writes first try `vector_store()`; if it returns false, the original scalar-store loop handles every element.

```ptx
st.global.v4.f32 [address], {%f1, %f2, %f3, %f4};
```

This is four consecutive elements from one thread, not one element from four different threads. The helper allows two or four values with a total size no greater than 16 bytes. Four f64 values therefore fall back to scalar stores. Half vectors contain converted `b16` values.

## Proving alignment

The fresh argument buffers used by the runner are recorded as having at least 16-byte base alignment. This assumption depends on that allocation contract; an API accepting offset views would need to supply a different guarantee. Shared arrays use their declared alignment, or their natural element alignment.

The starting flattened index must also be divisible by the number of vector elements. `index_divisor()` proves a conservative divisor:

```text
literal k        -> abs(k)
unknown variable -> 1
a * b            -> divisor(a) * divisor(b)
a + b            -> gcd(divisor(a), divisor(b))
a - b            -> gcd(divisor(a), divisor(b))
```

Zero has divisor zero because it is divisible by every positive width.

For `row * 8 + col * 4`, the guarantee is `gcd(8, 4) = 4`, enough for a four-element vector. For `row * 5 + col * 4`, it is only one, so we fall back. This includes row-stride alignment rather than checking only the final coordinate.

## Shared memory

`Smem` declares static storage when metadata requests no dynamic shared memory, or allocates an aligned slice of one dynamic arena when smem_bytes is positive. It binds the address and records the element shape/type. Subsequent ordinary `Load` and `Write` nodes use `ld.shared` / `st.shared` through `memory_space()`.

Native asynchronous copies and their B32/B64/B128 shared layouts are now supported; see [shared staging](17-shared-memory-async-copies.md). Atomic writes and general swizzled ordinary Write nodes remain unsupported and are rejected rather than silently emitted as plain stores.

Next: [arithmetic](05-arithmetic.md).
