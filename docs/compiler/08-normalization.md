# L2 normalization and RMSNorm

Implementation: existing memory, arithmetic and reduction handlers plus the new `sqrt` / `reciprocal` branch of `arithmetic()` in [compile.py](../../compile.py). No problem-name dispatch was added.

## L2 normalization

For each row:

```text
sum_of_squares = sum(x[column] * x[column])
inverse_length = 1 / sqrt(sum_of_squares)
output[column] = x[column] * inverse_length
```

The public [f03_l2norm IR](../../problems/f03_l2norm/k_div_reduce.ir.txt) has two strided loops. The first squares and accumulates inputs, followed by warp/shared reductions. Then `sqrt` and `reciprocal` produce the denominator multiplier. The second loop rereads inputs and writes normalized elements.

For a zero row, this expression has no epsilon: zero divided by zero produces NaN. We do not silently add an epsilon. The current hosted comparison helper does not count NaNs as equal, so exceptional-value tests need explicit classification assertions rather than ordinary allclose comparisons.

## RMSNorm

For a row with C elements and weights w:

```text
mean_square = sum(x[column] * x[column]) / C
inverse_rms = 1 / sqrt(mean_square + epsilon)
output[column] = w[column] * (x[column] * inverse_rms)
```

The public [f01_rmsnorm IR](../../problems/f01_rmsnorm/k_mean_cbd1de.ir.txt) expresses this as:

| IR portion | Existing/new emitter work |
| --- | --- |
| First `StridedLoop`, multiply, `Accum` | Per-worker sum of squares |
| `WarpShuffle`, `Smem`, `Sync`, `TreeHalve` | Combine and broadcast the row sum |
| Load `mean_count`, `reciprocal`, multiply | Divide the sum by the row length |
| Load epsilon, add | Stabilize the denominator |
| `sqrt`, `reciprocal` | Compute the normalization multiplier |
| Second `StridedLoop`, loads, multiplies, write | Normalize and apply weights |

Constants remain arguments. We do not hardcode C, epsilon, row count or weights in a dedicated RMSNorm handler.

## Follow one thread

For C=1024 and 128 workers per row, worker three reads columns 3, 131, 259, ..., 899 in the first loop. The block reduces every worker's partial sum. After the tree broadcast, worker three has the same row sum as the other workers. It computes the scaling factor and revisits its assigned columns in the second loop.

This explains why a correct sum-only kernel is not sufficient evidence for a fused kernel: a sum kernel might write from just one lane, while normalization requires the result in every participating lane.

## What changed in PTX emission

```ptx
sqrt.rn.f32 %root, %mean_plus_epsilon;
rcp.rn.f32 %inverse, %root;
```

The names here are illustrative; our allocator emits numbered registers. We preserve the IR's multiply ordering and separate sqrt/reciprocal rounding. No rsqrt fusion, approximate reciprocal, new intermediate IR or liveness analysis was introduced.

Next: [optimizations](09-optimizations.md).
