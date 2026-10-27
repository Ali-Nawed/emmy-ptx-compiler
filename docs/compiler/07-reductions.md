# Accumulators, warp shuffles and shared reductions

Implementation: `counted_loop`, `merge_states`, `tree_halve`, `sync`, and the `Accum`, `WarpShuffle`, `Smem`, `Sync`, `TreeHalve` branches in [compile.py](../../compile.py).

## We translate Emmy's existing reduction schedule

There is no separate reduction IR in our compiler. The Kernel IR already says which elements each worker visits, how lanes combine, where shared memory is used and when barriers occur. We implement those statements directly.

The usual row-sum sequence is:

```text
per-thread strided accumulation
    -> WarpShuffle: one sum per warp, available to its lanes
    -> lane zero of each warp writes its sum to shared memory
    -> Sync
    -> TreeHalve: combine warp sums and broadcast the block sum
    -> write or use the result
```

## Accum

`counted_loop()` seeds state with its operation's typed identity: zero for addition, one for multiplication, or the supplied extrema sentinel. `Accum` reads that mutable state, converts the incoming value to its dtype, performs the declared add/multiply/maximum/minimum and moves the result back into the original state register. If `base` is supplied, that binding supplies the left operand instead. Out-of-range extrema sentinels are clamped for integer state; floating finite sentinels are preserved, as explained in [operation expansion](11-operation-expansion.md).

The declared accumulator type matters. An f64 input can be converted into an explicitly f32 accumulator if that is what Emmy generated; our compiler must not silently rewrite the program into a different precision.

## WarpShuffle

For a full 32-lane sum, the XOR butterfly exchanges partners at distances 16, 8, 4, 2 and 1. Each round makes the partner state available under the IR's `state_b` names and executes the provided `combine_states` program.

```ptx
activemask.b32 %mask;
shfl.sync.bfly.b32 %partner, %value, 16, 31, %mask;
// execute the IR's combine assignments
```

Only power-of-two lengths up to 32 are accepted. For f64 or i64 state, a single 32-bit shuffle cannot carry the entire value: we split it into low/high 32-bit halves, shuffle both with the same partner, then reconstruct the 64-bit value.

## merge_states

This helper emits all assignments in `combine_states` before copying final results into the carried state registers. That ordering avoids overwriting one input while another merge assignment still needs its old value. It restores the original state-register bindings afterward.

The merge program can contain multiple assignments and coupled states. Online softmax uses maximum and sum state, with exponentials to rescale the sum as the maximum changes. A merge can compute in a different type from its outer state: final values are converted back to the saved target types after all merge assignments finish. The NumPy interpreter also retains this distinction; rejecting a temporary promoted binding would incorrectly reject valid mixed-type reductions.

## Shared memory, Sync and TreeHalve

For 128 threads there are four warps. Four lane-zero writers store the per-warp sums to a shared array. `Sync` emits `bar.sync` so later readers see all writes.

For a tree of length four, `tree_halve()` emits stages with `half=2`, then `half=1`:

1. Lane zero of warp 0 combines shared slots 0 and 2; lane zero of warp 1 combines slots 1 and 3.
2. A barrier makes these results visible.
3. Lane zero of warp 0 combines slots 0 and 1.
4. Another barrier completes the tree.
5. Every participating thread loads shared slot zero into its state register.

That final broadcast is essential for normalization: every thread needs the row denominator, not just the thread that stores a sum.

Only flat power-of-two trees with `tid_var='warp'` and no nested `inner` tree are implemented. `sync()` supports block barriers and named barriers with explicit counts; warp-scope barriers are currently rejected. Correct control flow must bring all required participants to each barrier.

Next: [normalization](08-normalization.md).
