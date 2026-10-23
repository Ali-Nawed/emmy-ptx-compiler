# Independent NumPy Kernel IR expectations

`tools/ir_reference.py` evaluates typed Kernel IR on the CPU. It is a testing
tool separate from the submission compiler. Nothing in compile.py calls it.
It uses Emmy's public Graph parser and statement classes, but does not import
our emitter or invoke Emmy's CUDA renderer, nvcc, or a GPU reference kernel.

## What the two checks mean

| Expectation | Source | Question answered |
| --- | --- | --- |
| expected.npz | Independent NumPy expression corresponding to the PyTorch program | Is the complete pipeline numerically correct for the original computation? |
| expected_ir.npz | NumPy interpretation of the generated typed Kernel IR | Does our PTX implement the IR's computation and reduction order? |

Both use the problem's original floating tolerances and exact integer equality.
IR expectations never replace expression expectations. Both GPU comparisons
consume one run's returned values, so this does not double hosted GPU requests.
The hosted runner's large-integer transport limitations still apply.

## Execution model

The interpreter loads inputs from meta.json using its explicit data or documented
seeded NumPy generator. It processes kernels in metadata order and chains their
in-memory intermediate arrays. It checks full output initialization.

For each block, a NumPy vector holds each logical thread's value. Tile coordinates
come from row-major unravelling of global worker IDs. Masked loads and stores
touch only active threads. Arithmetic converts operands/results at declared
dtype boundaries; f16 rounds at each operation, and native integers wrap without
a floating-point round trip. StridedLoop carries a typed accumulator separately
for every thread, including identity values for threads with no iterations.

WarpShuffle reads partners from the previous round's values before applying
the stored merge assignments. TreeHalve reads its prepopulated shared slots,
combines halves in the declared order, then broadcasts the root. Whole-vector
statement execution orders shared stores before a converged block barrier.
This reproduces the explicit schedule rather than substituting np.sum.

The checker supports the current Tile, Load, Assign, Write, Cond, StridedLoop,
additive Accum, Smem, Sync, WarpShuffle and flat TreeHalve forms. Supported data
types are f16/f32/f64/i32/i64. Supported arithmetic is add, subtract, multiply,
divide, ReLU, copy, integer bitwise-and and right-shift. Unstamped ordinary Assign
nodes are rejected; merge assignments use the enclosing reduction dtype.

It rejects unsupported statements, symbolic/general index expressions, atomic
or swizzled stores, auxiliary/rasterized tiles, by-value scalar parameters,
nonpositive loop steps, named/warp/divergent barriers, inactive shuffle partners,
out-of-bounds accesses, uninitialized reads and conflicting simultaneous stores.
Loop execution has a bound. Out-of-range float-to-integer conversions and
undefined integer division are rejected. This does not model all GPU races,
exceptional floating-point behavior, instruction fusion or device-specific modes,
and it is not an execution oracle for Emmy-generated CUDA code.

## Validation

The full 80-case suite covers all five types and eight operations at widths
1 and 129, with three rows for row sums. The CPU IR interpreter agrees with the
original expression tolerances on 77 cases and reproduces the three known f64
reduction gaps. The original expected.npz files remain byte-for-byte unchanged.
On the competition GPU, all 80 cases match the CPU IR expectations within the
original tolerances; 77 match the original expression expectations and the
three f64 reduction failures remain. GPU comparisons are saved alongside the
expression checks in `tests/extended/results.json`; the compact summary is
`review/ir-reference-results.json`.

Focused unit tests independently check wide integer low bits/wrapping, f64
precision, f16 operation rounding, masked memory access, uninitialized lanes,
separate-warp simultaneous exchanges, a rounding-sensitive shared tree, saved
expectation isolation and dual reporting that preserves failures. The shared
tree example [1e20, 3, -1e20, 4] returns 7 under the specified halving order,
demonstrating why a generic left-to-right reduction is not an adequate oracle.
The interpreter also matches the saved expression expectations for 35 additional
f16 edge/larger cases, i64 bit-extraction cases and raw numeric-edge fixtures on
the CPU. This CPU result does not bypass the hosted large-integer transport issue.

Run from Ubuntu:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tools/test_extended.py --reuse --ir-reference --runner
```

The full GPU run should exit with status 1 for the three original-expression
failures even if every comparison with the Kernel IR passes. A passing IR check
does not establish a pass against the competition's expected results or hidden
evaluation set.
