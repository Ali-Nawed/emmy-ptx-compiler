# Conditions, ordinary loops and strided loops

Implementation: `index_expr`, `predicate`, `counted_loop`, `label`, and the `Cond` branch of `statement()` in [compile.py](../../compile.py).

## Index expressions

`index_expr()` recursively handles `Var`, integer `Literal`, Python integers, and `BinaryExpr` with `+`, `-` or `*`. It emits signed 32-bit index arithmetic. This is distinct from `Assign`, which implements typed tensor-data arithmetic.

`predicate()` accepts an integer comparison: `==`, `!=`, `<`, `<=`, `>` or `>=`. It emits a `setp.*.s32` predicate. Index `TernaryExpr` uses this predicate plus a `selp.s32` between its two index expressions. Numeric tensor comparisons, internal boolean logic and numeric `where` now have a separate [typed Assign path](14-boolean-comparisons-and-where.md); this does not extend index predicates to arbitrary boolean expression trees.

## Cond

An IR condition becomes two labels and branches:

```ptx
setp.eq.s32 %p1, %lane, 0;
@!%p1 bra ELSE_1;
// then body
bra IF_DONE_2;
ELSE_1:
// else body
IF_DONE_2:
```

The predicate allows each thread to take the correct path. This is used to let one lane per warp write the warp's partial sum. It is not a general SSA join implementation: arbitrary branch-local bindings used after a join are outside the currently validated subset.

## StridedLoop

For a row reduction with 128 cooperative workers:

```text
for column in column_worker .. 1025 step 128:
    value = load x[row, column]
    accumulator += value
```

Worker zero visits columns 0, 128, ..., 1024. Worker one visits 1, 129, ..., 897. The explicit end check handles the irregular tail.

`counted_loop()` emits the following structure:

```text
initialize loop-owned accumulators if seed is enabled
coordinate = start
LOOP:
    if coordinate >= end: branch to LOOP_DONE
    emit body
    coordinate += step
    branch to LOOP
LOOP_DONE:
```

Start, step and optional end are compiled as index expressions. If no end is given, the loop axis's static extent is used. The emitter temporarily binds the loop axis to its mutable coordinate register and restores its previous binding afterward.

## Accumulator ownership and seed

We identify `Accum` nodes owned by this loop using their axes, so an outer loop does not initialize a nested loop's state accidentally. With `seed=True`, each owned state is initialized once before the loop. With `seed=False`, existing state must already be bound or compilation fails.

Addition, multiplication, maximum and minimum accumulators are supported. The update uses the declared combiner and any supplied `base`. The accumulator remains available after its carrying loop for a shuffle or subsequent elementwise computation. See [operation expansion](11-operation-expansion.md) for typed identities and why subtract/divide are not tree combiners.

## Let and Init

`Let` binds a pure literal, a typed variable copy, or an integer index expression. A typed floating literal uses its own width; half literals are rounded before use. `Init` emits an explicitly typed state seed at its current scope. These are existing Emmy statement kinds, not new IR invented by our compiler. Online softmax uses a `Let` to bind its literal one.

## Ordinary Loop

An ordinary Emmy `Loop` runs from zero to its axis's static extent, with step one. It uses the same `counted_loop()` emission structure above; no replacement IR nodes are constructed. Each thread which reaches the loop runs its own sequential iterations. We do not change Emmy's schedule to distribute those iterations across threads.

For `x.sum(-1) + y.sum(-1)`, Emmy can generate a cooperative `StridedLoop` for x followed by an ordinary `Loop` for y. The latter initializes its own accumulator from the actual operation identity, loads each y element, updates the accumulator and advances its index. The completed accumulator is then available to the final addition.

Only immediate `Accum` targets are seeded by an ordinary Loop, matching pinned Emmy's renderer; a nested reduction owns its own seed. `seed=False` requires an already-bound accumulator. Both loop forms restore their enclosing value/type bindings after emission and export only their owned accumulator bindings. Loads and temporary assignments inside the body cannot accidentally leak outside. This also restores a previous binding for a shadowed axis name.

Both loop forms use runtime PTX control flow even when bounds are static. The `unroll` field is a scheduling hint, so ignoring it preserves iteration semantics; unrolling is not implemented. We do not move address arithmetic out of the loop. Ordinary Loop with array-valued carried state is explicitly rejected; supporting scalar `Accum` does not imply support for all stateful Loop programs. Dynamic ordinary-loop extents and signed-32-bit index overflow are outside the current static-index path.

### Genuine generated examples and tests

[tools/test_serial_loops.py](../../tools/test_serial_loops.py) generates PyTorch expressions with two or three row sums using pinned Emmy, packages the unchanged IR with provenance, and compares our PTX against independent NumPy expression expectations and the typed-IR interpreter. The interpreter implements ordinary Loop iteration independently with NumPy vectors; it does not invoke the PTX emitter.

The matrix covers f16/f32/f64/i32/i64, widths 1/33/129, additional width-257 schedules with 32/256 cooperative workers, and a three-sum composition. Width-one expressions may simplify away the ordinary Loop; the report counts actual Loop nodes rather than claiming every fixture exercises it. Integer sums follow PyTorch's i64 promotion. Floating expected values preserve each sum's output rounding before the final addition.

All 18 generated cases passed on the hosted GPU against both the original expression and typed IR. Thirteen contain actual ordinary Loop nodes; five width-one cases simplify them away. All 34 local test methods passed, including fixture provenance, sm_89 assembly, datatype/shape/schedule checks, and verification that loop-local loads and indices do not escape while accumulators do.

The existing 169-case regression and six native-math edge cases were also rerun with this compiler. All 193 cases (including the 18 new cases) match their typed-IR reference on the hosted GPU. There are 186 original-expression passes and the same seven previously recorded f64 expression differences; no new failures were introduced. Competition specifications, the original generator and supplied problems remain unchanged.

Run in the pinned Ubuntu environment:

```bash
python tools/test_serial_loops.py --runner
python -m unittest discover -s tests -p 'test_*.py'
```

See [the generated-case report](../../review/serial-loops/validation.json) and [the regression report](../../review/native-math/serial-loop-regression) for observed results. Only our compiler's PTX is uploaded; no reference CUDA or reference PTX is submitted.

Next: [reductions](07-reductions.md).
