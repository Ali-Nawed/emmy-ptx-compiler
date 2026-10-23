# Kernel IR -> PTX: node checklist

Source: the competition repository's TASK.md (emmy-ml 0.3.11).

Use Emmy's Graph.from_dict to deserialize the JSON. These node classes already exist in Emmy; we need emitter handlers for them, not a second parser or duplicate node classes. Fields below follow TASK.md; inspect the installed classes for exact Python types and defaults.

Suggested interface: emit_stmt(node), emit_expr(expr), emit_address(buffer, index). Each handler emits our own PTX. This document is a plan, not an implemented compiler.

## 1. Kernel container

| Node | Fields | Handler responsibility |
|---|---|---|
| KernelOp | name, body, zero_delegated, serial | Emit directives, entry signature, register declarations, body and return. Take parameter order from the matching kernel in neighboring meta.json. serial is empty in the evaluation problems. |

## 2. Pointwise: implement first

| Node | Fields | Handler responsibility |
|---|---|---|
| Tile | axes, body, block_threads, aux_threads, raster_axes, raster_group, raster_orient | Decode logical axis coordinates from launch/thread coordinates; derive bounds guards and visit body. Read axis metadata rather than treating every Tile as a simple linear thread ID. |
| Load | names, input, index, dtype | Flatten indices, convert element offsets to bytes, load global or shared memory, bind result names to registers. Support scalar and grouped consecutive loads. |
| Assign | name, op, args, dtype | Lower the arithmetic operation and bind its result register. Dispatch by op; different arithmetic operations share this node kind. |
| Write | output, index, values, value_dtype, atomic, swizzle | Compute address and store scalar or consecutive vector values. Later support shared stores, atomic addition and any supplied address swizzle. |

Expressions needed alongside these statements:

| Expression | Fields | Handler responsibility |
|---|---|---|
| Var | name | Look up an axis coordinate or computed value. |
| Literal | value, dtype | Produce a correctly typed immediate, or materialize a register when needed. |
| BinaryExpr | op, left, right | Recursively emit arithmetic, comparisons and logical expressions; comparisons produce predicates. |
| TernaryExpr | cond, if_true, if_false | Select between values using a predicate; needed in more advanced problems. |

For p00 specifically: KernelOp -> Tile -> Load / Assign / Write, with Var / Literal / BinaryExpr for indices. There are 64 logical workers, two elements per worker, and 256 launched threads. Guard the inactive threads. The final pair of results is one grouped write.

## 3. Reductions: finish the MVP

| Node | Fields | Handler responsibility |
|---|---|---|
| StridedLoop | axis, start, step, body, unroll, end, seed | Initialize loop coordinate, test bound, emit body, increment and branch. Preserve accumulator lifetime across iterations. |
| Accum | name, value, op, dtype, axes, base | Initialize reduction state in the correct enclosing scope and update it. Respect the operation identity, base and supplied value semantics; do not assume every update is simply add(acc, value). |
| Cond | cond, body, else_body | Emit conditional control flow and optional else branch. |
| Smem | name, extents, dtype, align | Declare shared storage, record shape and alignment, and register its memory space. |
| Sync | barrier_id, count, warp | Emit the appropriate block, named or warp barrier. Preserve required thread participation. |
| WarpShuffle | state, state_b, combine_states, length, dtype | Shuffle partner state and execute the supplied combine computation at each butterfly step. Support multiple coupled state values for softmax later. |
| TreeHalve | bufs, state, state_b, combine_states, length, tid_var, dtype, barrier_id, barrier_count, inner | Combine shared partial states in a halving tree, synchronize correctly and make the final result available as required. |

MVP problems: p00, p01, p02, p03, r01, r02, r03, r04.

## 4. Matmul, fused kernels and attention

| Node | Fields | Handler responsibility |
|---|---|---|
| Let | name, value, dtype | Evaluate an expression and bind its value. Used for literal bindings in softmax and attention. |
| Init | name, identity, dtype | Allocate mutable state and initialize it to the given identity. |
| Loop | axis, body, unroll, seed | Emit a loop over the axis extent, respecting carried state and initialization. |
| CpAsyncCopy | smem, smem_index, src, src_index, nbytes, swizzle, lane_index, lane_rows | Compute global and shared addresses, including lane offsets/swizzle, and emit asynchronous copies of the specified byte width. |
| CpAsyncCommit | (none) | Emit cp.async.commit_group. |
| CpAsyncWait | group | Emit cp.async.wait_group with the specified outstanding-group limit. |

Extend existing handlers too: Write needs atomic additions; Smem/address generation need advanced layouts; Assign needs operations such as maximum, exp, reciprocal and sqrt; reduction handlers need coupled combine states. Operation coverage must follow the IR, not a fixed list of problem names.

## 5. Compiler context (our data, not additional IR nodes)

- Value environment: IR names -> typed virtual registers or immediates.
- Axis environment: axis names -> coordinate registers and extent information.
- Buffer table: names -> pointer, shape, dtype, memory space and alignment.
- Register counters: floating-point, integer, address and predicate registers.
- Label counter and emitted instruction/declaration lists.
- Kernel metadata: entry, arg_order, grid, block, smem_bytes, zero_outputs.
- Reduction state/scope information: where state is initialized and updated.

Keep host launch requirements separate from device code: meta.json's zero_outputs is handled by the runner; it is not itself an IR statement.

## 6. Explicitly outside evaluation scope

TASK.md marks these as absent from both open and hidden tests:

- State/control: Pre, Carry, Select, ZeroPrologue.
- TMA/barriers: TmaDescriptor, TmaLoad, MbarrierInit, MbarrierArriveExpectTx, MbarrierArrive, MbarrierWait, SetMaxNReg.
- Tensor-core fragments: RegFragment, LdmatrixLoad, BlockScaleLoad, MmaSyncPtx, FragmentPromote, FragmentApply, FragmentRowReduce, FragmentRepack, RegStore; supporting Lambda and FragLayout.
- Warp-group operations: WgmmaDescriptor, WgmmaMma, WgmmaFence, WgmmaCommit, WgmmaWait.
- Expressions: Builtin, FuncCallExpr, CastExpr, FlatIndex.

Fail clearly on unsupported nodes rather than silently skipping them. Builtin expression nodes are out of scope, but our Tile emitter still needs to read PTX hardware registers such as %tid.x and %ctaid.x.

## Lowering rules to keep visible

1. Entry name comes from KernelOp; argument order comes from meta.json.
2. IR indices are element offsets; memory addresses use byte offsets.
3. Flatten multidimensional coordinates using buffer shape.
4. Preserve arithmetic semantics initially; avoid automatic reassociation or FMA contraction.
5. Derive bounds from schedule metadata; do not insert divergent exits that invalidate a later barrier.
6. Emit ASCII PTX with valid version/target/address-size directives.
7. Allocate virtual registers; let ptxas do physical register allocation.
8. Never lower through Emmy's CUDA renderer, nvcc or Triton in the submission compiler.
