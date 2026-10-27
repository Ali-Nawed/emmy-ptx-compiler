# Launch geometry and Tile coordinates

Implementation: `compile`, `tile_coordinates`, `unsigned_divide` and `unsigned_remainder` in [compile.py](../../compile.py).

## A Tile is a logical work space

`Tile` axes and extents tell us how many logical workers exist and how to decode a flattened worker number. They do not independently choose the launch block size. The runner uses the block/grid in `meta.json`, and cooperative Tiles also specify `block_threads`.

The prologue calculates:

```text
worker = block_id * block_width + thread_id
```

with these PTX builtins and instructions:

```ptx
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mad.lo.s32 %r4, %r2, %r3, %r1;
```

For a plain Tile with `N=64` and a launch of 256 threads, the guard branches workers 64 through 255 to `DONE`. Only 64 workers execute the body. The compiler preserves this guard.

## Decoding coordinates

For extents `(17, 128)`, row-major decoding means:

```text
row = worker // 128
column_worker = worker % 128
```

Worker 259 therefore has coordinates `(2, 3)`. A subsequent strided loop can visit input columns 3, 131, 259, etc.; a cooperative coordinate is not necessarily one final array element.

For general dimensions, `tile_coordinates()` keeps the product of the remaining extents as a stride, divides the worker by that stride and takes the relevant extent remainder. The outermost remainder is unnecessary because the worker is already known to be within the Tile.

An empty axis tuple is a valid scalar Tile: its product is one, it needs no coordinate instructions, and the plain-Tile guard leaves worker zero active.

## Cheap division and remainder

For nonnegative worker indices:

| Calculation | Emission |
| --- | --- |
| `worker // 1` | Reuse `worker` |
| `worker % 1` | Immediate `0` |
| `worker // 128` | `shr.u32 ..., worker, 7` |
| `worker % 128` | `and.b32 ..., worker, 127` |
| Division/remainder by a non-power of two | `div.u32` / `rem.u32` |

This is a local emission optimization, not a rewrite of the input graph.

## Cooperative safety

Reductions use barriers, so we cannot let only part of a block participate accidentally. For cooperative Tiles, `compile()` requires matching `block_threads`, a Tile size divisible by that block size, and an exact grid-times-block match. It also binds:

```text
lane = thread_id & 31
warp = thread_id >> 5
```

Only one-dimensional launches, one nonserial top-level Tile, and ordinary non-rasterized layouts are currently supported. Auxiliary threads are rejected.

Next: [datatypes and registers](03-datatypes-and-registers.md).
