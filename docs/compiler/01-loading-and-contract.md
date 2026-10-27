# Loading the IR and the submission contract

Implementation: `KernelInput`, `load_kernel`, `inspect_kernel`, `emit_ptx`, `main`, and `PTXEmitter.compile` in [compile.py](../../compile.py).

## Start with a real problem

From the repository root in Ubuntu:

```bash
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json --inspect
python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json > /tmp/p00.ptx
```

The first command shows parsed objects on stderr. The second writes only PTX to stdout. `python` must be from an environment containing `emmy-ml==0.3.11`.

## What loading does

```python
graph = Graph.from_dict(json.loads(ir_path.read_text(encoding="utf-8")))
```

This reconstructs actual `KernelOp`, `Tile`, `Load`, `Assign` and other objects. We do not parse the human-readable body string ourselves. `load_kernel` requires exactly one `KernelOp` in this input file, then finds exactly one matching descriptor in its sibling `meta.json` using `kernel.name`.

`KernelInput` bundles the graph, graph node name, typed kernel and matching metadata. Graph node names and PTX entry names are different concepts; we emit the entry name recorded on the kernel.

## Metadata is part of the input

| Field | How we use it |
| --- | --- |
| `entry` | Match the kernel descriptor to the IR kernel name |
| `arg_order` | Exact positional parameter order in the PTX signature |
| `args` | Buffer shapes, element types and argument descriptions |
| `grid`, `block` | Validate the prescribed launch; runtime uses these values |
| `smem_bytes`, `zero_outputs` | Runtime launch/initialization contract, handled by the test runner |

The compiler does not run the kernel or perform output initialization. Metadata constants such as `mean_count` are buffer arguments; arithmetic reads them with `Load`. We do not replace the general program with hardcoded public-problem answers.

## Parameter emission

For each name in `arg_order`, `compile()` creates a parameter called `arg_0`, `arg_1`, etc. It loads that pointer and converts it to a global-memory address:

```ptx
.param .u64 arg_0
// Inside the entry:
ld.param.u64 %rd1, [arg_0];
cvta.to.global.u64 %rd2, %rd1;
```

The `.u64` describes the pointer, not its pointed-to elements. An `f32` buffer still has a 64-bit address. Current compiler support excludes arguments passed as scalar values rather than pointers.

## Why declarations are emitted last in Python

We walk the body first and count the virtual registers and shared arrays it needs. Then `compile()` builds the final text with declarations before the instructions. This is output construction, not a second optimization pass.

The module starts with `.version 8.0`, `.target sm_89`, and `.address_size 64`. It ends the entry with the `DONE` label and `ret`. Unsupported inputs produce a diagnostic on stderr and a nonzero exit status through `main()`.

Next: [launch and Tiles](02-launch-and-tiles.md).
