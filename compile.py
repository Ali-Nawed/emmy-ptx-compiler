#!/usr/bin/env python3
"""Load Emmy's typed Kernel IR and emit our own pointwise/reduction PTX.

    python compile.py problems/p00_scalar_add/k_add_pointwise.ir.json --inspect

Diagnostics go to stderr so stdout remains reserved for generated PTX.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.compiler.ir.kernel.ir import Tile, Smem, Sync, WarpShuffle, TreeHalve
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import Assign, Load, Write, StridedLoop, Accum, Cond


@dataclass
class KernelInput:
    graph: Graph
    node_name: str
    kernel: KernelOp
    metadata: dict[str, Any]


def load_kernel(ir_path: Path) -> KernelInput:
    """Use Emmy's parser, which reconstructs typed statement/expression nodes.

    Do not manually parse the serialized body strings in the JSON.
    """
    graph = Graph.from_dict(json.loads(ir_path.read_text(encoding="utf-8")))
    kernels = [(name, node.op) for name, node in graph.nodes.items()
               if isinstance(node.op, KernelOp)]
    if len(kernels) != 1:
        raise ValueError(f"Expected one KernelOp, found {len(kernels)}")
    node_name, kernel = kernels[0]

    # The problem ships this beside the IR, including in the hidden set.
    meta_path = ir_path.parent / "meta.json"
    problem_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    matches = [item for item in problem_meta["kernels"]
               if item["entry"] == kernel.name]
    if len(matches) != 1:
        raise ValueError(f"Expected one metadata entry for {kernel.name!r}")
    return KernelInput(graph, node_name, kernel, matches[0])


def inspect_kernel(source: KernelInput) -> None:
    """Show the real parsed objects before implementing statement handlers."""
    print(f"Graph node: {source.node_name}", file=sys.stderr)
    print(f"Kernel entry: {source.kernel.name}", file=sys.stderr)
    for field in ("arg_order", "grid", "block", "smem_bytes", "zero_outputs"):
        print(f"{field}: {source.metadata[field]}", file=sys.stderr)
    print("\nTyped kernel body (nested objects are already parsed):", file=sys.stderr)
    for statement in source.kernel.body:
        pprint(statement, stream=sys.stderr, width=100, sort_dicts=False)


class PTXEmitter:
    """Direct emitter for f16/f32 pointwise kernels and f32 reduction state.

    Keep this first version deliberately narrow. Unsupported forms fail rather
    than generating plausible but incorrect PTX. No problem-name recognition.
    """

    def __init__(self, source: KernelInput):
        self.source = source
        self.lines: list[str] = []
        self.counts = {"r": 0, "rd": 0, "f": 0, "h": 0, "p": 0}
        self.values: dict[str, str] = {}
        # Floating values live in f32 registers; f16 values are rounded at each
        # typed operation boundary. h registers transport the actual half bits.
        self.value_dtypes: dict[str, str] = {}
        self.pointers: dict[str, str] = {}
        self.buffers = {arg["name"]: arg for arg in source.metadata["args"]}
        self.shared: dict[str, str] = {}
        self.declarations: list[str] = []
        self.label_count = 0

    def label(self, prefix: str) -> str:
        self.label_count += 1
        return f"{prefix}_{self.label_count}"

    @staticmethod
    def float_literal(value) -> str:
        bits = struct.unpack("!I", struct.pack("!f", float(value)))[0]
        return f"0f{bits:08X}"

    def statements(self, body) -> None:
        for stmt in body:
            self.statement(stmt)

    def reg(self, kind: str) -> str:
        """Allocate a fresh virtual register, not a physical GPU register."""
        self.counts[kind] += 1
        return f"%{kind}{self.counts[kind]}"

    def emit(self, instruction: str) -> None:
        self.lines.append(f"    {instruction}")

    @staticmethod
    def element_bytes(dtype: str) -> int:
        if dtype not in ("f16", "f32"):
            raise NotImplementedError(f"Unsupported buffer dtype: {dtype}")
        return 2 if dtype == "f16" else 4

    def round_half(self, value: str) -> str:
        """Round to binary16, retaining the rounded value in an f32 register."""
        half, rounded = self.reg("h"), self.reg("f")
        self.emit(f"cvt.rn.f16.f32 {half}, {value};")
        self.emit(f"cvt.f32.f16 {rounded}, {half};")
        return rounded

    def index_expr(self, expr) -> str:
        """Return a 32-bit register/immediate for an integer index expression."""
        if isinstance(expr, Var):
            return self.values[expr.name]
        if isinstance(expr, Literal) and isinstance(expr.value, int):
            return str(expr.value)
        if isinstance(expr, int):
            return str(expr)
        if isinstance(expr, BinaryExpr):
            instructions = {"+": "add.s32", "-": "sub.s32", "*": "mul.lo.s32"}
            if expr.op in instructions:
                left = self.index_expr(expr.left)
                right = self.index_expr(expr.right)
                result = self.reg("r")
                self.emit(f"{instructions[expr.op]} {result}, {left}, {right};")
                return result
        raise NotImplementedError(f"Unsupported index expression: {expr!r}")

    def predicate(self, expr) -> str:
        if not isinstance(expr, BinaryExpr):
            raise NotImplementedError(f"Unsupported condition: {expr!r}")
        comparisons = {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}
        if expr.op not in comparisons:
            raise NotImplementedError(f"Unsupported comparison: {expr.op}")
        left, right = self.index_expr(expr.left), self.index_expr(expr.right)
        result = self.reg("p")
        self.emit(f"setp.{comparisons[expr.op]}.s32 {result}, {left}, {right};")
        return result

    def address(self, name: str, index) -> str:
        """IR indices count elements; PTX addresses count bytes."""
        buffer = self.buffers[name]
        if buffer.get("scalar"):
            raise NotImplementedError("Scalar by-value arguments are unsupported")
        element_bytes = self.element_bytes(buffer["dtype"])
        shape = buffer["shape"]
        if not index and math.prod(shape) == 1:
            element = "0"  # A scalar reduction result can have index=().
        elif len(index) == len(shape):
            # Row-major flattening: [row, col] -> row * columns + col.
            element = self.index_expr(index[0])
            for coordinate, extent in zip(index[1:], shape[1:]):
                scaled, flat = self.reg("r"), self.reg("r")
                self.emit(f"mul.lo.s32 {scaled}, {element}, {extent};")
                self.emit(f"add.s32 {flat}, {scaled}, {self.index_expr(coordinate)};")
                element = flat
        else:
            raise NotImplementedError("Buffer rank and index rank must agree")
        # mul.wide's first source must be a register, including for constant indices.
        element_reg = self.reg("r")
        self.emit(f"mov.s32 {element_reg}, {element};")
        byte_offset = self.reg("rd")
        self.emit(f"mul.wide.s32 {byte_offset}, {element_reg}, {element_bytes};")
        address = self.reg("rd")
        base = self.shared.get(name, self.pointers.get(name))
        self.emit(f"add.u64 {address}, {base}, {byte_offset};")
        return address

    def memory_space(self, name: str) -> str:
        return "shared" if name in self.shared else "global"

    def merge_states(self, stmt) -> None:
        """Run the IR's merge program, then copy results into carried registers."""
        targets = {name: self.values[name] for name in stmt.state}
        for assignment in stmt.combine_states:
            self.statement(assignment)
        # Complete the program before updating any original state register.
        for name, register in targets.items():
            self.emit(f"mov.f32 {register}, {self.values[name]};")
        self.values.update(targets)

    def strided_loop(self, stmt) -> None:
        # Only seed state owned by this loop, not a nested loop's accumulator.
        owned = [s for s in stmt.body.iter() if isinstance(s, Accum) and
                 (not s.axes or stmt.axis.name in s.axes)]
        seeded = set()
        for accumulator in owned:
            if stmt.seed and accumulator.name not in seeded:
                if accumulator.dtype is not None and accumulator.dtype.name != "f32":
                    raise NotImplementedError("Only f32 reduction state is supported")
                if accumulator.op.name != "add":
                    raise NotImplementedError("Only sum accumulators are supported initially")
                target = self.reg("f")
                self.values[accumulator.name] = target
                self.value_dtypes[accumulator.name] = "f32"
                self.emit(f"mov.f32 {target}, {self.float_literal(accumulator.op.identity)};")
                seeded.add(accumulator.name)
            elif not stmt.seed and accumulator.name not in self.values:
                raise ValueError(f"Uninitialized accumulator: {accumulator.name}")
        start, step = self.index_expr(stmt.start), self.index_expr(stmt.step)
        end = self.index_expr(stmt.end) if stmt.end is not None else str(stmt.axis.extent.as_static())
        coordinate = self.reg("r")
        previous = self.values.get(stmt.axis.name)
        self.values[stmt.axis.name] = coordinate
        self.emit(f"mov.s32 {coordinate}, {start};")
        head, done = self.label("LOOP"), self.label("LOOP_DONE")
        self.lines.append(f"{head}:")
        finished = self.reg("p")
        self.emit(f"setp.ge.s32 {finished}, {coordinate}, {end};")
        self.emit(f"@{finished} bra {done};")
        self.statements(stmt.body)
        self.emit(f"add.s32 {coordinate}, {coordinate}, {step};")
        self.emit(f"bra {head};")
        self.lines.append(f"{done}:")
        if previous is None:
            self.values.pop(stmt.axis.name)
        else:
            self.values[stmt.axis.name] = previous

    def tree_halve(self, stmt) -> None:
        if stmt.inner is not None or stmt.tid_var != "warp" or stmt.dtype.name != "f32":
            raise NotImplementedError("Only flat f32 trees of per-warp partials are supported")
        if stmt.length < 1 or stmt.length & (stmt.length - 1):
            raise ValueError("TreeHalve length must be a power of two")
        half = stmt.length // 2
        while half:
            saved = self.values.copy()
            saved_dtypes = self.value_dtypes.copy()
            active, leader, writer = self.reg("p"), self.reg("p"), self.reg("p")
            self.emit(f"setp.lt.u32 {active}, {self.values['warp']}, {half};")
            self.emit(f"setp.eq.u32 {leader}, {self.values['lane']}, 0;")
            self.emit(f"and.pred {writer}, {active}, {leader};")
            skip = self.label("TREE_SKIP")
            self.emit(f"@!{writer} bra {skip};")
            partner = self.reg("r")
            self.emit(f"add.s32 {partner}, {self.values['warp']}, {half};")
            partner_name = f"_tree_partner_{self.label_count}"
            self.values[partner_name] = partner
            addresses = []
            for buf, state, other in zip(stmt.bufs, stmt.state, stmt.state_b, strict=True):
                left = self.address(buf, (Var("warp"),))
                right = self.address(buf, (Var(partner_name),))
                addresses.append(left)
                self.values[state], self.values[other] = self.reg("f"), self.reg("f")
                self.value_dtypes[state] = self.value_dtypes[other] = "f32"
                self.emit(f"ld.shared.f32 {self.values[state]}, [{left}];")
                self.emit(f"ld.shared.f32 {self.values[other]}, [{right}];")
            self.merge_states(stmt)
            for address, state in zip(addresses, stmt.state, strict=True):
                self.emit(f"st.shared.f32 [{address}], {self.values[state]};")
            self.lines.append(f"{skip}:")
            self.sync(stmt.barrier_id, stmt.barrier_count)
            self.values = saved
            self.value_dtypes = saved_dtypes
            half //= 2
        # Every lane receives the block result, as specified by TreeHalve.
        for buf, state in zip(stmt.bufs, stmt.state, strict=True):
            root = self.address(buf, (Literal(0, "int"),))
            self.emit(f"ld.shared.f32 {self.values[state]}, [{root}];")

    def sync(self, barrier_id=0, count=None) -> None:
        if barrier_id and count is None:
            raise ValueError("Named barriers require a participating thread count")
        suffix = f", {count}" if count is not None else ""
        self.emit(f"bar.sync {barrier_id}{suffix};")

    def statement(self, stmt) -> None:
        if isinstance(stmt, Load):
            dtype = self.buffers[stmt.input]["dtype"]
            width = self.element_bytes(dtype)
            if stmt.dtype is not None and stmt.dtype.name != dtype:
                raise NotImplementedError("Load dtype must match its source buffer dtype")
            address = self.address(stmt.input, stmt.index)
            for offset, name in enumerate(stmt.names):
                result = self.reg("f")
                location = f"[{address}+{offset * width}]"
                space = self.memory_space(stmt.input)
                if dtype == "f16":
                    half = self.reg("h")
                    self.emit(f"ld.{space}.b16 {half}, {location};")
                    self.emit(f"cvt.f32.f16 {result}, {half};")
                else:
                    self.emit(f"ld.{space}.f32 {result}, {location};")
                self.values[name] = result
                self.value_dtypes[name] = dtype
        elif isinstance(stmt, Assign):
            dtype = stmt.dtype.name if stmt.dtype is not None else "f32"
            if dtype not in ("f16", "f32"):
                raise NotImplementedError(f"Unsupported arithmetic: {stmt.op}")
            args = [self.values[arg] for arg in stmt.args]
            operation = stmt.op.name
            if dtype == "f16" and operation != "copy":
                # Emmy converts operands to the assignment dtype before compute.
                args = [value if self.value_dtypes[name] == "f16" else self.round_half(value)
                        for name, value in zip(stmt.args, args, strict=True)]
            result = self.reg("f")
            binary = {"add": "add.rn.f32", "subtract": "sub.rn.f32",
                      "multiply": "mul.rn.f32", "divide": "div.rn.f32"}
            if operation in binary and len(args) == 2:
                # Explicit rounding keeps multiply and add separate (no FMA contraction).
                self.emit(f"{binary[operation]} {result}, {args[0]}, {args[1]};")
            elif operation == "relu" and len(args) == 1:
                # ReLU: x < 0 ? +0.0 : x. Preserve NaN and signed zero from x.
                negative = self.reg("p")
                self.emit(f"setp.lt.f32 {negative}, {args[0]}, 0f00000000;")
                self.emit(f"selp.f32 {result}, 0f00000000, {args[0]}, {negative};")
            elif operation == "copy" and len(args) == 1:
                self.emit(f"mov.f32 {result}, {args[0]};")
            else:
                raise NotImplementedError(f"Unsupported arithmetic: {stmt.op}")
            # A half-precision intermediate must round NOW, not just at the final
            # store, otherwise a chain such as multiply -> add changes semantics.
            self.values[stmt.name] = self.round_half(result) if dtype == "f16" else result
            self.value_dtypes[stmt.name] = dtype
        elif isinstance(stmt, Write):
            if (stmt.atomic or stmt.swizzle != "NONE" or
                    (stmt.value_dtype is not None and stmt.value_dtype.name not in ("f16", "f32"))):
                raise NotImplementedError("Only plain f16/f32 writes are supported")
            dtype = self.buffers[stmt.output]["dtype"]
            width = self.element_bytes(dtype)
            address = self.address(stmt.output, stmt.index)
            # Scalarize grouped stores for now: avoids requiring proof of vector alignment.
            for offset, name in enumerate(stmt.values):
                location = f"[{address}+{offset * width}]"
                space = self.memory_space(stmt.output)
                if dtype == "f16":
                    half = self.reg("h")
                    self.emit(f"cvt.rn.f16.f32 {half}, {self.values[name]};")
                    self.emit(f"st.{space}.b16 {location}, {half};")
                else:
                    self.emit(f"st.{space}.f32 {location}, {self.values[name]};")
        elif isinstance(stmt, StridedLoop):
            self.strided_loop(stmt)
        elif isinstance(stmt, Accum):
            if stmt.op.name != "add":
                raise NotImplementedError("Only sum accumulators are supported initially")
            target = self.values[stmt.name]
            base = self.values[stmt.base] if stmt.base is not None else target
            self.emit(f"add.rn.f32 {target}, {base}, {self.values[stmt.value]};")
        elif isinstance(stmt, Cond):
            condition = self.predicate(stmt.cond)
            otherwise, done = self.label("ELSE"), self.label("IF_DONE")
            self.emit(f"@!{condition} bra {otherwise};")
            self.statements(stmt.body)
            self.emit(f"bra {done};")
            self.lines.append(f"{otherwise}:")
            self.statements(stmt.else_body)
            self.lines.append(f"{done}:")
        elif isinstance(stmt, Smem):
            if stmt.dtype != "float":
                raise NotImplementedError("Only float shared storage is supported")
            symbol = f"scratch_{len(self.shared)}"
            size = math.prod(stmt.extents) * 4
            self.declarations.append(f"    .shared .align {stmt.align or 4} .b8 {symbol}[{size}];")
            base = self.reg("rd")
            self.emit(f"mov.u64 {base}, {symbol};")
            self.shared[stmt.name] = base
            self.buffers[stmt.name] = {"dtype": "f32", "shape": list(stmt.extents)}
        elif isinstance(stmt, Sync):
            if stmt.warp:
                raise NotImplementedError("Warp barriers are not supported yet")
            self.sync(stmt.barrier_id, stmt.count)
        elif isinstance(stmt, WarpShuffle):
            if (stmt.dtype.name != "f32" or stmt.length < 1 or stmt.length > 32
                    or stmt.length & (stmt.length - 1)):
                raise NotImplementedError("Shuffle requires f32 and a power-of-two width <= 32")
            mask = self.reg("r")
            self.emit(f"activemask.b32 {mask};")
            distance = stmt.length // 2
            while distance:
                for state, other in zip(stmt.state, stmt.state_b, strict=True):
                    partner = self.reg("f")
                    self.emit(f"shfl.sync.bfly.b32 {partner}, {self.values[state]}, {distance}, 31, {mask};")
                    self.values[other] = partner
                    self.value_dtypes[other] = "f32"
                self.merge_states(stmt)
                distance //= 2
        elif isinstance(stmt, TreeHalve):
            self.tree_halve(stmt)
        else:
            raise NotImplementedError(f"Unsupported statement: {type(stmt).__name__}")

    def compile(self) -> str:
        kernel, meta = self.source.kernel, self.source.metadata
        if kernel.serial or len(kernel.body) != 1 or not isinstance(kernel.body[0], Tile):
            raise NotImplementedError("Expected one nonserial Tile")
        tile = kernel.body[0]
        # An empty axis tuple is a valid scalar tile: product(()) == 1 worker.
        # The normal bounds guard below activates only global worker zero.
        if tile.aux_threads or tile.raster_group is not None:
            raise NotImplementedError("Auxiliary threads and rasterized tiles are unsupported")
        if meta["block"][1:] != [1, 1] or meta["grid"][1:] != [1, 1]:
            raise NotImplementedError("Only one-dimensional launches are supported")
        if tile.block_threads is not None:
            # Block barriers must not be reached by only part of a block.
            if (meta["block"][0] != tile.block_threads
                    or tile.n_elements % tile.block_threads
                    or meta["grid"][0] * tile.block_threads != tile.n_elements):
                raise NotImplementedError("Cooperative tiles require complete blocks covering the tile")

        # Load the ABI parameters in meta.json order, using safe generated names.
        for position, name in enumerate(meta["arg_order"]):
            if self.buffers[name].get("scalar"):
                raise NotImplementedError("Scalar by-value parameters are unsupported")
            param, pointer = self.reg("rd"), self.reg("rd")
            self.emit(f"ld.param.u64 {param}, [arg_{position}];")
            self.emit(f"cvta.to.global.u64 {pointer}, {param};")
            self.pointers[name] = pointer

        tid, block, width, worker = (self.reg("r") for _ in range(4))
        self.emit(f"mov.u32 {tid}, %tid.x;")
        self.emit(f"mov.u32 {block}, %ctaid.x;")
        self.emit(f"mov.u32 {width}, %ntid.x;")
        self.emit(f"mad.lo.s32 {worker}, {block}, {width}, {tid};")
        if tile.block_threads is None:
            outside = self.reg("p")
            self.emit(f"setp.ge.u32 {outside}, {worker}, {tile.n_elements};")
            self.emit(f"@{outside} bra DONE;")
        if len(tile.axes) == 1:
            self.values[tile.axes[0].name] = worker
        else:
            # Decode a row-major logical work ID into the Tile's axis coordinates.
            stride = tile.n_elements
            for axis, extent in zip(tile.axes, tile.extents, strict=True):
                stride //= extent
                quotient, coordinate = self.reg("r"), self.reg("r")
                self.emit(f"div.u32 {quotient}, {worker}, {stride};")
                self.emit(f"rem.u32 {coordinate}, {quotient}, {extent};")
                self.values[axis.name] = coordinate
        if tile.block_threads is not None:
            lane, warp = self.reg("r"), self.reg("r")
            self.emit(f"and.b32 {lane}, {tid}, 31;")
            self.emit(f"shr.u32 {warp}, {tid}, 5;")
            self.values.update(lane=lane, warp=warp)
        self.statements(tile.body)
        self.lines.extend(["DONE:", "    ret;"])

        # Emit the body first so we know how many registers to declare.
        signature = ",\n".join(f"    .param .u64 arg_{i}"
                                 for i, _ in enumerate(meta["arg_order"]))
        types = {"r": "b32", "rd": "b64", "f": "f32", "h": "b16", "p": "pred"}
        declarations = [f"    .reg .{types[k]} %{k}<{n + 1}>;"
                        for k, n in self.counts.items() if n]
        return "\n".join([
            ".version 8.0", ".target sm_89", ".address_size 64", "",
            f".visible .entry {kernel.name}(\n{signature}\n)", "{",
            *declarations, *self.declarations, "", *self.lines, "}", "",
        ])


def emit_ptx(source: KernelInput) -> str:
    return PTXEmitter(source).compile()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir_path", type=Path)
    parser.add_argument("--inspect", action="store_true", help="show parsed IR on stderr")
    args = parser.parse_args()
    try:
        source = load_kernel(args.ir_path)
        if args.inspect:
            inspect_kernel(source)
        else:
            print(emit_ptx(source))
    except (OSError, ValueError, KeyError, NotImplementedError) as exc:
        print(f"compile.py: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
