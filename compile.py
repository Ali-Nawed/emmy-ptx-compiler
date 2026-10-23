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
    """Direct typed emitter for the runner's five scalar datatypes.

    Keep this first version deliberately narrow. Unsupported forms fail rather
    than generating plausible but incorrect PTX. No problem-name recognition.
    """

    def __init__(self, source: KernelInput):
        self.source = source
        self.lines: list[str] = []
        self.counts = {"r": 0, "rd": 0, "f": 0, "h": 0, "d": 0, "l": 0, "p": 0}
        self.values: dict[str, str] = {}
        # f16 values use rounded f32 working registers. Other data types use
        # native-width registers; pointers and integer data have separate banks.
        self.value_dtypes: dict[str, str] = {}
        self.pointers: dict[str, str] = {}
        self.buffers = {arg["name"]: arg for arg in source.metadata["args"]}
        self.shared: dict[str, str] = {}
        self.declarations: list[str] = []
        self.label_count = 0
        self.combine_dtype = None

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
        widths = {"f16": 2, "f32": 4, "f64": 8, "i32": 4, "i64": 8}
        if dtype not in widths:
            raise NotImplementedError(f"Unsupported buffer dtype: {dtype}")
        return widths[dtype]

    @staticmethod
    def ptx_type(dtype: str) -> str:
        return {"f16": "f32", "f32": "f32", "f64": "f64", "i32": "s32", "i64": "s64"}[dtype]

    def data_reg(self, dtype: str) -> str:
        self.element_bytes(dtype)
        return self.reg({"f16": "f", "f32": "f", "f64": "d", "i32": "r", "i64": "l"}[dtype])

    def literal(self, value, dtype: str) -> str:
        if dtype.startswith("i"):
            return str(int(value))
        if dtype == "f64":
            bits = struct.unpack("!Q", struct.pack("!d", float(value)))[0]
            return f"0d{bits:016X}"
        return self.float_literal(value)

    def convert(self, value: str, source: str, destination: str) -> str:
        """Typed numeric conversion; do not route i64/f64 through f32."""
        self.element_bytes(source)
        self.element_bytes(destination)
        if source == destination:
            return value
        if destination == "f16":
            # Half bits are widened exactly for our working representation.
            half, result = self.reg("h"), self.reg("f")
            self.emit(f"cvt.rn.f16.{self.ptx_type(source)} {half}, {value};")
            self.emit(f"cvt.f32.f16 {result}, {half};")
            return result
        actual_source = self.ptx_type(source)
        actual_destination = self.ptx_type(destination)
        if actual_source == actual_destination:
            return value  # f16 -> f32 is already exact in the working register.
        result = self.data_reg(destination)
        if destination.startswith("i"):
            rounding = ".rzi" if source.startswith("f") else ""
        else:
            rounding = ".rn" if source.startswith("i") or (source == "f64" and destination == "f32") else ""
        self.emit(f"cvt{rounding}.{actual_destination}.{actual_source} {result}, {value};")
        return result

    def operand(self, name: str, dtype: str) -> str:
        return self.convert(self.values[name], self.value_dtypes[name], dtype)

    def move(self, target: str, source: str, dtype: str) -> None:
        self.emit(f"mov.{self.ptx_type(dtype)} {target}, {source};")

    def arithmetic(self, operation: str, operands: list[str], dtype: str) -> str:
        kind = self.ptx_type(dtype)
        integer = dtype.startswith("i")
        binary = {"add": "add", "subtract": "sub", "multiply": "mul", "divide": "div"}
        result = self.data_reg(dtype)
        if operation in binary and len(operands) == 2:
            opcode = binary[operation]
            suffix = ".lo" if integer and operation == "multiply" else "" if integer else ".rn"
            self.emit(f"{opcode}{suffix}.{kind} {result}, {operands[0]}, {operands[1]};")
        elif operation == "relu" and len(operands) == 1:
            negative = self.reg("p")
            zero = self.literal(0, dtype)
            self.emit(f"setp.lt.{kind} {negative}, {operands[0]}, {zero};")
            self.emit(f"selp.{kind} {result}, {zero}, {operands[0]}, {negative};")
        elif operation == "copy" and len(operands) == 1:
            self.move(result, operands[0], dtype)
        elif operation == "bitwise_and" and integer and len(operands) == 2:
            self.emit(f"and.b{self.element_bytes(dtype) * 8} {result}, {operands[0]}, {operands[1]};")
        elif operation == "right_shift" and integer and len(operands) == 2:
            # PTX's shift count is 32-bit even when the shifted value is 64-bit.
            count = operands[1]
            if dtype == "i64":
                clamped = self.data_reg("i64")
                self.emit(f"min.u64 {clamped}, {count}, 64;")
                count = self.convert(clamped, "i64", "i32")
            self.emit(f"shr.{kind} {result}, {operands[0]}, {count};")
        else:
            raise NotImplementedError(f"Unsupported arithmetic: {operation}/{dtype}")
        return self.round_half(result) if dtype == "f16" else result

    def memory_load(self, location: str, space: str, dtype: str) -> str:
        result = self.data_reg(dtype)
        if dtype == "f16":
            half = self.reg("h")
            self.emit(f"ld.{space}.b16 {half}, {location};")
            self.emit(f"cvt.f32.f16 {result}, {half};")
        else:
            self.emit(f"ld.{space}.{self.ptx_type(dtype)} {result}, {location};")
        return result

    def memory_store(self, location: str, space: str, dtype: str, value: str) -> None:
        if dtype == "f16":
            half = self.reg("h")
            self.emit(f"cvt.rn.f16.f32 {half}, {value};")
            self.emit(f"st.{space}.b16 {location}, {half};")
        else:
            self.emit(f"st.{space}.{self.ptx_type(dtype)} {location}, {value};")

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
        target_types = {name: self.value_dtypes[name] for name in stmt.state}
        previous_dtype = self.combine_dtype
        self.combine_dtype = stmt.dtype.name
        try:
            for assignment in stmt.combine_states:
                self.statement(assignment)
        finally:
            self.combine_dtype = previous_dtype
        # Complete the program before updating any original state register.
        for name, register in targets.items():
            value = self.operand(name, target_types[name])
            self.move(register, value, target_types[name])
        self.values.update(targets)
        self.value_dtypes.update(target_types)

    def strided_loop(self, stmt) -> None:
        # Only seed state owned by this loop, not a nested loop's accumulator.
        owned = [s for s in stmt.body.iter() if isinstance(s, Accum) and
                 (not s.axes or stmt.axis.name in s.axes)]
        seeded = set()
        for accumulator in owned:
            if stmt.seed and accumulator.name not in seeded:
                dtype = accumulator.dtype.name if accumulator.dtype is not None else "f32"
                if accumulator.op.name != "add":
                    raise NotImplementedError("Only sum accumulators are supported initially")
                target = self.data_reg(dtype)
                self.values[accumulator.name] = target
                self.value_dtypes[accumulator.name] = dtype
                self.move(target, self.literal(accumulator.op.identity, dtype), dtype)
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
        if stmt.inner is not None or stmt.tid_var != "warp":
            raise NotImplementedError("Only flat trees of per-warp partials are supported")
        dtype = stmt.dtype.name
        self.element_bytes(dtype)
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
                storage_dtype = self.buffers[buf]["dtype"]
                self.values[state] = self.convert(self.memory_load(f"[{left}]", "shared", storage_dtype), storage_dtype, dtype)
                self.values[other] = self.convert(self.memory_load(f"[{right}]", "shared", storage_dtype), storage_dtype, dtype)
                self.value_dtypes[state] = self.value_dtypes[other] = dtype
            self.merge_states(stmt)
            for address, buf, state in zip(addresses, stmt.bufs, stmt.state, strict=True):
                storage_dtype = self.buffers[buf]["dtype"]
                self.memory_store(f"[{address}]", "shared", storage_dtype, self.operand(state, storage_dtype))
            self.lines.append(f"{skip}:")
            self.sync(stmt.barrier_id, stmt.barrier_count)
            self.values = saved
            self.value_dtypes = saved_dtypes
            half //= 2
        # Every lane receives the block result, as specified by TreeHalve.
        for buf, state in zip(stmt.bufs, stmt.state, strict=True):
            root = self.address(buf, (Literal(0, "int"),))
            storage_dtype = self.buffers[buf]["dtype"]
            value = self.convert(self.memory_load(f"[{root}]", "shared", storage_dtype), storage_dtype, self.value_dtypes[state])
            self.move(self.values[state], value, self.value_dtypes[state])

    def sync(self, barrier_id=0, count=None) -> None:
        if barrier_id and count is None:
            raise ValueError("Named barriers require a participating thread count")
        suffix = f", {count}" if count is not None else ""
        self.emit(f"bar.sync {barrier_id}{suffix};")

    def statement(self, stmt) -> None:
        if isinstance(stmt, Load):
            dtype = self.buffers[stmt.input]["dtype"]
            width = self.element_bytes(dtype)
            result_dtype = stmt.dtype.name if stmt.dtype is not None else dtype
            address = self.address(stmt.input, stmt.index)
            for offset, name in enumerate(stmt.names):
                location = f"[{address}+{offset * width}]"
                space = self.memory_space(stmt.input)
                loaded = self.memory_load(location, space, dtype)
                self.values[name] = self.convert(loaded, dtype, result_dtype)
                self.value_dtypes[name] = result_dtype
        elif isinstance(stmt, Assign):
            dtype = stmt.dtype.name if stmt.dtype is not None else self.combine_dtype or "f32"
            operands = [self.operand(name, dtype) for name in stmt.args]
            self.values[stmt.name] = self.arithmetic(stmt.op.name, operands, dtype)
            self.value_dtypes[stmt.name] = dtype
        elif isinstance(stmt, Write):
            if stmt.atomic or stmt.swizzle != "NONE":
                raise NotImplementedError("Only plain writes are supported")
            dtype = self.buffers[stmt.output]["dtype"]
            width = self.element_bytes(dtype)
            address = self.address(stmt.output, stmt.index)
            # Scalarize grouped stores for now: avoids requiring proof of vector alignment.
            for offset, name in enumerate(stmt.values):
                location = f"[{address}+{offset * width}]"
                space = self.memory_space(stmt.output)
                self.memory_store(location, space, dtype, self.operand(name, dtype))
        elif isinstance(stmt, StridedLoop):
            self.strided_loop(stmt)
        elif isinstance(stmt, Accum):
            if stmt.op.name != "add":
                raise NotImplementedError("Only sum accumulators are supported initially")
            target = self.values[stmt.name]
            dtype = self.value_dtypes[stmt.name]
            base = self.operand(stmt.base, dtype) if stmt.base is not None else target
            result = self.arithmetic("add", [base, self.operand(stmt.value, dtype)], dtype)
            self.move(target, result, dtype)
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
            aliases = {"float": "f32", "double": "f64", "__half": "f16",
                       "int": "i32", "int32_t": "i32", "long long": "i64", "int64_t": "i64"}
            if stmt.dtype not in aliases:
                raise NotImplementedError(f"Unsupported shared storage: {stmt.dtype}")
            dtype = aliases[stmt.dtype]
            width = self.element_bytes(dtype)
            symbol = f"scratch_{len(self.shared)}"
            size = math.prod(stmt.extents) * width
            self.declarations.append(f"    .shared .align {stmt.align or width} .b8 {symbol}[{size}];")
            base = self.reg("rd")
            self.emit(f"mov.u64 {base}, {symbol};")
            self.shared[stmt.name] = base
            self.buffers[stmt.name] = {"dtype": dtype, "shape": list(stmt.extents)}
        elif isinstance(stmt, Sync):
            if stmt.warp:
                raise NotImplementedError("Warp barriers are not supported yet")
            self.sync(stmt.barrier_id, stmt.count)
        elif isinstance(stmt, WarpShuffle):
            if (stmt.length < 1 or stmt.length > 32
                    or stmt.length & (stmt.length - 1)):
                raise NotImplementedError("Shuffle requires a power-of-two width <= 32")
            dtype = stmt.dtype.name
            mask = self.reg("r")
            self.emit(f"activemask.b32 {mask};")
            distance = stmt.length // 2
            while distance:
                for state, other in zip(stmt.state, stmt.state_b, strict=True):
                    value = self.operand(state, dtype)
                    partner = self.data_reg(dtype)
                    if self.element_bytes(dtype) == 8:
                        low, high, low_b, high_b = (self.reg("r") for _ in range(4))
                        self.emit(f"mov.b64 {{{low}, {high}}}, {value};")
                        self.emit(f"shfl.sync.bfly.b32 {low_b}, {low}, {distance}, 31, {mask};")
                        self.emit(f"shfl.sync.bfly.b32 {high_b}, {high}, {distance}, 31, {mask};")
                        self.emit(f"mov.b64 {partner}, {{{low_b}, {high_b}}};")
                    else:
                        self.emit(f"shfl.sync.bfly.b32 {partner}, {value}, {distance}, 31, {mask};")
                    self.values[other] = partner
                    self.value_dtypes[other] = dtype
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
        types = {"r": "b32", "rd": "b64", "f": "f32", "h": "b16",
                 "d": "f64", "l": "b64", "p": "pred"}
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
