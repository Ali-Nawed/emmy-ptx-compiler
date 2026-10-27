#!/usr/bin/env python3
"""Independent, bounded NumPy interpreter for our current Emmy Kernel IR subset.

This is a test oracle, not a lowering pass or a CUDA emulator. It imports Emmy's
parser/classes but no compiler emitter, CUDA renderer or expression references.
Each NumPy vector represents the logical threads in one block. Typed operation
boundaries and the explicit reduction schedule determine the numerical result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var, TernaryExpr
from emmy.compiler.ir.kernel.ir import KernelOp, Tile, Smem, Sync, WarpShuffle, TreeHalve
from emmy.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, StridedLoop, Write, Let, Init
from tensor_reference import TensorReference

DTYPES = {name: np.dtype(dtype) for name, dtype in
          (("f16", "float16"), ("f32", "float32"), ("f64", "float64"),
           ("i32", "int32"), ("i64", "int64"), ("bool", "bool"))}
SHARED_TYPES = {"float": "f32", "double": "f64", "__half": "f16",
                "int": "i32", "int32_t": "i32", "long long": "i64", "int64_t": "i64"}


def cast(value, dtype):
    """NumPy implements finite narrowing, half rounding and integer wraparound.

    Undefined/out-of-range float-to-integer casts are deliberately rejected;
    this checker does not guess a backend's exceptional conversion behavior.
    """
    target = DTYPES[dtype]
    value = np.asarray(value)
    if target.kind == "i" and value.dtype.kind == "f":
        width = target.itemsize * 8
        # Range validation must not narrow the integer limits to f16/f32.
        # This widening is only the guard, not the numeric conversion itself.
        checked = value.astype(np.float64, copy=False)
        if np.any(~np.isfinite(checked) | (checked < -(2 ** (width - 1))) | (checked >= 2 ** (width - 1))):
            raise NotImplementedError("Out-of-range float-to-integer conversion")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        return value.astype(target)


def calculate(operation, operands, dtype):
    """Evaluate an operation with its declared operand/result precision."""
    comparisons = {"greater": np.greater, "greater_equal": np.greater_equal,
                   "less": np.less, "less_equal": np.less_equal,
                   "equal": np.equal, "not_equal": np.not_equal}
    if operation in comparisons:
        # Follow Emmy's C++ comparison promotion for mixed float/int inputs;
        # NumPy would instead promote i64+f32 to f64.
        kinds = {np.asarray(value).dtype.kind for value in operands}
        if "f" in kinds and "i" in kinds:
            working = np.float64 if any(np.asarray(value).dtype == np.float64 for value in operands) else np.float32
            operands = [np.asarray(value, dtype=working) for value in operands]
        return comparisons[operation](*operands).astype(np.bool_)
    if operation == "where":
        return cast(np.where(operands[0], cast(operands[1], dtype), cast(operands[2], dtype)), dtype)
    if dtype == "bool":
        logical = {"logical_and": np.logical_and, "logical_or": np.logical_or,
                   "logical_xor": np.logical_xor, "logical_not": np.logical_not,
                   "bitwise_and": np.logical_and, "bitwise_or": np.logical_or,
                   "bitwise_xor": np.logical_xor, "bitwise_not": np.logical_not,
                   "copy": lambda x: x.copy()}
        if operation not in logical:
            raise NotImplementedError(f"IR reference boolean operation: {operation}")
        return logical[operation](*operands).astype(np.bool_)
    args = [cast(value, dtype) for value in operands]
    binary = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
              "maximum": np.maximum, "minimum": np.minimum,
              "bitwise_and": np.bitwise_and, "right_shift": np.right_shift}
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        if operation in binary and len(args) == 2:
            result = binary[operation](*args)
        elif operation == "divide" and len(args) == 2:
            if dtype.startswith("i"):
                # Python integers avoid a float64 round trip for wide division.
                def quotient(a, b):
                    if b == 0 or (a == -(2 ** (DTYPES[dtype].itemsize * 8 - 1)) and b == -1):
                        raise NotImplementedError("Undefined integer division")
                    return (abs(int(a)) // abs(int(b))) * (-1 if (a < 0) != (b < 0) else 1)
                result = np.array([quotient(a, b) for a, b in zip(*args, strict=True)])
            else:
                result = np.divide(*args)
        elif operation in ("sqrt", "reciprocal") and not dtype.startswith("i") and len(args) == 1:
            # Match the backend's documented f16 working representation:
            # compute in f32, then narrow at the operation boundary.
            value = args[0].astype(np.float32) if dtype == "f16" else args[0]
            result = np.sqrt(value) if operation == "sqrt" else np.divide(np.ones_like(value), value)
        elif operation in ("negative", "abs", "absolute") and len(args) == 1:
            result = np.negative(args[0]) if operation == "negative" else np.abs(args[0])
        elif operation in ("sigmoid", "tanh") and dtype.startswith("f") and len(args) == 1:
            value = args[0].astype(np.float32) if dtype == "f16" else args[0]
            if operation == "tanh":
                result = np.tanh(value)
            else:
                t = np.exp(-np.abs(value))
                result = np.where(value >= 0, 1/(1+t), t/(1+t))
        elif operation in ("exp", "exp_fast", "exp2", "log", "log2", "log10", "rsqrt") and not dtype.startswith("i") and len(args) == 1:
            value = args[0].astype(np.float32) if dtype == "f16" else args[0]
            functions = {"exp": np.exp, "exp_fast": np.exp, "exp2": np.exp2,
                         "log": np.log, "log2": np.log2, "log10": np.log10,
                         "rsqrt": lambda x: 1 / np.sqrt(x)}
            result = functions[operation](value)
        elif operation in ("power", "pow", "float_power") and not dtype.startswith("i") and len(args) == 2:
            result = np.power(*(value.astype(np.float32) if dtype == "f16" else value for value in args))
        elif operation in ("power", "pow") and dtype.startswith("i") and len(args) == 2:
            if np.any(args[1] < 0):
                raise NotImplementedError("Negative integer exponents are outside integer power's domain")
            result = np.power(*args)
        elif operation == "square" and len(args) == 1:
            result = args[0] * args[0]
        elif operation == "relu" and len(args) == 1:
            result = np.where(args[0] < 0, np.zeros_like(args[0]), args[0])
        elif operation == "copy" and len(args) == 1:
            result = args[0].copy()
        else:
            raise NotImplementedError(f"IR reference operation: {operation}/{dtype}")
    return cast(result, dtype)


def accumulator_identity(identity, dtype):
    if DTYPES[dtype].kind == "i":
        limits = np.iinfo(DTYPES[dtype])
        identity = min(limits.max, max(limits.min, int(identity)))
    with np.errstate(over="ignore"):
        return np.asarray(identity).astype(DTYPES[dtype])


class Block(TensorReference):
    def __init__(self, count, buffers, initialized):
        self.count = count
        self.buffers = dict(buffers)
        self.initialized = dict(initialized)
        self.values = {}
        self.valid = {}
        self.carried = set()

    def bind(self, name, value, mask):
        value = np.broadcast_to(value, (self.count,))
        previous = self.values.get(name)
        if previous is not None and previous.dtype != value.dtype:
            raise NotImplementedError(f"A binding changes dtype within a scope: {name}")
        result = previous.copy() if previous is not None else np.zeros_like(value)
        result[mask] = value[mask]
        self.values[name] = result
        self.valid[name] = self.valid.get(name, np.zeros(self.count, bool)) | mask

    def read(self, name, mask):
        if name not in self.values or np.any(mask & ~self.valid[name]):
            raise ValueError(f"Read of uninitialized thread binding: {name}")
        return self.values[name]

    def evaluate(self, operation, operands, dtype, mask):
        # An inactive lane must not trigger an exceptional conversion/divide.
        result = np.zeros(self.count, DTYPES[dtype])
        result[mask] = calculate(operation, [value[mask] for value in operands], dtype)
        return result

    def expression(self, expr, mask):
        if isinstance(expr, Var):
            return self.read(expr.name, mask)
        if isinstance(expr, Literal):
            return np.full(self.count, expr.value, dtype=np.int64)
        if isinstance(expr, int):
            return np.full(self.count, expr, dtype=np.int64)
        if isinstance(expr, TernaryExpr):
            return np.where(self.expression(expr.cond, mask), self.expression(expr.if_true, mask),
                            self.expression(expr.if_false, mask))
        if isinstance(expr, BinaryExpr):
            if expr.op in ("/", "//", "%"):
                left, right = self.expression(expr.left, mask), self.expression(expr.right, mask)
                if np.any(right[mask] == 0):
                    raise ValueError("Zero divisor in IR index expression")
                # Integer indices use C++/PTX truncation toward zero, not
                # Python floor division when the operands have opposite signs.
                quotient = (np.abs(left)//np.maximum(np.abs(right), 1))*np.where((left < 0) ^ (right < 0), -1, 1)
                return left-quotient*right if expr.op == "%" else quotient
            operations = {"+": np.add, "-": np.subtract, "*": np.multiply,
                          "==": np.equal, "!=": np.not_equal, "<": np.less,
                          "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}
            if expr.op in operations:
                return operations[expr.op](self.expression(expr.left, mask), self.expression(expr.right, mask))
        raise NotImplementedError(f"IR reference index/condition: {expr!r}")

    def indices(self, name, index, mask):
        shape = self.buffers[name].shape
        if not index and self.buffers[name].size == 1:
            return np.zeros(self.count, np.int64)
        if len(index) != len(shape):
            raise ValueError("IR index rank differs from buffer rank")
        flat = np.zeros(self.count, np.int64)
        for coordinate, extent in zip(index, shape, strict=True):
            flat = flat * extent + self.expression(coordinate, mask)
        return flat

    def memory_read(self, name, index, mask):
        selected = index[mask]
        data = self.buffers[name].ravel()
        if np.any((selected < 0) | (selected >= len(data))):
            raise IndexError(f"Out-of-bounds IR load from {name}")
        if np.any(~self.initialized[name].ravel()[selected]):
            raise ValueError(f"Read of uninitialized memory: {name}")
        result = np.zeros(self.count, data.dtype)
        result[mask] = data[selected]
        return result

    def memory_write(self, name, index, values, mask):
        selected = index[mask]
        data = self.buffers[name].ravel()
        if np.any((selected < 0) | (selected >= len(data))):
            raise IndexError(f"Out-of-bounds IR store to {name}")
        converted = np.asarray(values[mask]).astype(data.dtype)
        unique, inverse = np.unique(selected, return_inverse=True)
        if len(unique) != len(selected):
            for slot in range(len(unique)):
                group = converted[inverse == slot]
                if not np.all(group == group[0]):
                    raise NotImplementedError("Conflicting concurrent IR stores")
        data[selected] = converted
        self.initialized[name].ravel()[selected] = True

    def scoped(self, body, mask):
        # Locals in a condition/loop body cannot leak into its enclosing scope.
        before, valid_before = self.values.copy(), self.valid.copy()
        outer_carried = self.carried.copy()
        self.statements(body, mask)
        carried = {name: (self.values[name], self.valid[name]) for name in outer_carried}
        self.carried = outer_carried
        self.values, self.valid = before, valid_before
        for name, (value, valid) in carried.items():
            self.values[name], self.valid[name] = value, valid

    def merge(self, stmt, mask):
        before, valid_before = self.values.copy(), self.valid.copy()
        for assignment in stmt.combine_states:
            args = [self.read(name, mask) for name in assignment.args]
            dtype = assignment.dtype.name if assignment.dtype is not None else stmt.dtype.name
            result = self.evaluate(assignment.op.name, args, dtype, mask)
            # The merge program may temporarily promote carried state. Keep
            # this local binding separate from the outer accumulator dtype.
            if assignment.name in self.values and self.values[assignment.name].dtype != result.dtype:
                self.values[assignment.name] = self.values[assignment.name].astype(result.dtype)
            self.bind(assignment.name, result, mask)
        state = {name: (self.values[name], self.valid[name]) for name in stmt.state}
        self.values, self.valid = before, valid_before
        for name, (value, valid) in state.items():
            self.bind(name, value.astype(before[name].dtype), mask)

    def statements(self, body, mask):
        for stmt in body:
            if np.any(mask):
                self.statement(stmt, mask)

    def statement(self, stmt, mask):
        if self.tensor_statement(stmt, mask):
            return
        if isinstance(stmt, Init):
            self.bind(stmt.name, accumulator_identity(stmt.identity, stmt.dtype.name), mask)
            # Init names mutable state that seed=False inner loops update.
            # Keep those updates across an enclosing staging-loop scope.
            self.carried.add(stmt.name)
        elif isinstance(stmt, Let):
            dtype = stmt.dtype.name if stmt.dtype is not None else "i32"
            if isinstance(stmt.value, Literal):
                self.bind(stmt.name, cast(np.asarray(stmt.value.value), dtype), mask)
            else:
                self.bind(stmt.name, cast(self.expression(stmt.value, mask), dtype), mask)
        elif isinstance(stmt, Load):
            indices = self.indices(stmt.input, stmt.index, mask)
            for offset, name in enumerate(stmt.names):
                value = self.memory_read(stmt.input, indices + offset, mask)
                self.bind(name, cast(value, stmt.dtype.name) if stmt.dtype is not None else value, mask)
        elif isinstance(stmt, Assign):
            args = [self.read(name, mask) for name in stmt.args]
            if stmt.dtype is None:
                # Explicitly limit unstamped statements rather than guessing promotion.
                raise NotImplementedError("Unstamped Assign outside a reduction merge")
            if stmt.op.name == "bitcast":
                target = DTYPES[stmt.dtype.name]
                if len(args) != 1 or target.itemsize != args[0].dtype.itemsize or target.itemsize not in (4, 8):
                    raise NotImplementedError("equal-width 32/64-bit bitcast required")
                self.bind(stmt.name, args[0].view(target), mask)
                return
            self.bind(stmt.name, self.evaluate(stmt.op.name, args, stmt.dtype.name, mask), mask)
        elif isinstance(stmt, Write):
            if stmt.atomic or stmt.swizzle != "NONE":
                raise NotImplementedError("Atomic/swizzled stores are outside this oracle")
            index = self.indices(stmt.output, stmt.index, mask)
            for offset, name in enumerate(stmt.values):
                self.memory_write(stmt.output, index + offset, self.read(name, mask), mask)
        elif isinstance(stmt, Cond):
            condition = self.expression(stmt.cond, mask).astype(bool)
            self.scoped(stmt.body, mask & condition)
            self.scoped(stmt.else_body, mask & ~condition)
        elif isinstance(stmt, (Loop, StridedLoop)):
            serial = isinstance(stmt, Loop)
            if serial and stmt.carries:
                raise NotImplementedError("Loop with carried array state is not supported")
            accumulators = {s.name: s for s in stmt.body if isinstance(s, Accum)}
            for name, accumulator in accumulators.items():
                if accumulator.op.name not in ("add", "multiply", "maximum", "minimum"):
                    raise NotImplementedError(f"Unsupported accumulator: {accumulator.op.name}")
                if stmt.seed:
                    dtype = accumulator.dtype.name if accumulator.dtype is not None else "f32"
                    self.bind(name, np.full(self.count, accumulator_identity(accumulator.op.identity, dtype), DTYPES[dtype]), mask)
                else:
                    self.read(name, mask)
            old_carried = self.carried.copy()
            self.carried.update(accumulators)
            coordinate = np.zeros(self.count, dtype=np.int64) if serial else self.expression(stmt.start, mask).copy()
            step = np.ones(self.count, dtype=np.int64) if serial else self.expression(stmt.step, mask)
            end = (self.expression(stmt.end, mask) if not serial and stmt.end is not None
                   else stmt.axis.extent.as_static())
            if np.any(step[mask] <= 0):
                raise NotImplementedError("Nonpositive loop step")
            old_axis = (self.values.get(stmt.axis.name), self.valid.get(stmt.axis.name))
            iterations = 0
            while np.any(active := mask & (coordinate < end)):
                self.bind(stmt.axis.name, coordinate, active)
                self.scoped(stmt.body, active)
                coordinate[active] += step[active]
                iterations += 1
                if iterations > 100000:
                    raise ValueError("IR reference loop iteration limit")
            self.carried = old_carried
            if old_axis[0] is None:
                self.values.pop(stmt.axis.name, None)
                self.valid.pop(stmt.axis.name, None)
            else:
                self.values[stmt.axis.name], self.valid[stmt.axis.name] = old_axis
        elif isinstance(stmt, Accum):
            dtype = stmt.dtype.name if stmt.dtype is not None else "f32"
            left = self.read(stmt.base or stmt.name, mask)
            self.bind(stmt.name, self.evaluate(stmt.op.name, [left, self.read(stmt.value, mask)], dtype, mask), mask)
        elif isinstance(stmt, Smem):
            if stmt.dtype not in SHARED_TYPES or not np.all(mask):
                raise NotImplementedError("Unsupported or conditional shared allocation")
            self.buffers[stmt.name] = np.zeros(stmt.extents, DTYPES[SHARED_TYPES[stmt.dtype]])
            self.initialized[stmt.name] = np.zeros(stmt.extents, bool)
        elif isinstance(stmt, Sync):
            if stmt.warp or stmt.barrier_id or stmt.count is not None or not np.all(mask):
                raise NotImplementedError("Only converged whole-block barriers are modeled")
            # Whole-array statement execution already orders this block's stores.
        elif isinstance(stmt, WarpShuffle):
            if stmt.length < 1 or stmt.length > 32 or stmt.length & (stmt.length - 1):
                raise NotImplementedError("Unsupported warp width")
            distance = stmt.length // 2
            while distance:
                partners = np.arange(self.count) ^ distance
                if np.any(partners[mask] >= self.count) or np.any(~mask[partners[mask]]):
                    raise NotImplementedError("Shuffle source lane is inactive")
                for name, other in zip(stmt.state, stmt.state_b, strict=True):
                    values = self.read(name, mask)
                    self.bind(other, cast(values[partners], stmt.dtype.name), mask)
                self.merge(stmt, mask)
                distance //= 2
        elif isinstance(stmt, TreeHalve):
            if (stmt.inner is not None or stmt.barrier_id or stmt.barrier_count is not None
                    or stmt.length < 1 or stmt.length & (stmt.length - 1) or not np.all(mask)):
                raise NotImplementedError("Unsupported shared-memory tree")
            slots = self.read(stmt.tid_var, mask)
            outer, valid_outer = self.values.copy(), self.valid.copy()
            half = stmt.length // 2
            while half:
                active = mask & (slots < half)
                for buf, state, other in zip(stmt.bufs, stmt.state, stmt.state_b, strict=True):
                    # Tree state shadows the outer per-thread partials.
                    for name, index in ((state, slots), (other, slots + half)):
                        self.values.pop(name, None)
                        self.valid.pop(name, None)
                        self.bind(name, cast(self.memory_read(buf, index, active), stmt.dtype.name), active)
                self.merge(stmt, active)
                for buf, state in zip(stmt.bufs, stmt.state, strict=True):
                    self.memory_write(buf, slots, self.read(state, active), active)
                half //= 2
            self.values, self.valid = outer, valid_outer
            for buf, state in zip(stmt.bufs, stmt.state, strict=True):
                value = self.memory_read(buf, np.zeros(self.count, np.int64), mask)
                self.bind(state, value.astype(self.values[state].dtype), mask)
        else:
            raise NotImplementedError(f"IR reference statement: {type(stmt).__name__}")


def interpret_kernel(kernel, descriptor, buffers, initialized):
    if kernel.serial or len(kernel.body) != 1 or not isinstance(kernel.body[0], Tile):
        raise NotImplementedError("Expected one nonserial Tile")
    tile = kernel.body[0]
    if (tile.aux_threads or tile.raster_group is not None or descriptor["grid"][1:] != [1, 1]
            or descriptor["block"][1:] != [1, 1]):
        raise NotImplementedError("Only ordinary one-dimensional launches are modeled")
    count = descriptor["block"][0]
    if tile.block_threads is not None and (tile.block_threads != count or tile.n_elements % count):
        raise NotImplementedError("Cooperative tile must contain complete blocks")
    if descriptor["grid"][0] * count < tile.n_elements:
        raise ValueError("Launch does not cover the tile")
    for block_id in range(descriptor["grid"][0]):
        block = Block(count, buffers, initialized)
        gids = np.arange(count, dtype=np.int64) + block_id * count
        active = gids < tile.n_elements
        coordinates = np.unravel_index(gids[active], tile.extents) if tile.axes else ()
        for axis, coordinate in zip(tile.axes, coordinates, strict=True):
            values = np.zeros(count, np.int64)
            values[active] = coordinate
            block.bind(axis.name, values, active)
        if tile.block_threads is not None:
            block.bind("lane", np.arange(count, dtype=np.int64) % 32, active)
            block.bind("warp", np.arange(count, dtype=np.int64) // 32, active)
        block.statements(tile.body, active)


def simulate(folder):
    """Return graph outputs using only metadata inputs and parsed Kernel IR."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    buffers, initialized = {}, {}
    for descriptor in meta["kernels"]:
        for arg in descriptor["args"]:
            name, dtype, shape = arg["name"], DTYPES[arg["dtype"]], arg["shape"]
            if arg.get("scalar"):
                raise NotImplementedError("By-value scalar arguments are outside this oracle")
            if arg.get("intermediate"):
                if name not in buffers or np.any(~initialized[name]):
                    raise ValueError(f"Uninitialized intermediate: {name}")
            elif arg.get("output"):
                buffers[name] = np.zeros(shape, dtype)
                initialized[name] = np.zeros(shape, bool)
            elif "data" in arg:
                buffers[name] = np.asarray(arg["data"], dtype=dtype).reshape(shape)
                initialized[name] = np.ones(shape, bool)
            elif "seed" in arg:
                buffers[name] = np.random.default_rng(arg["seed"]).standard_normal(shape).astype(dtype)
                initialized[name] = np.ones(shape, bool)
            else:
                raise NotImplementedError(f"Unknown input initializer: {name}")
        for name in descriptor["zero_outputs"]:
            buffers[name].fill(0)
            initialized[name].fill(True)
        graph = Graph.from_dict(json.loads((folder / f"{descriptor['entry']}.ir.json").read_text()))
        kernels = [node.op for node in graph.nodes.values() if isinstance(node.op, KernelOp)]
        if len(kernels) != 1 or kernels[0].name != descriptor["entry"]:
            raise ValueError("Kernel name/count disagrees with metadata")
        interpret_kernel(kernels[0], descriptor, buffers, initialized)
    outputs = meta.get("graph_outputs")
    if outputs is None:
        with np.load(folder / "expected.npz") as saved:
            outputs = saved.files
    for name in outputs:
        if np.any(~initialized[name]):
            raise ValueError(f"Graph output was not completely written: {name}")
    return {name: buffers[name].copy() for name in outputs}


def save_reference(folder):
    folder = Path(folder)
    outputs = simulate(folder)
    np.savez(folder / "expected_ir.npz", **outputs)
    sources = [folder / "meta.json", *sorted(folder.glob("*.ir.json"))]
    sources.append(Path(__file__))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    provenance = {"oracle": "independent NumPy typed Kernel IR interpreter",
                  "numpy": np.__version__, "emmy": importlib.metadata.version("emmy-ml"),
                  "source_sha256": hashes}
    (folder / "expected_ir.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args()
    for folder in args.folders:
        outputs = save_reference(folder)
        print(json.dumps({"case": folder.name, "outputs": {name: {"shape": list(value.shape),
                           "dtype": str(value.dtype)} for name, value in outputs.items()}}), flush=True)


if __name__ == "__main__":
    main()
