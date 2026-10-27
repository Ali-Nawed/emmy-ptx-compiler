"""Independent NumPy model of the supported warp MMA fragment semantics.

Fragments are arrays of numeric elements, never PTX registers or packed bits.
This module imports no compiler emitter and invokes no CUDA implementation.
"""

import numpy as np

from emmy.compiler.ir.kernel.ir import RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore


class TensorReference:
    def fragment_coordinates(self, role):
        lane = np.arange(self.count) % 32
        group, thread = lane // 4, lane % 4
        count = {"a": 8, "b": 4, "c": 4}[role]
        element = np.arange(count)[None, :]
        if role == "a":
            rows = group[:, None] + ((element // 2) % 2)*8
            cols = thread[:, None]*2 + (element // 4)*8 + element % 2
        elif role == "b":
            rows = thread[:, None]*2 + (element // 2)*8 + element % 2
            cols = np.broadcast_to(group[:, None], rows.shape)
        else:
            rows = group[:, None] + (element // 2)*8
            cols = thread[:, None]*2 + element % 2
        return rows, cols

    def tensor_statement(self, stmt, mask):
        if isinstance(stmt, RegFragment):
            if tuple(stmt.shape) != (16, 8, 16) or stmt.count != 1:
                raise NotImplementedError("Reference supports single m16n8k16 fragments")
            if not hasattr(self, "tensor_values"):
                self.tensor_values = {}
            count = {"a": 8, "b": 4, "c": 4}[stmt.role]
            self.tensor_values[stmt.name] = {"role": stmt.role,
                "data": np.zeros((self.count, count), np.float32 if stmt.role == "c" else np.float16)}
        elif isinstance(stmt, LdmatrixLoad):
            if stmt.staged or stmt.swizzle != "NONE" or stmt.pair_frag or stmt.fragment_layout != "m16n8k16":
                raise NotImplementedError("Reference supports global m16n8k16 fragment loads")
            rows, cols = self.fragment_coordinates(stmt.role)
            rows, cols = rows.copy(), cols.copy()
            k = cols.copy() if stmt.role == "a" else rows.copy()
            if stmt.gmem_guard is not None:
                remaining = self.expression(stmt.gmem_guard[1], mask)-self.expression(stmt.gmem_guard[0], mask)
                clipped = np.maximum(remaining-1, 0)[:, None]
                if stmt.role == "a":
                    rows = np.minimum(rows, clipped)
                else:
                    cols = np.minimum(cols, clipped)
            if stmt.role == "b" and stmt.b_trans:
                rows, cols = cols, rows
            base = self.indices(stmt.src_buffer, stmt.src_index, mask)
            stride = stmt.ldm or self.buffers[stmt.src_buffer].shape[-1]
            valid = np.broadcast_to(mask[:, None], rows.shape).copy()
            if stmt.k_zero is not None:
                remaining = self.expression(stmt.k_zero[1], mask)-self.expression(stmt.k_zero[0], mask)
                valid &= k < remaining[:, None]
            data = self.tensor_values[stmt.frag]["data"]
            for element in range(data.shape[1]):
                indices = base + rows[:, element]*stride + cols[:, element]
                data[mask, element] = self.memory_read(stmt.src_buffer, indices, valid[:, element])[mask]
        elif isinstance(stmt, MmaSyncPtx):
            if tuple(stmt.shape) != (16, 8, 16) or stmt.ab_dtype != "f16" or stmt.c_dtype != "f32":
                raise NotImplementedError("Reference supports f16/f32 m16n8k16 MMA")
            if stmt.b_row_major or stmt.sfa_frag or stmt.sfb_frag:
                raise NotImplementedError("Unsupported MMA operand layout/scales")
            coordinates = {role: self.fragment_coordinates(role) for role in ("a", "b", "c")}
            for start in range(0, self.count, 32):
                active = mask[start:start+32]
                if not np.any(active):
                    continue
                if len(active) != 32 or not np.all(active):
                    raise ValueError("MMA requires a complete active warp")
                matrices = []
                for role, name, shape in (("a",stmt.a_frag,(16,16)), ("b",stmt.b_frag,(16,8)), ("c",stmt.c_frag,(16,8))):
                    matrix = np.empty(shape, np.float64)
                    rows, cols = coordinates[role]
                    matrix[rows[start:start+32], cols[start:start+32]] = self.tensor_values[name]["data"][start:start+32]
                    matrices.append(matrix)
                # Numerical oracle, not a promise of the hardware's internal
                # summation order: round each completed MMA to f32.
                result = (matrices[0] @ matrices[1] + matrices[2]).astype(np.float32)
                rows, cols = coordinates["c"]
                self.tensor_values[stmt.c_frag]["data"][start:start+32] = result[rows[start:start+32], cols[start:start+32]]
        elif isinstance(stmt, RegStore):
            if stmt.epilogue or stmt.atomic or stmt.extra_frags or stmt.swizzle != "NONE" or stmt.fragment_layout != "m16n8k16":
                raise NotImplementedError("Reference supports plain m16n8k16 stores")
            rows, cols = self.fragment_coordinates("c")
            base = self.indices(stmt.dst_buffer, stmt.dst_index, mask)
            shape = self.buffers[stmt.dst_buffer].shape
            if stmt.row_dim not in (None, len(shape)-2) or stmt.col_dim not in (None, len(shape)-1):
                raise NotImplementedError("Reference requires trailing row/column dimensions")
            stride = stmt.ldm or shape[-1]
            data = self.tensor_values[stmt.frag]["data"]
            for element in range(4):
                valid = mask.copy()
                for guard, offset in ((stmt.m_guard,rows[:,element]), (stmt.n_guard,cols[:,element])):
                    if guard is not None:
                        valid &= self.expression(guard[0],mask)+offset < self.expression(guard[1],mask)
                self.memory_write(stmt.dst_buffer, base+rows[:,element]*stride+cols[:,element], data[:,element], valid)
        else:
            return False
        return True
