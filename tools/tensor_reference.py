"""Independent NumPy model of the supported warp MMA fragment semantics.

Fragments are arrays of numeric elements, never PTX registers or packed bits.
This module imports no compiler emitter and invokes no CUDA implementation.
"""

import numpy as np

from emmy.compiler.ir.kernel.ir import RegFragment, LdmatrixLoad, MmaSyncPtx, RegStore
from emmy.compiler.ir.kernel.ir import CpAsyncCopy, CpAsyncCommit, CpAsyncWait


class TensorReference:
    @staticmethod
    def swizzle_elements(flat, mode):
        name, sep, override = mode.partition("@")
        if name == "NONE" and not sep:
            return flat
        masks = {"B32": 1, "B64": 3, "B128": 7}
        if name not in masks or (sep and not override.isdecimal()):
            raise NotImplementedError(f"Reference swizzle: {mode}")
        shift = int(override) if sep else 6
        if not 3 <= shift < 32:
            raise NotImplementedError("Reference swizzle shift")
        return flat ^ (((flat >> shift) & masks[name])*8)

    def async_statement(self, stmt, mask):
        if not isinstance(stmt, (CpAsyncCopy, CpAsyncCommit, CpAsyncWait)):
            return False
        if not hasattr(self, "async_open"):
            self.async_open = [[] for _ in range(self.count)]
            self.async_groups = [[] for _ in range(self.count)]
        if isinstance(stmt, CpAsyncCopy):
            if stmt.nbytes not in (4, 8, 16):
                raise NotImplementedError("Reference async copy size")
            dtype = self.buffers[stmt.src].dtype
            if self.buffers[stmt.smem].dtype != dtype or stmt.nbytes % dtype.itemsize:
                raise NotImplementedError("Reference async copies require matching element types")
            source = self.indices(stmt.src, stmt.src_index, mask)
            destination = self.indices(stmt.smem, stmt.smem_index, mask)
            if stmt.lane_index is not None:
                destination += self.expression(stmt.lane_index[0],mask)*self.buffers[stmt.smem].shape[-1]
                destination += self.expression(stmt.lane_index[1],mask)
            destination = self.swizzle_elements(destination,stmt.swizzle)
            if np.any((source[mask]*dtype.itemsize) % stmt.nbytes) or np.any((destination[mask]*dtype.itemsize) % stmt.nbytes):
                raise ValueError("Misaligned async copy in generated IR")
            elements = stmt.nbytes//dtype.itemsize
            values = np.stack([self.memory_read(stmt.src, source+offset, mask) for offset in range(elements)],axis=1)
            for lane in np.flatnonzero(mask):
                self.async_open[lane].append((stmt.smem, int(destination[lane]), values[lane].copy()))
        elif isinstance(stmt, CpAsyncCommit):
            for lane in np.flatnonzero(mask):
                self.async_groups[lane].append(self.async_open[lane])
                self.async_open[lane] = []
                if len(self.async_groups[lane]) > 8:
                    raise ValueError("Too many outstanding cp.async groups")
        else:
            if not isinstance(stmt.group, int) or not 0 <= stmt.group <= 7:
                raise ValueError("Invalid cp.async wait group")
            for lane in np.flatnonzero(mask):
                groups = self.async_groups[lane]
                while len(groups) > stmt.group:
                    for name, start, data in groups.pop(0):
                        indices = start+np.arange(len(data))
                        if np.any((indices < 0) | (indices >= self.buffers[name].size)):
                            raise IndexError("Async copy outside shared allocation")
                        self.buffers[name].ravel()[indices] = data
                        self.initialized[name].ravel()[indices] = True
        return True

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
        if self.async_statement(stmt, mask):
            return True
        if isinstance(stmt, RegFragment):
            if tuple(stmt.shape) != (16, 8, 16) or stmt.count != 1:
                raise NotImplementedError("Reference supports single m16n8k16 fragments")
            if not hasattr(self, "tensor_values"):
                self.tensor_values = {}
            count = {"a": 8, "b": 4, "c": 4}[stmt.role]
            self.tensor_values[stmt.name] = {"role": stmt.role,
                "data": np.zeros((self.count, count), np.float32 if stmt.role == "c" else np.float16)}
        elif isinstance(stmt, LdmatrixLoad):
            if stmt.pair_frag or stmt.fragment_layout != "m16n8k16" or stmt.byte_slab or stmt.scale_buffer:
                raise NotImplementedError("Reference supports single f16 m16n8k16 fragment loads")
            if not stmt.staged and stmt.swizzle != "NONE":
                raise NotImplementedError("Reference global fragment load cannot be swizzled")
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
                if stmt.staged:
                    indices = self.swizzle_elements(indices,stmt.swizzle)
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
