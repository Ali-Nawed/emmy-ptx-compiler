"""Our direct PTX emitter for the m16n8k16 f16/f32 tensor-core subset."""

import math


class TensorMath:
    def tensor_i32(self, opcode, a, b):
        result = self.reg("r")
        self.emit(f"{opcode} {result}, {a}, {b};")
        return result

    def tensor_address(self, base, row, column, stride, dtype):
        offset = self.tensor_i32("mul.lo.s32", row, stride)
        offset = self.tensor_i32("add.s32", offset, column)
        wide, address = self.reg("rd"), self.reg("rd")
        self.emit(f"mul.wide.s32 {wide}, {offset}, {self.element_bytes(dtype)};")
        self.emit(f"add.u64 {address}, {base}, {wide};")
        return address

    def reg_fragment(self, stmt):
        counts = {"a": 4, "b": 2, "c": 4}
        expected = "f32" if stmt.role == "c" else "f16"
        if (tuple(stmt.shape) != (16, 8, 16) or stmt.role not in counts or stmt.count != 1
                or stmt.dtype.name != expected or stmt.nregs not in (None, counts[stmt.role])):
            raise NotImplementedError("Only single m16n8k16 f16 A/B and f32 C fragments are supported")
        if "lane" not in self.values or self.source.metadata["block"][0] % 32:
            raise NotImplementedError("Tensor fragments require complete cooperative warps")
        registers = tuple(self.reg("f" if stmt.role == "c" else "r") for _ in range(counts[stmt.role]))
        self.fragments[stmt.name] = {"role": stmt.role, "dtype": stmt.dtype.name, "registers": registers}
        if stmt.role == "c":
            for register in registers:
                self.emit(f"mov.f32 {register}, 0f00000000;")

    def matrix_load(self, stmt):
        if (stmt.staged or stmt.swizzle != "NONE" or stmt.pair_frag is not None or stmt.byte_slab
                or stmt.scale_buffer is not None or stmt.fragment_layout != "m16n8k16"):
            raise NotImplementedError("Only unstaged unswizzled f16 matrix fragment loads are supported")
        fragment = self.fragments[stmt.frag]
        if fragment["role"] != stmt.role or fragment["dtype"] != "f16" or stmt.role not in ("a", "b"):
            raise ValueError("Matrix load disagrees with fragment declaration")
        if self.buffers[stmt.src_buffer]["dtype"] != "f16":
            raise NotImplementedError("Matrix fragment loads currently require f16 source buffers")
        base = self.address(stmt.src_buffer, stmt.src_index)
        stride = stmt.ldm or self.buffers[stmt.src_buffer]["shape"][-1]
        group = self.tensor_i32("shr.u32", self.values["lane"], 2)
        thread = self.tensor_i32("and.b32", self.values["lane"], 3)
        pair = self.tensor_i32("mul.lo.s32", thread, 2)
        left = None
        if stmt.gmem_guard is not None:
            left = self.tensor_i32("sub.s32", self.index_expr(stmt.gmem_guard[1]), self.index_expr(stmt.gmem_guard[0]))
            left = self.tensor_i32("sub.s32", left, 1)
            left = self.tensor_i32("max.s32", left, 0)
        k_left = None
        if stmt.k_zero is not None:
            k_left = self.tensor_i32("sub.s32", self.index_expr(stmt.k_zero[1]), self.index_expr(stmt.k_zero[0]))
        for index, register in enumerate(fragment["registers"]):
            halves = []
            for half_index in (0, 1):
                if stmt.role == "a":
                    row = self.tensor_i32("add.s32", group, (index % 2)*8)
                    col = self.tensor_i32("add.s32", pair, (index // 2)*8+half_index)
                    k = col
                    if left is not None:
                        row = self.tensor_i32("min.s32", row, left)
                else:
                    row = self.tensor_i32("add.s32", pair, index*8+half_index)
                    col, k = group, row
                    if left is not None:
                        col = self.tensor_i32("min.s32", col, left)
                    if stmt.b_trans:
                        row, col = col, row
                address = self.tensor_address(base, row, col, stride, "f16")
                half = self.reg("h")
                if k_left is None:
                    self.emit(f"ld.global.b16 {half}, [{address}];")
                else:
                    valid = self.reg("p")
                    self.emit(f"mov.b16 {half}, 0;")
                    self.emit(f"setp.lt.s32 {valid}, {k}, {k_left};")
                    self.emit(f"@{valid} ld.global.b16 {half}, [{address}];")
                halves.append(half)
            self.emit(f"mov.b32 {register}, {{{halves[0]}, {halves[1]}}};")

    def mma_sync(self, stmt):
        if (tuple(stmt.shape) != (16, 8, 16) or stmt.ab_dtype != "f16" or stmt.c_dtype != "f32"
                or stmt.b_row_major or stmt.sfa_frag is not None or stmt.sfb_frag is not None):
            raise NotImplementedError("Only mma.sync m16n8k16 row.col f32.f16.f16.f32 is supported")
        fragments = [self.fragments[name] for name in (stmt.c_frag, stmt.a_frag, stmt.b_frag)]
        if [fragment["role"] for fragment in fragments] != ["c", "a", "b"]:
            raise ValueError("MMA fragment roles disagree")
        c, a, b = ("{" + ", ".join(fragment["registers"]) + "}" for fragment in fragments)
        self.emit(f"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {c}, {a}, {b}, {c};")

    def reg_store(self, stmt):
        if (tuple(stmt.shape) != (16, 8, 16) or stmt.fragment_layout != "m16n8k16" or stmt.atomic
                or stmt.epilogue is not None or stmt.extra_frags or stmt.swizzle != "NONE" or stmt.volta_interleaved):
            raise NotImplementedError("Only plain m16n8k16 fragment stores without epilogues are supported")
        fragment = self.fragments[stmt.frag]
        if fragment["role"] != "c" or fragment["dtype"] != "f32":
            raise ValueError("Fragment store requires an f32 accumulator")
        buffer = self.buffers[stmt.dst_buffer]
        rank = len(buffer["shape"])
        row_dim = rank-2 if stmt.row_dim is None else stmt.row_dim
        col_dim = rank-1 if stmt.col_dim is None else stmt.col_dim
        if row_dim != rank-2 or col_dim != rank-1 or rank < 2:
            raise NotImplementedError("Fragment stores require trailing row/column dimensions")
        dtype = buffer["dtype"]
        base = self.address(stmt.dst_buffer, stmt.dst_index)
        stride = stmt.ldm or math.prod(buffer["shape"][row_dim+1:])
        group = self.tensor_i32("shr.u32", self.values["lane"], 2)
        thread = self.tensor_i32("and.b32", self.values["lane"], 3)
        pair = self.tensor_i32("mul.lo.s32", thread, 2)
        for index, value in enumerate(fragment["registers"]):
            row = self.tensor_i32("add.s32", group, (index // 2)*8)
            col = self.tensor_i32("add.s32", pair, index % 2)
            done = self.label("FRAGMENT_STORE_DONE")
            for guard, offset in ((stmt.m_guard, row), (stmt.n_guard, col)):
                if guard is not None:
                    coordinate = self.tensor_i32("add.s32", self.index_expr(guard[0]), offset)
                    outside = self.reg("p")
                    self.emit(f"setp.ge.s32 {outside}, {coordinate}, {self.index_expr(guard[1])};")
                    self.emit(f"@{outside} bra {done};")
            address = self.tensor_address(base, row, col, stride, dtype)
            self.memory_store(f"[{address}]", self.memory_space(stmt.dst_buffer), dtype, self.convert(value, "f32", dtype))
            self.lines.append(f"{done}:")
