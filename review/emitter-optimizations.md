# Tile decoding and grouped stores

These changes live in compile.py's existing emitter. There is no new IR,
optimization-pass framework, liveness analysis or physical register allocator.
The emitter continues to hand out fresh virtual registers; ptxas allocates them.
Floating operations, FMA policy, loop order and reduction schedules are unchanged.

## Tile coordinates

tile_coordinates decodes a worker already known to satisfy 0 <= worker < N.
The existing bounds guard provides that proof for ordinary Tiles; cooperative
Tiles already require complete blocks and exact metadata coverage.

For extents (17, 128), the previous generic code calculated
(worker / 128) % 17 and (worker / 1) % 128. The emitter now produces:

```ptx
shr.u32 %row, %worker, 7;
and.b32 %column_worker, %worker, 127;
```

unsigned_divide removes divide-by-one and uses shifts for powers of two.
unsigned_remainder handles extent one and uses masks for powers of two.
Other divisors keep the generic unsigned instructions. The outer quotient needs
no modulo because the worker is in range. The scalar/empty-axis Tile remains
valid. Unit checks compare emitted instruction semantics with NumPy row-major
coordinates for zero through three axes, including non-power-of-two extents.

## Grouped stores

vector_store preserves a Write's ordered tuple when the access is proved aligned:

- Supported widths are two and four elements, with total width at most 16 bytes.
  Four f64/i64 elements therefore keep scalar stores; two can use a vector.
- The base must have the required alignment. Global arguments use 16 bytes of
  the runner's fresh CUDA-allocation guarantee; its argument schema has no offset
  views. Shared buffers use their declared alignment or natural element alignment.
- A small recursive divisor calculation proves that the flattened row-major
  element offset is a multiple of the vector width. It accounts for all row
  strides. An odd row stride, unknown offset or insufficient base alignment
  retains scalar stores. This is a local index proof, not liveness analysis.
- f16 values are rounded into b16 registers exactly as in the old scalar stores,
  then written together. Other types use their native store element type.

For example, a four-value f32 Write at a multiple-of-four index emits
st.global.v4.f32 instead of four scalar stores. The order and data conversions
are unchanged. Loads, atomics and swizzled writes are outside this change.

The base-alignment assumption is specific to the runner's freshly allocated
buffers. Supporting arbitrary offset views in another host application would
require supplying their alignment or choosing the scalar fallback.
[NVIDIA documents CUDA allocation alignment](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#effects-of-misaligned-accesses).

## Local assembly evidence

The old review/emmy-ptx-comparison artifacts are preserved. The new audit uses
the same sm_89 target and CUDA 12.0.140 ptxas -O3. No nvcc invocation is involved
in the audit. All 19 new kernels assemble without spills.

| Kernel | Non-NOP static SASS before -> after | Registers before -> after |
| --- | --- | --- |
| p01 vector add | 25 -> 24 | 18 -> 16 |
| p02 scale/shift | 41 -> 38 | 23 -> 24 |
| p03 ReLU/multiply | 47 -> 44 | 22 -> 22 |
| r04 row sum | 152 -> 105 | 17 -> 16 |
| row sum 17x1025 | 152 -> 105 | 17 -> 16 |

Both row kernels lose all three MUFU.RCP instructions used by the old index
division sequences. Pointwise grouped stores become STG.E.64 or STG.E.128.
The scale/shift register increase is a real tradeoff: fewer stores can keep
more results live until one vector store. No manual register reuse was added.

Counts include static control-flow/slow-path instructions and exclude NOPs;
they are not executed instruction counts or proof of a runtime speedup.
The audit report and before/after opcodes are in
review/emitter-optimizations/assembly-results.json.

## Validation and reproduction

21 local tests pass. All 480 selected GPU regression cases match the independent
IR checker, with zero unexpected CPU/GPU prediction differences. Of these, 453
match the original expression; the same 27 known precision/order failures remain.
No expectations or tolerances were changed to obtain these results.

| Group | Cases | IR matches | Expression matches |
| --- | ---: | ---: | ---: |
| Public MVP | 8 | 8 | 8 |
| Original extended matrix | 80 | 80 | 77 |
| Boundary/schedule matrix | 317 | 317 | 305 |
| Explicit reduction inputs | 40 | 40 | 28 |
| f16 edge cases | 8 | 8 | 8 |
| i64 output-bit checks | 8 | 8 | 8 |
| Larger f16 cases | 4 | 4 | 4 |
| New vector-store add cases | 10 | 10 | 10 |
| New wide-store scale/shift cases | 5 | 5 | 5 |
| Total | 480 | 480 | 453 |

The main run is recorded in validation.json (475 cases); the five additional
wide-store checks are in wide-vectors.json. A fresh default invocation includes
all 480 cases in one report. summary.json combines the saved runs.

Focused local tests check coordinate results, alignment failure on odd row
strides and nonzero offsets, insufficient base alignment, and assembly for all
five dtypes at widths two/four. Genuine new PyTorch -> Emmy fixtures additionally
cover add at sizes 128/256 and scale/shift at size 4096 for all five datatypes.

The regression tool writes new PTX and reports under review/emitter-optimizations;
it leaves the old fixture expectations, reports and baseline PTX untouched.
It computes the CPU IR expectation independently, assembles our PTX locally,
then optionally checks the hosted results against both expectation sets.
Known expression precision/order failures remain failures.

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tools/test_extended.py --reuse --ir-reference --ops add --sizes 128 256 --out tests/vector-store-cases
python tools/test_extended.py --reuse --ir-reference --ops scale_shift --sizes 4096 --out tests/vector-store-wide
python tools/validate_emitter_optimizations.py --runner
python tools/audit_emitter_optimizations.py
python tools/benchmark_emitter_optimizations.py
```

The benchmark uploads only our saved before PTX and our new after PTX. Reference
Emmy/nvcc PTX is never uploaded. Three alternating paired rounds use 10 warmup
launches and 100 timed iterations; every output is checked. GPU clocks are not
locked, so these are indicative samples, not controlled performance guarantees.
An unchanged scalar sum is included to expose run-to-run variation.

## Hosted timing samples

The median of each variant's three returned medians was:

| Kernel | Before, microseconds | After, microseconds |
| --- | ---: | ---: |
| p01 vector add | 2.944 | 2.976 |
| p02 scale/shift | 3.072 | 3.040 |
| p03 ReLU/multiply | 2.976 | 2.976 |
| r04 row sum | 2.976 | 2.976 |
| row sum 17x1025 | 3.040 | 2.976 |
| r01 sum, unchanged control | 2.976 | 2.976 |

Observed differences range from about 2.1% faster to 1.1% slower. With these small
kernel times, few samples and unlocked clocks, this does not establish a material
runtime speedup. The instruction/store reductions are confirmed by assembly.
Raw paired samples, PTX hashes and settings are retained in timing-results.json.
Every measured output passed both expression and IR checks.
