# Extended tests generated from PyTorch

## Workflow

1. Choose a PyTorch operation, input shape, dtype and optional reduction scheduling knobs.
2. Verify the independent NumPy reference against eager CPU PyTorch on small typed inputs.
3. Run pinned emmy-ml==0.3.11 for target sm_89, using the organizer's tools/generate.py functions.
4. Keep the fresh per-kernel IR and Emmy-derived launch metadata. Never patch IR text to change dimensions or dtype.
5. Package expected.npz using the organizer's seeded input construction and our NumPy reference.
6. Compile through our compile.py; assemble locally. Optionally upload only our compiler-generated PTX for numeric validation.

The PyTorch random constructor establishes the traced input's shape/dtype. Actual test values follow meta.json's NumPy seeds, as on the competition runner; they are not the random samples that happened to be created during tracing.

## Files

- tools/expressions_extended.py: matrix construction, PyTorch source and NumPy references.
- tools/test_extended.py: generation, reference self-checks, local assembly, optional runner checks and results.
- tests/extended/<case>/: expression.py, generated .ir.json, .ir.txt, meta.json, expected.npz, provenance and local reference CUDA. Our PTX has an .ours.ptx suffix.
- tests/extended/results.json: status of every attempted case, including failed generation and unsupported compilation. Saved after each case.
- tests/extended/matrix.json: selected cases, source expressions, tolerances and scheduling settings, without Python callables.

Reference CUDA is only a local fixture artifact produced by Emmy's normal generator. It is never compiled or uploaded by this test harness. No generation logic is used by the submission compiler.

## Matrix

Runner-supported dtypes: f16, f32, f64, i32, i64. The compiler implements all five, including conversions and typed reduction state. Emmy 0.3.11 generates f32 state for accumulating reductions even with f64/integer inputs; this limits their accuracy. BF16 and narrower integer types are outside the runner's documented argument schema.

Operations: addition, subtraction, multiplication, scale/shift, ReLU/multiply, sum, mean and row sum. Integer sum uses PyTorch's default i64 result; integer mean explicitly requests f32. Constants and reference intermediates preserve the appropriate dtype. Integer output comparisons require exact equality, without an f32 conversion.

Default: five dtypes x eight operations x two sizes (1 and 129), with three rows for row-sum tests: 80 attempted cases. Full boundary sizes: 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4096 and 65536. Additional row counts and cooperative work widths are configurable.

## Ubuntu commands, from the repo root

```bash
source .venv/bin/activate

# Start small; generate fixtures and compile/assemble locally.
python tools/test_extended.py --dtypes f32 --ops add sum row_sum --sizes 129

# All default size/dtype/operation combinations, without contacting the runner.
python tools/test_extended.py

# Rerun our compiler on existing fixtures after changing it, without retracing.
python tools/test_extended.py --reuse

# Generate the broad matrix without requiring our compiler to support it yet.
python tools/test_extended.py --generate-only --full-sizes

# Explicit selection, including real alternate schedules.
python tools/test_extended.py --dtypes f16 f32 --ops sum row_sum --sizes 127 129 1025 --rows 1 3 17 --work 64

# Hosted correctness tests: only our compiler's output is uploaded, serially.
python tools/test_extended.py --dtypes f32 --sizes 1 129 --reuse --runner
```

Case limits are available with --limit. Generating every case can take substantial time. Each invocation writes its own results.json; use --out with a different directory to preserve separate runs. The generator uses isolated temporary evidence files so machine-local tuning data does not select schedules.

## Interpreting results

- reference_failure: fix or investigate NumPy/PyTorch semantics first.
- generation_failure / packaging_failure: upstream tracing, scheduling or packaging issue; not our compiler failing a valid fixture.
- fixture_validation_failure: generated input dtype/shape or NumPy output dtype/size does not match the fixture metadata.
- compiler_unsupported_or_failure: generated case reveals an unsupported IR form or our compiler error. Inspect the saved error.
- local_assembly_failure / runner_assembly_failure: invalid or unsupported emitted PTX.
- runner_execution_failure / wrong_result: kernel execution or numerical correctness issue; investigate the IR and runner too.
- runner_output_conversion_failure: returned values cannot be represented in the declared NumPy output dtype. The raw values and error are retained.
- compiled_not_executed: local compilation succeeded, not a correctness pass.
- generated_not_tested: only fixture generation succeeded.
- pass: our generated PTX ran and matched every NumPy expected output.

Reports record the installed Torch/NumPy/Emmy versions. New versions of Torch can change tracing; the competition's PROVENANCE.md records the organizer's original versions. This suite does not claim to reproduce the private hidden set.

## Additional f16 validation

```bash
# Exact numeric checks on rounding-sensitive, subnormal and cancellation inputs.
# IR is unchanged from the real Emmy-generated f16 fixtures; only input data varies.
python tools/test_f16_edges.py --runner

# Fresh larger PyTorch -> Emmy cases with their own result report.
python tools/test_extended.py --dtypes f16 --ops add scale_shift sum row_sum --sizes 1024 --rows 3 --out tests/extended-f16-large --runner
```

Edge results are saved in review/f16-edge-results.json; larger-case results in
tests/extended-f16-large/results.json. The default matrix currently has 77 GPU
passes and three strict f64 reduction failures. Every case compiles and assembles;
all 16 cases for each of f16, f32, i32 and i64 pass. The small integer inputs in
this matrix do not establish correctness for wide integer reductions.

## Wide datatype validation

```bash
python -m unittest discover -s tests -p 'test_*.py'
python tools/test_numeric_edges.py --runner
python tools/test_i64_output_bits.py --reuse --runner
python tools/diagnose_reduction_precision.py --runner
```

Numeric edges reuse real Emmy IR with explicit input arrays and independent NumPy
expectations. The f64/i32 pointwise edge cases pass. Raw large i64 outputs lose
low bits in the hosted response, and some inputs or rounded outputs overflow its
conversion path. These failures remain in review/numeric-i64-raw-results.json
and review/numeric-edge-runner-input-limit-results.json.

The eight i64 bit-extraction cases are fresh PyTorch -> Emmy fixtures that retain
the arithmetic in the traced expression and extract low/high result bits on the
GPU before returning small exact integers. All eight pass. This tests native
integer arithmetic without relying on precise serialization of large outputs.
It does not turn the raw-output failures into passes.

The reduction diagnosis reproduces the f32 accumulation/shuffle/shared-memory
steps on the CPU. Its results match the GPU on all three failing f64 cases.
Local backend unit cases separately assemble all 25 datatype conversion pairs
and explicitly typed f64/i64 reduction state; these are not claimed as fresh
PyTorch-generated tests. See review/all-dtypes.md for details.

## Load the NumPy expected-value object

```python
from pathlib import Path
import numpy as np

case = Path("tests/extended/ext_add_f32_129")
with np.load(case / "expected.npz") as saved:
    expected = {name: saved[name].copy() for name in saved.files}

# Keys are the graph's actual output buffer names, not hardcoded labels.
for name, array in expected.items():
    print(name, array.shape, array.dtype)
```
