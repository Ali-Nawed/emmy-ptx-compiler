# Boolean comparisons and where

`torch.where(condition, x, y)` chooses x or y independently at each tensor position. The compiler now handles boolean SSA values produced by comparisons and logical operations, followed by numeric selection. This is direct emission from Emmy's original IR; it adds no pass or intermediate IR.

## Read the implementation

In [compile.py](../../compile.py), the `Assign` handler calls `boolean_assign()` before ordinary numeric arithmetic. This matters because a comparison has numeric inputs and a boolean output, while where has a boolean condition and numeric alternatives. Converting every argument to the result datatype would destroy those distinctions.

`comparison()` determines the operands' numeric working type and emits `setp`. `data_reg('bool')` allocates a PTX predicate register, and `value_dtypes` records the boolean binding. Equal-width integer comparisons stay signed integer operations; i64 operands do not pass through floating point. Mixed integer widths use i64. Floating operands use the wider floating type, with f16 values already represented exactly in f32 working registers.

For mixed integer/f32 or integer/f64 comparisons, floating operands dominate. This matches Emmy's bare C++ comparison rendering, including the f32 constant it emits for `x > 0` with an integer tensor. Such mixed comparisons can inherently lose integer precision; the compiler does not silently replace Emmy's types with a different operation. The independent NumPy interpreter explicitly implements this rule because NumPy's default i64/f32 promotion would instead choose f64.

| Emmy operation | PTX comparison |
| --- | --- |
| greater / greater_equal | gt / ge |
| less / less_equal | lt / le |
| equal | eq |
| not_equal | ne for integers; neu for floating values |

Floating `neu` means not-equal **or unordered**, so NaN != anything is true. The other listed floating comparisons are ordered and return false for NaN.

Boolean AND/OR/XOR/NOT use predicate instructions. The implementation accepts Emmy's logical operation names and its bitwise names on boolean values; numeric integer bitwise operations retain their existing path. The generated tests currently exercise bitwise AND, OR and NOT on comparison results. Extra recognized names alone are not evidence of tested end-to-end support.

For numeric `where`, the condition remains a predicate and only the two alternatives are converted to the declared output datatype:

```ptx
setp.gt.f32 %p2, %f1, %f2;
selp.f32 %f3, %f1, %f2, %p2;
```

The select itself requires no branch. i64 selection uses `selp.b64`, preserving the entire value's bits. f64 remains f64; f16 uses f32 working selection followed by the usual half operation-boundary rounding. Broadcasting and grouped stores continue to follow Emmy's supplied indices and the existing memory emitter.

## Genuine examples and evidence

[tools/test_where.py](../../tools/test_where.py) generates all cases from PyTorch through pinned Emmy 0.3.11. Packaged IR is compared against Emmy's dump and gets `emmy-proof.json`. No body, datatype, shape or schedule is rewritten. NumPy expression expectations and the independent typed-IR interpreter remain separate references.

The 41 distinct examples cover five numeric output datatypes at sizes 1/33/129, scalar-zero comparisons, all six comparison relations, predicate AND/OR/NOT, nested where, row/column broadcasting, a selected-value sum, adjacent large i64 values and NaN comparisons. Fixtures include source expressions, unchanged IR and our emitted PTX. [The report](../../review/where/validation.json) records current source hashes and both comparisons.

All 41 cases passed on the hosted GPU against both references, without loosening their selected comparison tolerances. All 37 local tests passed, including all fixture provenance/assembly/reference checks, wide-integer instruction selection and unordered NaN comparison emission. Repeated public and serial-loop regression checks are in [the 11-case public report](../../review/native-math/where-regression.json) and [the 18-case Loop report](../../review/serial-loops/validation.json).

For the wide-integer cases, PyTorch adds 2^60 to small integer tensors inside the kernel, compares the resulting adjacent i64 values, then returns small selected inputs. This catches narrowing to either f32 or f64 without depending on precision of large integers sent through the hosted interface. An earlier experiment sending/returning adjacent large values produced ambiguous remote failures and is preserved in [the initial diagnostic report](../../review/where/initial-direct-wide-values.json); it is not counted as passing coverage. We did not change tolerances to hide those results.

For NaN cases, PyTorch computes x/x from finite input tensors containing zero, compares the result, and selects finite y or -y. This exercises NaN comparison behavior on the GPU without sending nonfinite JSON numbers or asking the ordinary checker to accept NaN outputs.

Run from the repo in the pinned Ubuntu environment:

```bash
python tools/test_where.py --runner
python tools/test_where.py --reuse --runner
python -m unittest discover -s tests -p 'test_*.py'
```

Only our compiler's PTX is uploaded. Reference CUDA emitted by the fixture packager is available for reading but is never submitted or used to generate our PTX. Original competition specifications, generator, runner and public problems remain unchanged.

## Limits

This supports internal boolean SSA values and numeric where outputs. Boolean memory arguments/outputs are outside the runner's documented five-type ABI and are not implemented. Boolean-valued where results, comparisons between boolean predicates, arbitrary condition expressions, and mixed half/integer comparison operands are not covered by this implementation. An unsupported form fails explicitly. This does not add the separate coordinate-predicated `Select` statement.

These tests do not establish every datatype/shape/scheduling combination, every exceptional selected output, or hidden-set coverage. The full older 169-case suite was not rerun for this change; current public and serial-loop checks are recorded separately from historical reports. Sigmoid and tanh were subsequently implemented; see [their accuracy checks](15-sigmoid-and-tanh.md). Exploration of genuine Emmy-generated tensor-core/copy-engine programs follows.
