# Required kernel-test pipeline

Every kernel test must follow:

```text
PyTorch expression and input settings
    -> pinned Emmy compiler
    -> Emmy's Kernel IR and schedule
    -> our compile.py
    -> our PTX
    -> ptxas / hosted runner
```

This requirement applies to kernel unit tests as well as large regression suites. Do not construct test KernelOp/Tile/Assign/Accum/etc. nodes, change datatypes or dimensions inside parsed IR, substitute a combiner in an existing kernel, or fabricate a kernel because an upstream trace fails. Pure NumPy checker tests may test arithmetic directly; statement-level interpreter tests must read their statements from actual Emmy output and must not invent nodes.

**Scope rule:** if pinned Emmy cannot generate a case, treat that case as out of scope for this compiler effort. Record the upstream failure; do not invent IR to replace it. A failure under one schedule proves only that particular attempted case failed upstream. Compiler gaps are cases Emmy successfully generates which our compiler cannot translate or execute correctly.

## Where to set the cases

Change shapes, tensor datatypes, operations and cooperative settings in the PyTorch expression/spec before tracing. Explicit input-value patterns should be represented in the PyTorch setup and used consistently for metadata inputs, eager PyTorch checking and NumPy expectations. Changing input data does not permit changing the kernel's schedule or types afterward.

Keep both independent references: original NumPy expression expectations, and optional typed-IR interpreter expectations. A match to the IR does not erase a failure against the original expression.

## Proof that the IR came from Emmy

[tools/emmy_fixtures.py](../../tools/emmy_fixtures.py) wraps the original organizer packager without editing it. It compares the packaged kernel payload against Emmy's actual dump, allowing only the organizer's entry-name normalization. The body, types, axes, dimensions and schedule must match exactly.

It saves `emmy-proof.json` containing the PyTorch source, schedule knobs, package versions, target and hashes of the emitted/packaged IR. `verify()` rejects changed IR or a different source/kernel set. Missing proof is not silently invented: regenerate that case from its PyTorch source before strict reuse. Older legitimate fixture snapshots without this file are legacy evidence, not retroactively attested dumps.

The official packager can normalize an empty KernelOp entry name to the CUDA launch descriptor's entry. That ABI naming step is the sole allowed packaging normalization; our test code performs no body/schedule rewrite.

## Active tools

- `tools/test_extended.py`: original generated arithmetic/reduction matrix.
- `tools/test_operations.py`: composed expressions, max and softmax; unsupported amin/prod are optional upstream probes.
- `tools/validate_normalization.py --generate`: regenerate normalization through Emmy.
- `tools/test_pytorch_cases.py`: all 25 source/target datatype conversions and nine broad exp/log/power range cases, including explicit tensors in the PyTorch source.
- `tests/test_fixture_policy.py`: disallows IR constructors/rewrites in active test code, verifies fresh hashes, and checks that synthetic fixture families are absent.

Use the pinned Ubuntu environment. For example:

```bash
python tools/test_pytorch_cases.py --runner
python tools/validate_normalization.py --generate --runner
python -m unittest discover -s tests -v
```

`--reuse` on newly guarded generators requires a valid proof. If an older fixture has no proof, rerun without reuse. Supplied public problems are read from the competition's unchanged problem packages; they are not rewritten.

## What was retired

The synthetic generators `tools/test_custom.py` and `tools/test_operation_backends.py`, their hand-built fixture families, and earlier unit-test implementations are archived under [review/retired-synthetic-ir](../../review/retired-synthetic-ir/README.md). Their code is preserved as text and their fixtures are outside active tests.

The previous 88 hand-typed backend cases, including their special-value classifications and forced native-width reductions, are excluded from active coverage totals. The old 223-case aggregate is historical. Removing them leaves 135 previously generated/public cases, including seven retained f64 expression/IR precision gaps. The replacement 34-case conversion/range suite is generated through PyTorch and Emmy rather than manufactured from IR.

This withdraws active coverage claims for combinations that Emmy has not produced: amin/prod reductions, direct exp2/log2/log10 where packaging rejects them, forced 64-bit reduction states, and the former hand-typed special-value classification programs. Compiler handlers can remain implemented, but their existence is not proof of end-to-end tested support. Equivalent PyTorch expressions are valid tests only for the actual IR they produce.

Under the scope rule, upstream-rejected cases are excluded from our compiler backlog. The new genuine coverage probes and current regression results are in [native math and coverage](13-native-exp-log-and-coverage.md).

The original competition README, specifications, supplied problems, runner and generator remain unchanged. This policy changes only our compiler documentation and test setup.

## Validation after the change

All 34 newly generated conversion/range cases passed on the hosted GPU, against both original NumPy expectations and the typed-IR reference. All 29 revised local tests passed. The 18 normalization fixtures were also regenerated through the guarded pipeline and passed with the unchanged public examples, for 29 normalization/public cases.

The current operation aggregate excludes all 88 synthetic cases: 169 genuine generated/public cases, 162 original-expression passes, and seven preserved f64 precision differences whose GPU outputs match their supplied IR. All 169 GPU-versus-IR numeric checks passed. Of the active fixtures, the 34 replacement cases and 18 regenerated normalization cases now have fresh dump-checked proof files; older legitimate generated snapshots retain their earlier provenance rather than fabricated retrospective proof.
