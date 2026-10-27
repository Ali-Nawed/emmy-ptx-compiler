# Retired synthetic Kernel IR tests

These files are historical only and are excluded from active tests and coverage totals. The user requires every kernel test to use PyTorch expressions compiled by Emmy into Kernel IR; manually constructed or rewritten IR is not permitted.

`code/*.py.txt` preserves the retired generators and previous test implementations as text. `fixtures/` preserves their old synthetic inputs for audit history only. Do not run or use them as compiler correctness evidence. Current generators must use `tools/emmy_fixtures.py` to package actual Emmy dumps, and active tests must load genuine upstream IR rather than building or rewriting nodes.

The previous 88-case hand-typed backend result report is historical, not active coverage. Missing upstream support (for example amin/prod tracing or direct exp2/log2/log10 packaging) remains a documented generation gap; it must not be bypassed with synthetic Kernel IR.
