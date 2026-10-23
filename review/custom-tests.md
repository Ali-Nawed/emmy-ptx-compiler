# Custom tests and runner limits

The documented API has no stated requests-per-minute or daily quota. A health check on October 22, 2026 returned no rate-limit headers. This does not establish that no additional server-side quota exists.

Documented bounds (docs/API.md):

- Launches execute one at a time; a full queue returns stage=busy. submit.py waits two seconds and retries, up to 60 attempts.
- At most 20 warmup launches and 200 timed iterations per run request.
- 20-second budget for warmup, timed runs and copies together; 30 seconds for PTX assembly.
- PTX source: at most 2,000,000 characters.
- Buffers: at most 2^26 elements total across all arguments.
- Returned output: at most 2^22 values total.
- Dynamic shared memory: at most 99 KiB per block.
- Compilations are cached by PTX, architecture, ptxas version and flags.

## Custom fixture workflow

Run in Ubuntu from the repository root:

```bash
# Generate custom cases and compile/assemble locally:
.venv/bin/python tools/test_custom.py

# Also check numerical results using only our compiler's PTX on the runner:
.venv/bin/python tools/test_custom.py --runner
```

Generated inputs, metadata, NumPy expected outputs and our PTX are under tests/generated/. The summary is review/custom-test-results.json. Tests run serially with a one-second courtesy delay; that delay is not a published rate limit.

The tool modifies typed Emmy objects and serializes through Graph.to_dict. It does not parse serialized body strings or use Emmy's CUDA renderer. These are synthetic variants of known schedules, not fresh schedules emitted by the full Emmy pipeline.

Cases cover:

- Add lengths 2, 130, 514 and 1026: tiny workloads and partially occupied final blocks.
- Scale/shift lengths 12 and 1028 with new constants -1.25 and 0.3.
- ReLU/multiply lengths 20 and 1036 with explicit negative, zero and positive values.
- Sum lengths 1, 31, 127, 129 and 1003: idle accumulator lanes, boundaries and uneven loop trip counts.
- Row-sum shapes 3x129 and 7x513: changed row counts and uneven row lengths.

This suite checks current supported operations only. Remaining coverage needs include additional operators (subtract, maximum/minimum reductions), different cooperative widths and shared trees, grouped loads, metadata parameter permutations, and fresh Emmy-generated schedules. Passing these cases cannot predict hidden-set results. Compiler rejections are coverage gaps, not reasons to discard legitimate tests.
