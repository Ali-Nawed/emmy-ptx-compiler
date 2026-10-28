# Our Kernel IR to PTX compiler

For setup, compilation and validation commands, see the root [compiler run guide](../../COMPILER_README.md). For the short presentation, see the [walkthrough](18-presentation-walkthrough.md).

Read these notes alongside [compile.py](../../compile.py). They describe the current implementation, including its limitations; they are not a plan for an additional compiler framework.

## Suggested reading order

1. [Loading the IR and the submission contract](01-loading-and-contract.md)
2. [Launch geometry and Tile coordinates](02-launch-and-tiles.md)
3. [Datatypes, conversions, constants and registers](03-datatypes-and-registers.md)
4. [Addresses, loads and stores](04-memory.md)
5. [Arithmetic operations](05-arithmetic.md)
6. [Conditions, ordinary loops and strided loops](06-control-flow.md)
7. [Accumulators, warp shuffles and shared reductions](07-reductions.md)
8. [L2 normalization and RMSNorm](08-normalization.md)
9. [Implemented optimizations and their safety conditions](09-optimizations.md)
10. [Testing, evidence and unsupported features](10-testing-and-coverage.md)
11. [Operation expansion: reasoning, math and validation](11-operation-expansion.md)
12. [Required PyTorch-to-Emmy test pipeline](12-required-test-pipeline.md)
13. [Native exp/log: implementation, validation and coverage gaps](13-native-exp-log-and-coverage.md)
14. [Boolean comparisons and where](14-boolean-comparisons-and-where.md)
15. [Sigmoid and tanh: native instructions and accuracy](15-sigmoid-and-tanh.md)
16. [First native tensor-core matmul](16-first-tensor-core-matmul.md)
17. [Shared-memory staging and native asynchronous copies](17-shared-memory-async-copies.md)
18. [Presentation walkthrough and feature/run checklist](18-presentation-walkthrough.md)
19. [Hidden-style MVP coverage: fresh PyTorch matrix and remaining gaps](19-mvp-hidden-coverage.md)

## The complete path

```text
Kernel IR JSON + sibling meta.json
    load_kernel(): Emmy Graph.from_dict -> typed objects + launch metadata
    emit_ptx(): construct PTXEmitter
    compile(): parameters, launch coordinates, body, declarations, return
    statements()/statement(): translate each typed statement
    PTX text on stdout
    ptxas: assemble and allocate physical registers downstream
```

The translation is ours. Emmy provides the input types/parser; our submission does not call Emmy's CUDA renderer, nvcc, LLVM or another compiler to translate the input. The local test generator may use upstream tooling to produce fixtures and metadata. Only PTX emitted by our own `compile.py` is sent to the hosted runner.

`compile.py` imports our [ptx_math.py](../../ptx_math.py) for scalar math and [ptx_tensor.py](../../ptx_tensor.py) for tensor-core fragment emission. Keep all three files at the submission repository root. These modules are our emission code, not another compiler or a new IR pass.

## Where state lives

| Emitter field | What it remembers |
| --- | --- |
| `values` | IR value/axis name -> PTX register or immediate |
| `value_dtypes` | Numeric datatype of a bound data value |
| `buffers` | Buffer name -> shape, element dtype and argument metadata |
| `pointers` | Argument buffer name -> global pointer register |
| `shared` | Scratch buffer name -> shared address register |
| `buffer_alignments` | Guaranteed base alignment used for grouped stores |
| `counts` | Fresh virtual register counters by register bank |
| `lines` / `declarations` | Instructions and separate storage declarations |
| `combine_dtype` | Default arithmetic type while emitting a reduction merge |

`Tile` and loop coordinates are index values; regular loaded/computed data also has a `value_dtypes` entry. That distinction keeps integer address calculations separate from tensor arithmetic.

For the organizer's full vocabulary and contract, see [TASK.md](../../TASK.md) and [MVP.md](../../MVP.md).

The earlier sqrt/reciprocal baseline passed 25 local tests and 28 hosted cases. Subsequent exp/log/power and nonadditive reduction results are recorded separately in [the operation expansion notes](11-operation-expansion.md).

**Current testing rule:** all kernel tests must originate from PyTorch expressions compiled by Emmy. Synthetic IR tests and their former 88-case coverage claim have been retired. [The required pipeline](12-required-test-pipeline.md) supersedes the historical testing descriptions.
