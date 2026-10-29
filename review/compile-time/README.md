# Compile-time measurements

**Our actual command-line compiler and nvcc have similar latency on this local machine.** The already-imported Python API is much faster than a fresh Python process, but that is a separate boundary and is not the competition CLI's time.

## Method

- Ubuntu/WSL, Intel Core i7-10750H, Python 3.12.3, Emmy 0.3.11, CUDA/nvcc 12.0.140, target `sm_89`.
- Six representative kernels; one untimed warmup and seven timed repetitions. Alternate which compiler runs first. Report medians, with raw samples and ranges in [results.json](results.json).
- Our measured command is `python compile.py kernel.ir.json`, including process startup, Emmy imports, JSON/metadata parsing, emission and PTX output writing.
- nvcc receives the saved Emmy CUDA for the same kernel: `nvcc --ptx --gpu-architecture=compute_89 -O3 --std=c++17`. Emmy tracing, scheduling and CUDA rendering are excluded.
- Stage both input/output sets and our three compiler modules on Linux temporary storage to reduce mounted-Windows-file-system differences.
- Separately assemble each PTX with `ptxas -arch=sm_89 -O3`. No reference kernel is uploaded or executed.

## PTX generation: process-level medians

| Kernel | Our CLI, ms | nvcc, ms |
| --- | ---: | ---: |
| Public p00 add | 432.7 | 456.2 |
| Public p02 scale/shift | 423.0 | 419.2 |
| Public r04 row sum | 447.6 | 413.7 |
| f32 tanh | 419.7 | 426.1 |
| f16 scalar matmul, 16×8×16 | 492.1 | 435.7 |
| f16 staged tensor-core matmul, 16×8×16 | 425.0 | 430.3 |

These samples do not establish a material overall CLI speedup. Some kernels favor our CLI, others favor nvcc. CPU load/clocks were uncontrolled, OS caches were not flushed, and the measurements are local compilation times rather than hosted-runner or GPU execution timings.

## Assembly and in-process costs

Our PTX assembly medians were **32.9–39.8 ms**; assembly of nvcc PTX was **30.3–39.9 ms**. Median combined generation-plus-assembly times ranged **450.9–526.0 ms** for our path and **454.1–486.7 ms** for the nvcc path. Combined medians are calculated from paired total samples, not by adding separately computed medians.

With the Python modules already imported, 100 API measurements per case gave:

| Boundary | Median range across cases |
| --- | ---: |
| Parse IR/metadata + emit PTX | 0.78–2.99 ms |
| Emit from an already parsed kernel | 0.10–0.31 ms |

The small API costs show that startup/import overhead dominates the current CLI. Do not compare these API-only numbers with nvcc's process-level numbers to claim a speedup. The submission entry point remains a new Python CLI invocation.

## Reproduce

Use the [compiler setup](../../COMPILER_README.md), with `nvcc` and `ptxas` on `PATH`:

```bash
python tools/benchmark_compile_time.py
```

Optional selections:

```bash
python tools/benchmark_compile_time.py --only p00_add tensor_matmul_f16 --out /tmp/emmy-compile-time
```

[results.json](results.json) records versions, source/input hashes, commands and individual samples. The benchmark does not modify the compiler, supplied problems or expected outputs. It compares IR-to-PTX against CUDA-to-PTX, not end-to-end PyTorch compilation, and it does not establish identical numeric behavior between our PTX and reference CUDA.
