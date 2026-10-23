"""PyTorch expressions + independent NumPy references for the extended suite.

Change shapes/dtypes here or use tools/test_extended.py's matrix options.
No serialized Kernel IR is edited. Emmy chooses the schedule from each expression.
"""

from functools import partial

import numpy as np

DTYPES = {"f16": "float16", "f32": "float32", "f64": "float64", "i32": "int32", "i64": "int64"}
OPS = ("add", "subtract", "multiply", "scale_shift", "relu_mul", "sum", "mean", "row_sum")
SIZES = (1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 4096, 65536)


def numpy_reference(*arrays, operation, dtype):
    x = arrays[0]
    if operation == "add":
        result = x + arrays[1]
    elif operation == "subtract":
        result = x - arrays[1]
    elif operation == "multiply":
        result = x * arrays[1]
    elif operation == "scale_shift":
        scale, shift = (2, -3) if dtype.startswith("i") else (-1.25, 0.5)
        result = x * np.asarray(scale, dtype=x.dtype) + np.asarray(shift, dtype=x.dtype)
    elif operation == "relu_mul":
        result = np.maximum(x, np.asarray(0, dtype=x.dtype)) * arrays[1]
    elif operation in ("sum", "row_sum", "mean"):
        integer = dtype.startswith("i")
        wide = x.astype(np.int64 if integer and operation != "mean" else np.float64)
        if operation == "mean":
            result = wide.mean().astype(np.float32 if integer else x.dtype)
        else:
            result = wide.sum(axis=-1 if operation == "row_sum" else None)
            result = result.astype(np.int64 if integer else x.dtype)
    else:
        raise ValueError(operation)
    return [np.asarray(result)]


def build_specs(dtypes=("f16", "f32", "f64", "i32", "i64"), sizes=SIZES, operations=OPS,
                row_counts=(1, 3, 17), work=128):
    specs = []
    for dtype in dtypes:
        for operation in operations:
            for size in sizes:
                for rows in row_counts if operation == "row_sum" else (None,):
                    shape = (rows, size) if rows is not None else (size,)
                    shape_text = repr(shape)
                    ctor = (f"torch.randint(-8, 9, {shape_text}, dtype=torch.{DTYPES[dtype]})"
                            if dtype.startswith("i") else
                            f"torch.randn({shape_text}, dtype=torch.{DTYPES[dtype]})")
                    setup = f"x = {ctor}; "
                    if operation in ("add", "subtract", "multiply", "relu_mul"):
                        setup += f"y = {ctor}; "
                    tail = {
                        "add": "x + y", "subtract": "x - y", "multiply": "x * y",
                        "relu_mul": "torch.relu(x) * y", "sum": "x.sum()",
                        "row_sum": "x.sum(-1)",
                        "mean": "x.mean(dtype=torch.float32)" if dtype.startswith("i") else "x.mean()",
                        "scale_shift": "x * 2 - 3" if dtype.startswith("i") else "x * -1.25 + 0.5",
                    }[operation]
                    integer = dtype.startswith("i") and operation != "mean"
                    tolerance = (0, 0) if integer or operation in ("add", "subtract", "multiply", "relu_mul") else {
                        "f16": (5e-3, 5e-3), "f32": (1e-4, 1e-4), "f64": (1e-10, 1e-10),
                        "i32": (1e-5, 1e-5), "i64": (1e-5, 1e-5),
                    }[dtype]
                    suffix = "x".join(map(str, shape))
                    spec = dict(id=f"ext_{operation}_{dtype}_{suffix}", code=setup + tail,
                                ref=partial(numpy_reference, operation=operation, dtype=dtype),
                                rtol=tolerance[0], atol=tolerance[1],
                                dtype=dtype, operation=operation, shape=list(shape))
                    if operation in ("sum", "mean", "row_sum"):
                        spec["knobs"] = f"WORK=t{work},REDUCE=coop"
                    specs.append(spec)
    return specs


# Compatible with the organizer's unmodified tools/generate.py.
OPEN = build_specs(dtypes=("f32",), sizes=(129,), row_counts=(3,))
