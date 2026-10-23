#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const __half* x, const __half* y, __half* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = x[0];
        __half in1 = y[0];
        __half v0 = in0 + in1;
        add[0] = v0;
    }
}
