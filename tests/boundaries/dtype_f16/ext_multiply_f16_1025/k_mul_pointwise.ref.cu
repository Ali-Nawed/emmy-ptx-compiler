#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __half* x, const __half* y, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1025) {
        int a0 = _gid;
        __half in0 = x[a0];
        __half in1 = y[a0];
        __half v0 = in0 * in1;
        mul[a0] = v0;
    }
}
