#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const __half* x, const __half* y, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = x[0];
        __half in1 = y[0];
        __half v1 = in1 * fmaxf(0.0f, in0);
        mul[0] = v1;
    }
}
