#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const __half* y, const __half* x, __half* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = x[0];
        __half in1 = y[0];
        __half v2 = fmaxf(in0, in1) - fminf(in0, in1);
        sub[0] = v2;
    }
}
