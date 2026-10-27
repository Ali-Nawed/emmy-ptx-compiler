#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const __half* x, const __half* mul_c1, const __half* mul_1_c1, __half* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in1 = mul_c1[0];
        __half in2 = x[0];
        __half in0 = mul_1_c1[0];
        __half v2 = hexp((in0 * (in1 * in2)));
        exp[0] = v2;
    }
}
