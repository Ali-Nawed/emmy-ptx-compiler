#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_rsqrt_pointwise(const __half* x, const __half* add_c1, __half* rsqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in1 = x[0];
        float v0 = __half2float(in1) * __half2float(in1);
        __half in0 = add_c1[0];
        float v1 = __half2float(in0) + v0;
        float v2 = rsqrtf(v1);
        rsqrt[0] = __float2half(v2);
    }
}
