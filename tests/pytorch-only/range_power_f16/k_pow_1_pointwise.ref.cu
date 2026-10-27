#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const __half* x, const __half* y, __half* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        __half in0 = x[a0];
        __half in1 = y[a0];
        float v0 = powf(__half2float(in0), __half2float(in1));
        pow_1[a0] = __float2half(v0);
    }
}
