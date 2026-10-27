#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_sum_1_pointwise(const __half* add_c1, const __half* y, const __half* x, __half* sum_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        __half in2 = y[a0];
        float v0 = __half2float(in2) * __half2float(in2);
        __half in0 = add_c1[0];
        float v1 = __half2float(in0) + v0;
        __half in1 = x[a0];
        float v3 = __half2float((in1 - in2)) / v1;
        sum_1[a0] = __float2half(v3);
    }
}
