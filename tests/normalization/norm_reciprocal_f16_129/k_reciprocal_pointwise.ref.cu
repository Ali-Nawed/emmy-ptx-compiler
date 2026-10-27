#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_reciprocal_pointwise(const __half* add_c1, const __half* x, __half* reciprocal) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        __half in1__u0 = x[a0 * 3];
        float v0__u0 = __half2float(in1__u0) * __half2float(in1__u0);
        __half in0__u0 = add_c1[0];
        float v1__u0 = __half2float(in0__u0) + v0__u0;
        float v2__u0 = 1.0f / v1__u0;
        __half in1__u1 = x[a0 * 3 + 1];
        float v0__u1 = __half2float(in1__u1) * __half2float(in1__u1);
        __half in0__u1 = add_c1[0];
        float v1__u1 = __half2float(in0__u1) + v0__u1;
        float v2__u1 = 1.0f / v1__u1;
        __half in1__u2 = x[a0 * 3 + 2];
        float v0__u2 = __half2float(in1__u2) * __half2float(in1__u2);
        __half in0__u2 = add_c1[0];
        float v1__u2 = __half2float(in0__u2) + v0__u2;
        float v2__u2 = 1.0f / v1__u2;
        reciprocal[a0 * 3] = __float2half(v2__u0);
        reciprocal[a0 * 3 + 1] = __float2half(v2__u1);
        reciprocal[a0 * 3 + 2] = __float2half(v2__u2);
    }
}
