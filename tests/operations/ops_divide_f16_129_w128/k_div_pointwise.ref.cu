#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const __half* add_c1, const __half* y, const __half* x, __half* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        __half in2__u0 = y[a0 * 3];
        float v0__u0 = __half2float(in2__u0) * __half2float(in2__u0);
        __half in0__u0 = add_c1[0];
        float v1__u0 = __half2float(in0__u0) + v0__u0;
        __half in1__u0 = x[a0 * 3];
        float v3__u0 = __half2float((in1__u0 - in2__u0)) / v1__u0;
        __half in2__u1 = y[a0 * 3 + 1];
        float v0__u1 = __half2float(in2__u1) * __half2float(in2__u1);
        __half in0__u1 = add_c1[0];
        float v1__u1 = __half2float(in0__u1) + v0__u1;
        __half in1__u1 = x[a0 * 3 + 1];
        float v3__u1 = __half2float((in1__u1 - in2__u1)) / v1__u1;
        __half in2__u2 = y[a0 * 3 + 2];
        float v0__u2 = __half2float(in2__u2) * __half2float(in2__u2);
        __half in0__u2 = add_c1[0];
        float v1__u2 = __half2float(in0__u2) + v0__u2;
        __half in1__u2 = x[a0 * 3 + 2];
        float v3__u2 = __half2float((in1__u2 - in2__u2)) / v1__u2;
        div[a0 * 3] = __float2half(v3__u0);
        div[a0 * 3 + 1] = __float2half(v3__u1);
        div[a0 * 3 + 2] = __float2half(v3__u2);
    }
}
