#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const __half* add_c1, const __half* div_c1, const __half* x, __half* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        __half in1__u0 = div_c1[0];
        __half in2__u0 = x[a0 * 3];
        float v1__u0 = __half2float(in2__u0) * __half2float(in2__u0);
        __half in0__u0 = add_c1[0];
        float v2__u0 = __half2float(in0__u0) + v1__u0;
        float v4__u0 = __half2float((1.0f / in1__u0)) * log(v2__u0);
        __half in1__u1 = div_c1[0];
        __half in2__u1 = x[a0 * 3 + 1];
        float v1__u1 = __half2float(in2__u1) * __half2float(in2__u1);
        __half in0__u1 = add_c1[0];
        float v2__u1 = __half2float(in0__u1) + v1__u1;
        float v4__u1 = __half2float((1.0f / in1__u1)) * log(v2__u1);
        __half in1__u2 = div_c1[0];
        __half in2__u2 = x[a0 * 3 + 2];
        float v1__u2 = __half2float(in2__u2) * __half2float(in2__u2);
        __half in0__u2 = add_c1[0];
        float v2__u2 = __half2float(in0__u2) + v1__u2;
        float v4__u2 = __half2float((1.0f / in1__u2)) * log(v2__u2);
        div[a0 * 3] = __float2half(v4__u0);
        div[a0 * 3 + 1] = __float2half(v4__u1);
        div[a0 * 3 + 2] = __float2half(v4__u2);
    }
}
