#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_3_pointwise(const __half* mul_1_c1, const __half* add_c1, const __half* mul_3_c1, const __half* add_2_c1, const __half* x, const __half* y, __half* add_3) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        __half in4__u0 = x[a0 * 3];
        float v0__u0 = __half2float(in4__u0) * __half2float(in4__u0);
        __half in0__u0 = add_2_c1[0];
        float v1__u0 = __half2float(in0__u0) + v0__u0;
        float v2__u0 = __half2float(in4__u0) * __half2float(in4__u0);
        __half in1__u0 = add_c1[0];
        float v3__u0 = __half2float(in1__u0) + v2__u0;
        __half in2__u0 = mul_1_c1[0];
        __half in5__u0 = y[a0 * 3];
        float v7__u0 = log(v3__u0) + __half2float(expf((in2__u0 * in5__u0)));
        __half in3__u0 = mul_3_c1[0];
        float v9__u0 = powf(v1__u0, __half2float((in3__u0 * in5__u0)));
        float v10__u0 = v7__u0 + v9__u0;
        __half in4__u1 = x[a0 * 3 + 1];
        float v0__u1 = __half2float(in4__u1) * __half2float(in4__u1);
        __half in0__u1 = add_2_c1[0];
        float v1__u1 = __half2float(in0__u1) + v0__u1;
        float v2__u1 = __half2float(in4__u1) * __half2float(in4__u1);
        __half in1__u1 = add_c1[0];
        float v3__u1 = __half2float(in1__u1) + v2__u1;
        __half in2__u1 = mul_1_c1[0];
        __half in5__u1 = y[a0 * 3 + 1];
        float v7__u1 = log(v3__u1) + __half2float(expf((in2__u1 * in5__u1)));
        __half in3__u1 = mul_3_c1[0];
        float v9__u1 = powf(v1__u1, __half2float((in3__u1 * in5__u1)));
        float v10__u1 = v7__u1 + v9__u1;
        __half in4__u2 = x[a0 * 3 + 2];
        float v0__u2 = __half2float(in4__u2) * __half2float(in4__u2);
        __half in0__u2 = add_2_c1[0];
        float v1__u2 = __half2float(in0__u2) + v0__u2;
        float v2__u2 = __half2float(in4__u2) * __half2float(in4__u2);
        __half in1__u2 = add_c1[0];
        float v3__u2 = __half2float(in1__u2) + v2__u2;
        __half in2__u2 = mul_1_c1[0];
        __half in5__u2 = y[a0 * 3 + 2];
        float v7__u2 = log(v3__u2) + __half2float(expf((in2__u2 * in5__u2)));
        __half in3__u2 = mul_3_c1[0];
        float v9__u2 = powf(v1__u2, __half2float((in3__u2 * in5__u2)));
        float v10__u2 = v7__u2 + v9__u2;
        add_3[a0 * 3] = __float2half(v10__u0);
        add_3[a0 * 3 + 1] = __float2half(v10__u1);
        add_3[a0 * 3 + 2] = __float2half(v10__u2);
    }
}
