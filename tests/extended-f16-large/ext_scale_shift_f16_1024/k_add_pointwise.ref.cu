#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const __half* mul_c1, const __half* add_c1, const __half* x, __half* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid;
        __half in1__u0 = mul_c1[0];
        __half in2__u0 = x[a0 * 2];
        __half in0__u0 = add_c1[0];
        __half v1__u0 = in0__u0 + (in1__u0 * in2__u0);
        __half in1__u1 = mul_c1[0];
        __half in2__u1 = x[a0 * 2 + 1];
        __half in0__u1 = add_c1[0];
        __half v1__u1 = in0__u1 + (in1__u1 * in2__u1);
        __half2 _vs_v1__u0_13c38830 = __halves2half2(v1__u0, v1__u1);
        *reinterpret_cast<__half2*>(&add[a0 * 2]) = _vs_v1__u0_13c38830;
}
