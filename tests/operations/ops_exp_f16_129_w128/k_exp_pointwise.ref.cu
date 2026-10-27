#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const __half* mul_c1, const __half* x, __half* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        __half in0__u0 = mul_c1[0];
        __half in1__u0 = x[a0 * 3];
        __half v1__u0 = hexp((in0__u0 * in1__u0));
        __half in0__u1 = mul_c1[0];
        __half in1__u1 = x[a0 * 3 + 1];
        __half v1__u1 = hexp((in0__u1 * in1__u1));
        __half in0__u2 = mul_c1[0];
        __half in1__u2 = x[a0 * 3 + 2];
        __half v1__u2 = hexp((in0__u2 * in1__u2));
        exp[a0 * 3] = v1__u0;
        exp[a0 * 3 + 1] = v1__u1;
        exp[a0 * 3 + 2] = v1__u2;
    }
}
