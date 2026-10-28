#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const int* y, const __half* x, __half* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 33) {
        int a0 = _gid / 11;
        int a1 = (_gid) % 11;
        __half in0__u0 = x[a0 * 33 + a1 * 3];
        int in1__u0 = y[a1 * 3];
        __half v0__u0 = __float2half(__half2float(in0__u0) + in1__u0);
        __half v1__u0 = __float2half(__half2float(in0__u0) - in1__u0);
        __half v2__u0 = v0__u0 * v1__u0;
        __half in0__u1 = x[a0 * 33 + (a1 * 3 + 1)];
        int in1__u1 = y[a1 * 3 + 1];
        __half v0__u1 = __float2half(__half2float(in0__u1) + in1__u1);
        __half v1__u1 = __float2half(__half2float(in0__u1) - in1__u1);
        __half v2__u1 = v0__u1 * v1__u1;
        __half in0__u2 = x[a0 * 33 + (a1 * 3 + 2)];
        int in1__u2 = y[a1 * 3 + 2];
        __half v0__u2 = __float2half(__half2float(in0__u2) + in1__u2);
        __half v1__u2 = __float2half(__half2float(in0__u2) - in1__u2);
        __half v2__u2 = v0__u2 * v1__u2;
        mul[a0 * 33 + a1 * 3] = v2__u0;
        mul[a0 * 33 + (a1 * 3 + 1)] = v2__u1;
        mul[a0 * 33 + (a1 * 3 + 2)] = v2__u2;
    }
}
