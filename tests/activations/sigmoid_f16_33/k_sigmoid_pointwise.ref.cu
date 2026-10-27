#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_sigmoid_pointwise(const __half* x, __half* sigmoid) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        __half in0__u0 = x[a0 * 3];
        __half v0__u0 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u0))));
        __half in0__u1 = x[a0 * 3 + 1];
        __half v0__u1 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u1))));
        __half in0__u2 = x[a0 * 3 + 2];
        __half v0__u2 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0__u2))));
        sigmoid[a0 * 3] = v0__u0;
        sigmoid[a0 * 3 + 1] = v0__u1;
        sigmoid[a0 * 3 + 2] = v0__u2;
    }
}
