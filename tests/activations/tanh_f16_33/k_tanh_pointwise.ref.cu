#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const __half* x, __half* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        __half in0__u0 = x[a0 * 3];
        __half v0__u0 = htanh(in0__u0);
        __half in0__u1 = x[a0 * 3 + 1];
        __half v0__u1 = htanh(in0__u1);
        __half in0__u2 = x[a0 * 3 + 2];
        __half v0__u2 = htanh(in0__u2);
        tanh[a0 * 3] = v0__u0;
        tanh[a0 * 3 + 1] = v0__u1;
        tanh[a0 * 3 + 2] = v0__u2;
    }
}
