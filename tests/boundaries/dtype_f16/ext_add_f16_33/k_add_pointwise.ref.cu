#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const __half* x, const __half* y, __half* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        __half in0__u0 = x[a0 * 3];
        __half in1__u0 = y[a0 * 3];
        __half v0__u0 = in0__u0 + in1__u0;
        __half in0__u1 = x[a0 * 3 + 1];
        __half in1__u1 = y[a0 * 3 + 1];
        __half v0__u1 = in0__u1 + in1__u1;
        __half in0__u2 = x[a0 * 3 + 2];
        __half in1__u2 = y[a0 * 3 + 2];
        __half v0__u2 = in0__u2 + in1__u2;
        add[a0 * 3] = v0__u0;
        add[a0 * 3 + 1] = v0__u1;
        add[a0 * 3 + 2] = v0__u2;
    }
}
