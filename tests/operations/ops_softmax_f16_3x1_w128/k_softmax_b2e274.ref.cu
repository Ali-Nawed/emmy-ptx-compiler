#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_softmax_b2e274(const __half* x, __half* softmax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        __half in0 = x[a0];
        __half v1 = hexp((in0 - in0));
        __half v2 = v1 / v1;
        softmax[a0] = v2;
    }
}
