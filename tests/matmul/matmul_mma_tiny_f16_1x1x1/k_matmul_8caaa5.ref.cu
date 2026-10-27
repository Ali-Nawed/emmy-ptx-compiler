#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul_8caaa5(const __half* b, const __half* a, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = a[0];
        __half in1 = b[0];
        __half v0 = in0 * in1;
        matmul[0] = v0;
    }
}
