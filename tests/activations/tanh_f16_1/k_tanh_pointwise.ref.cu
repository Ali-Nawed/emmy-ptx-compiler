#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const __half* x, __half* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = x[0];
        __half v0 = htanh(in0);
        tanh[0] = v0;
    }
}
