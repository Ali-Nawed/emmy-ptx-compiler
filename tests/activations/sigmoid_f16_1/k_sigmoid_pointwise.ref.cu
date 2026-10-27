#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_sigmoid_pointwise(const __half* x, __half* sigmoid) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = x[0];
        __half v0 = __float2half(1.0f / (1.0f + expf(0.0f - __half2float(in0))));
        sigmoid[0] = v0;
    }
}
