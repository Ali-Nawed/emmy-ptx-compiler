#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mean_2a9be0(const __half* x, const __half* mean_keepdim_count, __half* mean) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = mean_keepdim_count[0];
        __half in1 = x[0];
        __half v0 = in1 / in0;
        mean[0] = v0;
    }
}
