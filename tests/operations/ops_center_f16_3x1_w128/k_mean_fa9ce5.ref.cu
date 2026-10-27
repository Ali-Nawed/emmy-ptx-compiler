#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_mean_fa9ce5(const __half* mean_count, const __half* x, __half* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        __half in0 = mean_count[0];
        __half in1 = x[a0];
        __half v2 = in1 - (in1 * (1.0f / in0));
        sub[a0] = v2;
    }
}
