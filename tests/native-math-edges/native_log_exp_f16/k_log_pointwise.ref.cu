#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_log_pointwise(const __half* x, __half* log) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        __half in0 = x[a0];
        __half v1 = hlog(expf(in0));
        log[a0] = v1;
    }
}
