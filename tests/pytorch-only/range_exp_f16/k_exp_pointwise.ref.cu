#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const __half* x, __half* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        __half in0 = x[a0];
        __half v0 = hexp(in0);
        exp[a0] = v0;
    }
}
