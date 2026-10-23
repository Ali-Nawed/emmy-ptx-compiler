#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const __half* mul_c1, const __half* add_c1, const __half* x, __half* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        __half in1 = mul_c1[0];
        __half in2 = x[a0];
        __half in0 = add_c1[0];
        __half v1 = in0 + (in1 * in2);
        add[a0] = v1;
    }
}
