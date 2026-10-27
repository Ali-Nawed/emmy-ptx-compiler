#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const __half* y, const __half* x, __half* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        __half in0 = x[0];
        __half in1 = y[0];
        bool v0 = __half2float(in0) > __half2float(in1);
        __half v1 = __float2half(((v0) ? (__half2float(in0)) : (__half2float(in1))));
        where[0] = v1;
    }
}
