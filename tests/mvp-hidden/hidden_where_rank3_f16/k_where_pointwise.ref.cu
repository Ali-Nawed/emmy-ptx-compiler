#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const __half* y, const __half* x, __half* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 99) {
        int a0 = _gid;
        __half in0__u0 = x[a0 * 2];
        __half in1__u0 = y[a0 * 2];
        bool v1__u0 = __half2float(in0__u0) > __half2float(in1__u0);
        __half v3__u0 = __float2half(((v1__u0) ? (__half2float((in0__u0 - in1__u0))) : (__half2float((in0__u0 + in1__u0)))));
        __half in0__u1 = x[a0 * 2 + 1];
        __half in1__u1 = y[a0 * 2 + 1];
        bool v1__u1 = __half2float(in0__u1) > __half2float(in1__u1);
        __half v3__u1 = __float2half(((v1__u1) ? (__half2float((in0__u1 - in1__u1))) : (__half2float((in0__u1 + in1__u1)))));
        __half2 _vs_v3__u0_7ad3dd4c = __halves2half2(v3__u0, v3__u1);
        *reinterpret_cast<__half2*>(&where[a0 * 2]) = _vs_v3__u0_7ad3dd4c;
    }
}
