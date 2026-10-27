#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const __half* y, const __half* x, __half* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        __half in0__u0 = x[a0 * 3];
        __half in1__u0 = y[a0 * 3];
        bool v0__u0 = __half2float(in0__u0) > __half2float(in1__u0);
        __half v1__u0 = __float2half(((v0__u0) ? (__half2float(in0__u0)) : (__half2float(in1__u0))));
        __half in0__u1 = x[a0 * 3 + 1];
        __half in1__u1 = y[a0 * 3 + 1];
        bool v0__u1 = __half2float(in0__u1) > __half2float(in1__u1);
        __half v1__u1 = __float2half(((v0__u1) ? (__half2float(in0__u1)) : (__half2float(in1__u1))));
        __half in0__u2 = x[a0 * 3 + 2];
        __half in1__u2 = y[a0 * 3 + 2];
        bool v0__u2 = __half2float(in0__u2) > __half2float(in1__u2);
        __half v1__u2 = __float2half(((v0__u2) ? (__half2float(in0__u2)) : (__half2float(in1__u2))));
        where[a0 * 3] = v1__u0;
        where[a0 * 3 + 1] = v1__u1;
        where[a0 * 3 + 2] = v1__u2;
    }
}
