#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(32) void k_sum_1_reduce(const __half* y, const __half* x, __half* sum_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32;
        int a1_co = (_gid) % 32;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 129; a1 += 32) {
            __half in0 = x[a0 * 129 + a1];
            __half in1 = y[a0 * 129 + a1];
            bool v1 = __half2float(in0) > __half2float(in1);
            __half v3 = __float2half(((v1) ? (__half2float((in0 - in1))) : (__half2float((in0 + in1)))));
            acc0 += __half2float(v3);
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }
        if (a1_co == 0) {
            __half v4 = __float2half(acc0);
            sum_1[a0] = v4;
        }
}
