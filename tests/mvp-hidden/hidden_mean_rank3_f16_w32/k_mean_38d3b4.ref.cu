#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(32) void k_mean_38d3b4(const __half* mean_keepdim_count, const __half* x, __half* mean) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32;
        int a1_co = (_gid) % 32;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 129; a1 += 32) {
            __half in1 = x[a0 * 129 + a1];
            acc0 += __half2float(in1);
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
            __half in0 = mean_keepdim_count[0];
            float v1 = acc0 * __half2float((1.0f / in0));
            mean[a0] = __float2half(v1);
        }
}
