#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(32) void k_slice_reduce_3f2df5(const __half* x, __half* add) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_co = _gid;
        float acc0 = 0.0f;
        for (int a0 = a0_co; a0 < 128; a0 += 32) {
            __half in0 = x[a0 + 1];
            acc0 += __half2float(in0);
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
        __half v0 = __float2half(acc0);
        for (int a1 = a0_co; a1 < 129; a1 += 32) {
            __half in1 = x[a1];
            __half v1 = in1 + v0;
            add[a1] = v1;
        }
}
