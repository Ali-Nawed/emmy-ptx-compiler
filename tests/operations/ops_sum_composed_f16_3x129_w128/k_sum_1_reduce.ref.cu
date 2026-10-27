#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(128) void k_sum_1_reduce(const __half* add_c1, const __half* y, const __half* x, __half* sum_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        __half in0 = add_c1[0];
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 129; a1 += 128) {
            __half in2 = y[a0 * 129 + a1];
            float v0 = __half2float(in2) * __half2float(in2);
            float v1 = __half2float(in0) + v0;
            __half in1 = x[a0 * 129 + a1];
            float v3 = __half2float((in1 - in2)) / v1;
            acc0 += v3;
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
        __shared__ float acc0_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        if (a1_co == 0) {
            __half v4 = __float2half(acc0);
            sum_1[a0] = v4;
        }
}
