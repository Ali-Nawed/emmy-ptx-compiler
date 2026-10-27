extern "C" __global__
__launch_bounds__(128) void k_mean_reduce_cb35f5(const float* mean_1_keepdim_count, const float* mean_keepdim_count, const float* x, float* sub) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 128;
        int a1_co = (_gid) % 128;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int a1 = a1_co; a1 < 129; a1 += 128) {
            float in2 = x[a0 * 129 + a1];
            float v2 = in2 * in2;
            acc0 += in2;
            acc1 += v2;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 16);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 8);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            acc0 = acc0 + acc0__o;
            acc1 = acc1 + acc1__o;
        }
        __shared__ float acc0_smem[4];
        __shared__ float acc1_smem[4];
        if (lane == 0) {
            acc0_smem[warp] = acc0;
            acc1_smem[warp] = acc1;
        }
        __syncthreads();
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                float acc1 = acc1_smem[warp];
                float acc1__o = acc1_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc1 = acc1 + acc1__o;
                acc0_smem[warp] = acc0;
                acc1_smem[warp] = acc1;
            }
            __syncthreads();
        }
        acc0 = acc0_smem[0];
        acc1 = acc1_smem[0];
        if (a1_co == 0) {
            float in0 = mean_1_keepdim_count[0];
            float v4 = acc0 * (1.0f / in0);
            float in1 = mean_keepdim_count[0];
            float v6 = (acc1 * (1.0f / in1)) - (v4 * v4);
            sub[a0] = v6;
        }
}
