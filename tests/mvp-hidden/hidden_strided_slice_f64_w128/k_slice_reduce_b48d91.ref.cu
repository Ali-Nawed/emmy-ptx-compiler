extern "C" __global__
__launch_bounds__(128) void k_slice_reduce_b48d91(const double* x, double* add) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_co = _gid;
        float acc0 = 0.0f;
        for (int a0 = a0_co; a0 < 128; a0 += 128) {
            float in0 = x[a0 + 1];
            acc0 += in0;
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
        float v0 = acc0;
        for (int a1 = a0_co; a1 < 129; a1 += 128) {
            float in1 = x[a1];
            float v1 = in1 + v0;
            add[a1] = v1;
        }
}
