extern "C" __global__
__launch_bounds__(32) void k_mean_425dd1(const float* mean_count, const float* x, float* sub) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 32;
        int a1_co = (_gid) % 32;
        float acc0 = 0.0f;
        for (int a1 = a1_co; a1 < 31; a1 += 32) {
            float in1 = x[a0 * 31 + a1];
            acc0 += in1;
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
        float in0 = mean_count[0];
        float v1 = acc0 * (1.0f / in0);
        for (int a2 = a1_co; a2 < 31; a2 += 32) {
            float in2 = x[a0 * 31 + a2];
            float v2 = in2 - v1;
            sub[a0 * 31 + a2] = v2;
        }
}
