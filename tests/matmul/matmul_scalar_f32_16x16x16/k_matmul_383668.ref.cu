extern "C" __global__
__launch_bounds__(256) void k_matmul_383668(const float* b, const float* a, float* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 16;
        int a1 = (_gid) % 16;
        float acc0 = 0.0f;
        for (int a2 = 0; a2 < 16; a2++) {
            float in0 = a[a0 * 16 + a2];
            float in1 = b[a2 * 16 + a1];
            float v0 = in0 * in1;
            acc0 += v0;
        }
        matmul[a0 * 16 + a1] = acc0;
}
