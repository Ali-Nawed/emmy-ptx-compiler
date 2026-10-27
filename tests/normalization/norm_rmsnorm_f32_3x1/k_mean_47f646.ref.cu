extern "C" __global__
__launch_bounds__(256) void k_mean_47f646(const float* mean_count, const float* add_c1, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        float in1 = mean_count[0];
        float in2 = x[a0];
        float in0 = add_c1[0];
        float v5 = in2 / sqrtf((in0 + ((1.0f / in1) * (in2 * in2))));
        div[a0] = v5;
    }
}
