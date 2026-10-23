extern "C" __global__
__launch_bounds__(256) void k_mean_3a1b36(const int* x, const float* mean_keepdim_count, float* mean) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = mean_keepdim_count[0];
        int in1 = x[0];
        float v0 = in1 / in0;
        mean[0] = v0;
    }
}
