extern "C" __global__
__launch_bounds__(256) void k_mean_731e5e(const float* x, const float* mean_keepdim_count, float* mean) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = mean_keepdim_count[0];
        float in1 = x[0];
        float v0 = in1 / in0;
        mean[0] = v0;
    }
}
