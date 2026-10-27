extern "C" __global__
__launch_bounds__(256) void k_mean_712baf(const double* mean_count, const double* x, double* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        float in0 = mean_count[0];
        float v0 = 1.0f / in0;
        float in1 = x[a0];
        float v1 = in1 * v0;
        float v2 = in1 - v1;
        sub[a0] = v2;
    }
}
