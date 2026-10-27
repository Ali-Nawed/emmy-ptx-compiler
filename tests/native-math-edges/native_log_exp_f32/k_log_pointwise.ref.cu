extern "C" __global__
__launch_bounds__(256) void k_log_pointwise(const float* x, float* log) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        float in0 = x[a0];
        float v1 = log(expf(in0));
        log[a0] = v1;
    }
}
