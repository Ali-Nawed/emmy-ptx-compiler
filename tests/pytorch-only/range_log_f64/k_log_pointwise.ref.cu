extern "C" __global__
__launch_bounds__(256) void k_log_pointwise(const double* x, double* log) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        float in0 = x[a0];
        float v0 = log(in0);
        log[a0] = v0;
    }
}
