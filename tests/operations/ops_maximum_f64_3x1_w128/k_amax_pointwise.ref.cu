extern "C" __global__
__launch_bounds__(256) void k_amax_pointwise(const double* x, double* amax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        float in0 = x[a0];
        amax[a0] = in0;
    }
}
