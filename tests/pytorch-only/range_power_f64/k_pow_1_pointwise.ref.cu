extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const double* x, const double* y, double* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        float in0 = x[a0];
        float in1 = y[a0];
        float v0 = powf(in0, in1);
        pow_1[a0] = v0;
    }
}
