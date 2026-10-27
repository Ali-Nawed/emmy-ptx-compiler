extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const float* x, float* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 67) {
        int a0 = _gid;
        float in0 = x[a0];
        float v0 = expf(in0);
        exp[a0] = v0;
    }
}
