extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const float* x, float* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x[0];
        float v0 = tanhf(in0);
        tanh[0] = v0;
    }
}
