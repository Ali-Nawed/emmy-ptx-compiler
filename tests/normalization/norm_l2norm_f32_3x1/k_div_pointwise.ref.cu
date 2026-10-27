extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        float in0 = x[a0];
        float v2 = in0 / sqrtf((in0 * in0));
        div[a0] = v2;
    }
}
