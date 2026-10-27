extern "C" __global__
__launch_bounds__(256) void k_sqrt_pointwise(const float* x, const float* add_c1, float* sqrt) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in1 = x[0];
        float in0 = add_c1[0];
        float v2 = sqrtf((in0 + (in1 * in1)));
        sqrt[0] = v2;
    }
}
