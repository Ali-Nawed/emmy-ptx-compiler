extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* x, const float* y, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1025) {
        int a0 = _gid;
        float in0 = x[a0];
        float in1 = y[a0];
        float v1 = in1 * fmaxf(0.0f, in0);
        mul[a0] = v1;
    }
}
