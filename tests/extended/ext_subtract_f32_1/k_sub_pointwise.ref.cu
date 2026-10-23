extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* x, const float* y, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x[0];
        float in1 = y[0];
        float v0 = in0 - in1;
        sub[0] = v0;
    }
}
