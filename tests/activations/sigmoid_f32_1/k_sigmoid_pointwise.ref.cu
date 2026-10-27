extern "C" __global__
__launch_bounds__(256) void k_sigmoid_pointwise(const float* x, float* sigmoid) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x[0];
        float v0 = 1.0f / (1.0f + expf(0.0f - in0));
        sigmoid[0] = v0;
    }
}
