extern "C" __global__
__launch_bounds__(256) void k_reciprocal_pointwise(const double* x, const double* add_c1, double* reciprocal) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in1 = x[0];
        float in0 = add_c1[0];
        float v2 = 1.0f / (in0 + (in1 * in1));
        reciprocal[0] = v2;
    }
}
