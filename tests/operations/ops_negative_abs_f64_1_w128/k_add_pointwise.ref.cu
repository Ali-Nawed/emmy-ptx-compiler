extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const double* x, const double* y, double* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x[0];
        float in1 = y[0];
        float v2 = in1 + (0.0f - fabsf(in0));
        add[0] = v2;
    }
}
