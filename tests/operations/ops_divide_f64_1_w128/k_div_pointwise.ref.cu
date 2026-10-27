extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const double* y, const double* x, const double* add_c1, double* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in2 = y[0];
        float in0 = add_c1[0];
        float in1 = x[0];
        float v3 = (in1 - in2) / (in0 + (in2 * in2));
        div[0] = v3;
    }
}
