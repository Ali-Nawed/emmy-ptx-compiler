extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const double* x, const double* add_c1, const double* div_c1, double* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in2 = x[0];
        float in0 = add_c1[0];
        float in1 = div_c1[0];
        float v3 = log((in0 + (in2 * in2))) / in1;
        div[0] = v3;
    }
}
