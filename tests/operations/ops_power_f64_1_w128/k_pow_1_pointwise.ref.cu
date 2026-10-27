extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const double* y, const double* mul_1_c1, const double* x, const double* add_c1, double* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in2 = x[0];
        float in0 = add_c1[0];
        float in1 = mul_1_c1[0];
        float in3 = y[0];
        float v3 = powf((in0 + (in2 * in2)), (in1 * in3));
        pow_1[0] = v3;
    }
}
