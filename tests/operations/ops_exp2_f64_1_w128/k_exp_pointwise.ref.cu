extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const double* x, const double* mul_c1, const double* mul_1_c1, double* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in1 = mul_c1[0];
        float in2 = x[0];
        float in0 = mul_1_c1[0];
        float v2 = expf((in0 * (in1 * in2)));
        exp[0] = v2;
    }
}
