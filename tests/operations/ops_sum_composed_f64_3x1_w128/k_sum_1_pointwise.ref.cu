extern "C" __global__
__launch_bounds__(256) void k_sum_1_pointwise(const double* add_c1, const double* y, const double* x, double* sum_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        float in2 = y[a0];
        float in0 = add_c1[0];
        float in1 = x[a0];
        float v3 = (in1 - in2) / (in0 + (in2 * in2));
        sum_1[a0] = v3;
    }
}
