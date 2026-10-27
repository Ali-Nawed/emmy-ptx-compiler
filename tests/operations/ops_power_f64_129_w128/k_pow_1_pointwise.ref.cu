extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const double* mul_1_c1, const double* add_c1, const double* x, const double* y, double* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in2__u0 = x[a0 * 3];
        float in0__u0 = add_c1[0];
        float in1__u0 = mul_1_c1[0];
        float in3__u0 = y[a0 * 3];
        float v3__u0 = powf((in0__u0 + (in2__u0 * in2__u0)), (in1__u0 * in3__u0));
        float in2__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float in1__u1 = mul_1_c1[0];
        float in3__u1 = y[a0 * 3 + 1];
        float v3__u1 = powf((in0__u1 + (in2__u1 * in2__u1)), (in1__u1 * in3__u1));
        float in2__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float in1__u2 = mul_1_c1[0];
        float in3__u2 = y[a0 * 3 + 2];
        float v3__u2 = powf((in0__u2 + (in2__u2 * in2__u2)), (in1__u2 * in3__u2));
        pow_1[a0 * 3] = v3__u0;
        pow_1[a0 * 3 + 1] = v3__u1;
        pow_1[a0 * 3 + 2] = v3__u2;
    }
}
