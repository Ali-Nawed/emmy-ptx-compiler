extern "C" __global__
__launch_bounds__(256) void k_add_3_pointwise(const double* mul_1_c1, const double* add_c1, const double* mul_3_c1, const double* add_2_c1, const double* x, const double* y, double* add_3) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in4__u0 = x[a0 * 3];
        float in0__u0 = add_2_c1[0];
        float in1__u0 = add_c1[0];
        float in2__u0 = mul_1_c1[0];
        float in5__u0 = y[a0 * 3];
        float in3__u0 = mul_3_c1[0];
        float v10__u0 = (log((in1__u0 + (in4__u0 * in4__u0))) + expf((in2__u0 * in5__u0))) + powf((in0__u0 + (in4__u0 * in4__u0)), (in3__u0 * in5__u0));
        float in4__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_2_c1[0];
        float in1__u1 = add_c1[0];
        float in2__u1 = mul_1_c1[0];
        float in5__u1 = y[a0 * 3 + 1];
        float in3__u1 = mul_3_c1[0];
        float v10__u1 = (log((in1__u1 + (in4__u1 * in4__u1))) + expf((in2__u1 * in5__u1))) + powf((in0__u1 + (in4__u1 * in4__u1)), (in3__u1 * in5__u1));
        float in4__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_2_c1[0];
        float in1__u2 = add_c1[0];
        float in2__u2 = mul_1_c1[0];
        float in5__u2 = y[a0 * 3 + 2];
        float in3__u2 = mul_3_c1[0];
        float v10__u2 = (log((in1__u2 + (in4__u2 * in4__u2))) + expf((in2__u2 * in5__u2))) + powf((in0__u2 + (in4__u2 * in4__u2)), (in3__u2 * in5__u2));
        add_3[a0 * 3] = v10__u0;
        add_3[a0 * 3 + 1] = v10__u1;
        add_3[a0 * 3 + 2] = v10__u2;
    }
}
