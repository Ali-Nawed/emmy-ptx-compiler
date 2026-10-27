extern "C" __global__
__launch_bounds__(256) void k_exp_pointwise(const float* mul_c1, const float* mul_1_c1, const float* x, float* exp) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        float in1__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 3];
        float in0__u0 = mul_1_c1[0];
        float v2__u0 = expf((in0__u0 * (in1__u0 * in2__u0)));
        float in1__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 3 + 1];
        float in0__u1 = mul_1_c1[0];
        float v2__u1 = expf((in0__u1 * (in1__u1 * in2__u1)));
        float in1__u2 = mul_c1[0];
        float in2__u2 = x[a0 * 3 + 2];
        float in0__u2 = mul_1_c1[0];
        float v2__u2 = expf((in0__u2 * (in1__u2 * in2__u2)));
        exp[a0 * 3] = v2__u0;
        exp[a0 * 3 + 1] = v2__u1;
        exp[a0 * 3 + 2] = v2__u2;
    }
}
