extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* div_c1, const float* x, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        float in1__u0 = div_c1[0];
        float in2__u0 = x[a0 * 3];
        float in0__u0 = add_c1[0];
        float v4__u0 = (1.0f / in1__u0) * log((in0__u0 + (in2__u0 * in2__u0)));
        float in1__u1 = div_c1[0];
        float in2__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float v4__u1 = (1.0f / in1__u1) * log((in0__u1 + (in2__u1 * in2__u1)));
        float in1__u2 = div_c1[0];
        float in2__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float v4__u2 = (1.0f / in1__u2) * log((in0__u2 + (in2__u2 * in2__u2)));
        div[a0 * 3] = v4__u0;
        div[a0 * 3 + 1] = v4__u1;
        div[a0 * 3 + 2] = v4__u2;
    }
}
