extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const double* x, const double* y, double* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 3];
        float in1__u0 = y[a0 * 3];
        float v2__u0 = in1__u0 + (0.0f - fabsf(in0__u0));
        float in0__u1 = x[a0 * 3 + 1];
        float in1__u1 = y[a0 * 3 + 1];
        float v2__u1 = in1__u1 + (0.0f - fabsf(in0__u1));
        float in0__u2 = x[a0 * 3 + 2];
        float in1__u2 = y[a0 * 3 + 2];
        float v2__u2 = in1__u2 + (0.0f - fabsf(in0__u2));
        add[a0 * 3] = v2__u0;
        add[a0 * 3 + 1] = v2__u1;
        add[a0 * 3 + 2] = v2__u2;
    }
}
