extern "C" __global__
__launch_bounds__(256) void k_reciprocal_pointwise(const double* add_c1, const double* x, double* reciprocal) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in1__u0 = x[a0 * 3];
        float in0__u0 = add_c1[0];
        float v2__u0 = 1.0f / (in0__u0 + (in1__u0 * in1__u0));
        float in1__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float v2__u1 = 1.0f / (in0__u1 + (in1__u1 * in1__u1));
        float in1__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float v2__u2 = 1.0f / (in0__u2 + (in1__u2 * in1__u2));
        reciprocal[a0 * 3] = v2__u0;
        reciprocal[a0 * 3 + 1] = v2__u1;
        reciprocal[a0 * 3 + 2] = v2__u2;
    }
}
