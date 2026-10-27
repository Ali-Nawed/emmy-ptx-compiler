extern "C" __global__
__launch_bounds__(256) void k_where_1_pointwise(const float* gt_1_c1, const float* y, const float* x, float* where_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in0__u0 = gt_1_c1[0];
        float in1__u0 = x[a0 * 3];
        bool v0__u0 = in1__u0 > in0__u0;
        float in2__u0 = y[a0 * 3];
        bool v1__u0 = in1__u0 > in2__u0;
        float v2__u0 = ((v0__u0) ? (in1__u0) : (in2__u0));
        float v3__u0 = ((v1__u0) ? (v2__u0) : (in2__u0));
        float in0__u1 = gt_1_c1[0];
        float in1__u1 = x[a0 * 3 + 1];
        bool v0__u1 = in1__u1 > in0__u1;
        float in2__u1 = y[a0 * 3 + 1];
        bool v1__u1 = in1__u1 > in2__u1;
        float v2__u1 = ((v0__u1) ? (in1__u1) : (in2__u1));
        float v3__u1 = ((v1__u1) ? (v2__u1) : (in2__u1));
        float in0__u2 = gt_1_c1[0];
        float in1__u2 = x[a0 * 3 + 2];
        bool v0__u2 = in1__u2 > in0__u2;
        float in2__u2 = y[a0 * 3 + 2];
        bool v1__u2 = in1__u2 > in2__u2;
        float v2__u2 = ((v0__u2) ? (in1__u2) : (in2__u2));
        float v3__u2 = ((v1__u2) ? (v2__u2) : (in2__u2));
        where_1[a0 * 3] = v3__u0;
        where_1[a0 * 3 + 1] = v3__u1;
        where_1[a0 * 3 + 2] = v3__u2;
    }
}
