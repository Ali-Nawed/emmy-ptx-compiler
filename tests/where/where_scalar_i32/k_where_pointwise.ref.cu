extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* gt_c1, const int* x, const int* y, int* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in0__u0 = gt_c1[0];
        int in1__u0 = x[a0 * 3];
        bool v0__u0 = in1__u0 > in0__u0;
        int in2__u0 = y[a0 * 3];
        int v1__u0 = ((v0__u0) ? (in1__u0) : (in2__u0));
        float in0__u1 = gt_c1[0];
        int in1__u1 = x[a0 * 3 + 1];
        bool v0__u1 = in1__u1 > in0__u1;
        int in2__u1 = y[a0 * 3 + 1];
        int v1__u1 = ((v0__u1) ? (in1__u1) : (in2__u1));
        float in0__u2 = gt_c1[0];
        int in1__u2 = x[a0 * 3 + 2];
        bool v0__u2 = in1__u2 > in0__u2;
        int in2__u2 = y[a0 * 3 + 2];
        int v1__u2 = ((v0__u2) ? (in1__u2) : (in2__u2));
        where[a0 * 3] = v1__u0;
        where[a0 * 3 + 1] = v1__u1;
        where[a0 * 3 + 2] = v1__u2;
    }
}
