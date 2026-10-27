extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* y, const float* x, float* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 3];
        float in1__u0 = y[a0 * 3];
        bool v0__u0 = in0__u0 > in1__u0;
        float v2__u0 = (((v0__u0 == 0)) ? (in0__u0) : (in1__u0));
        float in0__u1 = x[a0 * 3 + 1];
        float in1__u1 = y[a0 * 3 + 1];
        bool v0__u1 = in0__u1 > in1__u1;
        float v2__u1 = (((v0__u1 == 0)) ? (in0__u1) : (in1__u1));
        float in0__u2 = x[a0 * 3 + 2];
        float in1__u2 = y[a0 * 3 + 2];
        bool v0__u2 = in0__u2 > in1__u2;
        float v2__u2 = (((v0__u2 == 0)) ? (in0__u2) : (in1__u2));
        where[a0 * 3] = v2__u0;
        where[a0 * 3 + 1] = v2__u1;
        where[a0 * 3 + 2] = v2__u2;
    }
}
