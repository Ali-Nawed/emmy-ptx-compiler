extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const double* y, const double* x, double* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 99) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 2];
        float in1__u0 = y[a0 * 2];
        bool v1__u0 = in0__u0 > in1__u0;
        float v3__u0 = ((v1__u0) ? ((in0__u0 - in1__u0)) : ((in0__u0 + in1__u0)));
        float in0__u1 = x[a0 * 2 + 1];
        float in1__u1 = y[a0 * 2 + 1];
        bool v1__u1 = in0__u1 > in1__u1;
        float v3__u1 = ((v1__u1) ? ((in0__u1 - in1__u1)) : ((in0__u1 + in1__u1)));
        where[a0 * 2] = v3__u0;
        where[a0 * 2 + 1] = v3__u1;
    }
}
