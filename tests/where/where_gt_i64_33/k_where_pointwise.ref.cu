extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const long long* y, const long long* x, long long* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        long long in0__u0 = x[a0 * 3];
        long long in1__u0 = y[a0 * 3];
        bool v0__u0 = in0__u0 > in1__u0;
        long long v1__u0 = ((v0__u0) ? (in0__u0) : (in1__u0));
        long long in0__u1 = x[a0 * 3 + 1];
        long long in1__u1 = y[a0 * 3 + 1];
        bool v0__u1 = in0__u1 > in1__u1;
        long long v1__u1 = ((v0__u1) ? (in0__u1) : (in1__u1));
        long long in0__u2 = x[a0 * 3 + 2];
        long long in1__u2 = y[a0 * 3 + 2];
        bool v0__u2 = in0__u2 > in1__u2;
        long long v1__u2 = ((v0__u2) ? (in0__u2) : (in1__u2));
        where[a0 * 3] = v1__u0;
        where[a0 * 3 + 1] = v1__u1;
        where[a0 * 3 + 2] = v1__u2;
    }
}
