extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const long long* x, const long long* y, long long* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 64) {
        int a0 = _gid;
        long long in0__u0 = x[a0 * 2];
        long long in1__u0 = y[a0 * 2];
        long long v0__u0 = in0__u0 + in1__u0;
        long long in0__u1 = x[a0 * 2 + 1];
        long long in1__u1 = y[a0 * 2 + 1];
        long long v0__u1 = in0__u1 + in1__u1;
        add[a0 * 2] = v0__u0;
        add[a0 * 2 + 1] = v0__u1;
    }
}
