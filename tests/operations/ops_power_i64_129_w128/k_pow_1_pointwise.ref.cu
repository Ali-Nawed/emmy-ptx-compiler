extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const long long* y, const long long* x, long long* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        long long in1__u0 = y[a0 * 3];
        long long in0__u0 = x[a0 * 3];
        long long v1__u0 = powf(in0__u0, (in1__u0 * in1__u0));
        long long in1__u1 = y[a0 * 3 + 1];
        long long in0__u1 = x[a0 * 3 + 1];
        long long v1__u1 = powf(in0__u1, (in1__u1 * in1__u1));
        long long in1__u2 = y[a0 * 3 + 2];
        long long in0__u2 = x[a0 * 3 + 2];
        long long v1__u2 = powf(in0__u2, (in1__u2 * in1__u2));
        pow_1[a0 * 3] = v1__u0;
        pow_1[a0 * 3 + 1] = v1__u1;
        pow_1[a0 * 3 + 2] = v1__u2;
    }
}
