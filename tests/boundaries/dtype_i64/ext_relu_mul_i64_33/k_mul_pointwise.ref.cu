extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const long long* x, const long long* y, long long* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        long long in0__u0 = x[a0 * 3];
        long long in1__u0 = y[a0 * 3];
        long long v1__u0 = in1__u0 * fmaxf(0.0f, in0__u0);
        long long in0__u1 = x[a0 * 3 + 1];
        long long in1__u1 = y[a0 * 3 + 1];
        long long v1__u1 = in1__u1 * fmaxf(0.0f, in0__u1);
        long long in0__u2 = x[a0 * 3 + 2];
        long long in1__u2 = y[a0 * 3 + 2];
        long long v1__u2 = in1__u2 * fmaxf(0.0f, in0__u2);
        mul[a0 * 3] = v1__u0;
        mul[a0 * 3 + 1] = v1__u1;
        mul[a0 * 3 + 2] = v1__u2;
    }
}
