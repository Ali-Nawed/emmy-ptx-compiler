extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const long long* y, const long long* x, long long* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 99) {
        int a0 = _gid;
        long long in0__u0 = x[a0 * 2];
        long long in1__u0 = y[a0 * 2];
        long long v3__u0 = (in0__u0 + in1__u0) * fmaxf(0.0f, (in0__u0 - in1__u0));
        long long in0__u1 = x[a0 * 2 + 1];
        long long in1__u1 = y[a0 * 2 + 1];
        long long v3__u1 = (in0__u1 + in1__u1) * fmaxf(0.0f, (in0__u1 - in1__u1));
        mul[a0 * 2] = v3__u0;
        mul[a0 * 2 + 1] = v3__u1;
    }
}
