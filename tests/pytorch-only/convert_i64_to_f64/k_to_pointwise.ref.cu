extern "C" __global__
__launch_bounds__(256) void k_to_pointwise(const long long* x, double* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        long long in0__u0 = x[a0 * 3];
        float v0__u0 = in0__u0;
        long long in0__u1 = x[a0 * 3 + 1];
        float v0__u1 = in0__u1;
        long long in0__u2 = x[a0 * 3 + 2];
        float v0__u2 = in0__u2;
        to[a0 * 3] = v0__u0;
        to[a0 * 3 + 1] = v0__u1;
        to[a0 * 3 + 2] = v0__u2;
    }
}
