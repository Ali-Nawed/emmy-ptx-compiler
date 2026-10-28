extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const long long* y, const int* x, long long* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 33) {
        int a0 = _gid / 11;
        int a1 = (_gid) % 11;
        int in0__u0 = x[a0 * 33 + a1 * 3];
        long long in1__u0 = y[a1 * 3];
        long long v0__u0 = in0__u0 + in1__u0;
        long long v1__u0 = in0__u0 - in1__u0;
        long long v2__u0 = v0__u0 * v1__u0;
        int in0__u1 = x[a0 * 33 + (a1 * 3 + 1)];
        long long in1__u1 = y[a1 * 3 + 1];
        long long v0__u1 = in0__u1 + in1__u1;
        long long v1__u1 = in0__u1 - in1__u1;
        long long v2__u1 = v0__u1 * v1__u1;
        int in0__u2 = x[a0 * 33 + (a1 * 3 + 2)];
        long long in1__u2 = y[a1 * 3 + 2];
        long long v0__u2 = in0__u2 + in1__u2;
        long long v1__u2 = in0__u2 - in1__u2;
        long long v2__u2 = v0__u2 * v1__u2;
        mul[a0 * 33 + a1 * 3] = v2__u0;
        mul[a0 * 33 + (a1 * 3 + 1)] = v2__u1;
        mul[a0 * 33 + (a1 * 3 + 2)] = v2__u2;
    }
}
