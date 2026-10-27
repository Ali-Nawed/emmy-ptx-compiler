extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* y, const float* x, float* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 129) {
        int a0 = _gid / 43;
        int a1 = (_gid) % 43;
        float in1__u0 = x[a0 * 129 + a1 * 3];
        float in0__u0 = y[a0];
        bool v0__u0 = in1__u0 > in0__u0;
        float v1__u0 = ((v0__u0) ? (in1__u0) : (in0__u0));
        float in1__u1 = x[a0 * 129 + (a1 * 3 + 1)];
        float in0__u1 = y[a0];
        bool v0__u1 = in1__u1 > in0__u1;
        float v1__u1 = ((v0__u1) ? (in1__u1) : (in0__u1));
        float in1__u2 = x[a0 * 129 + (a1 * 3 + 2)];
        float in0__u2 = y[a0];
        bool v0__u2 = in1__u2 > in0__u2;
        float v1__u2 = ((v0__u2) ? (in1__u2) : (in0__u2));
        where[a0 * 129 + a1 * 3] = v1__u0;
        where[a0 * 129 + (a1 * 3 + 1)] = v1__u1;
        where[a0 * 129 + (a1 * 3 + 2)] = v1__u2;
    }
}
