extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const double* x, double* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 3];
        float v0__u0 = tanhf(in0__u0);
        float in0__u1 = x[a0 * 3 + 1];
        float v0__u1 = tanhf(in0__u1);
        float in0__u2 = x[a0 * 3 + 2];
        float v0__u2 = tanhf(in0__u2);
        tanh[a0 * 3] = v0__u0;
        tanh[a0 * 3 + 1] = v0__u1;
        tanh[a0 * 3 + 2] = v0__u2;
    }
}
