extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const double* y, const double* x, double* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 99) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 2];
        float in1__u0 = y[a0 * 2];
        float v3__u0 = (in0__u0 + in1__u0) * fmaxf(0.0f, (in0__u0 - in1__u0));
        float in0__u1 = x[a0 * 2 + 1];
        float in1__u1 = y[a0 * 2 + 1];
        float v3__u1 = (in0__u1 + in1__u1) * fmaxf(0.0f, (in0__u1 - in1__u1));
        mul[a0 * 2] = v3__u0;
        mul[a0 * 2 + 1] = v3__u1;
    }
}
