extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const double* x, double* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 3];
        float v1__u0 = in0__u0 * (1.0f / (1.0f + expf(0.0f - in0__u0)));
        float in0__u1 = x[a0 * 3 + 1];
        float v1__u1 = in0__u1 * (1.0f / (1.0f + expf(0.0f - in0__u1)));
        float in0__u2 = x[a0 * 3 + 2];
        float v1__u2 = in0__u2 * (1.0f / (1.0f + expf(0.0f - in0__u2)));
        mul[a0 * 3] = v1__u0;
        mul[a0 * 3 + 1] = v1__u1;
        mul[a0 * 3 + 2] = v1__u2;
    }
}
