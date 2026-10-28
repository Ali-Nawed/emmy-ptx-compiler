extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const double* y, const double* x, double* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 129) {
        int a0 = _gid / 43;
        int a1 = (_gid) % 43;
        float in0__u0 = x[a0 * 129 + a1 * 3];
        float in1__u0 = y[a1 * 3];
        float v2__u0 = in1__u0 * fmaxf(0.0f, (in0__u0 + in1__u0));
        float in0__u1 = x[a0 * 129 + (a1 * 3 + 1)];
        float in1__u1 = y[a1 * 3 + 1];
        float v2__u1 = in1__u1 * fmaxf(0.0f, (in0__u1 + in1__u1));
        float in0__u2 = x[a0 * 129 + (a1 * 3 + 2)];
        float in1__u2 = y[a1 * 3 + 2];
        float v2__u2 = in1__u2 * fmaxf(0.0f, (in0__u2 + in1__u2));
        mul[a0 * 129 + a1 * 3] = v2__u0;
        mul[a0 * 129 + (a1 * 3 + 1)] = v2__u1;
        mul[a0 * 129 + (a1 * 3 + 2)] = v2__u2;
    }
}
