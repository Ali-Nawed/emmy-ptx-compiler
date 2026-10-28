extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* y, const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 330) {
        int a0 = _gid / 165;
        int a1 = (_gid / 55) % 3;
        int a2 = (_gid / 11) % 5;
        int a3 = (_gid) % 11;
        float in1__u0 = x[a0 * 99 + a1 * 33 + a3 * 3];
        float in0__u0 = y[a1 * 5 + a2];
        float v3__u0 = fmaxf(0.0f, (in0__u0 + in1__u0)) * (in1__u0 - in0__u0);
        float in1__u1 = x[a0 * 99 + a1 * 33 + (a3 * 3 + 1)];
        float in0__u1 = y[a1 * 5 + a2];
        float v3__u1 = fmaxf(0.0f, (in0__u1 + in1__u1)) * (in1__u1 - in0__u1);
        float in1__u2 = x[a0 * 99 + a1 * 33 + (a3 * 3 + 2)];
        float in0__u2 = y[a1 * 5 + a2];
        float v3__u2 = fmaxf(0.0f, (in0__u2 + in1__u2)) * (in1__u2 - in0__u2);
        mul[a0 * 495 + a1 * 165 + a2 * 33 + a3 * 3] = v3__u0;
        mul[a0 * 495 + a1 * 165 + a2 * 33 + (a3 * 3 + 1)] = v3__u1;
        mul[a0 * 495 + a1 * 165 + a2 * 33 + (a3 * 3 + 2)] = v3__u2;
    }
}
