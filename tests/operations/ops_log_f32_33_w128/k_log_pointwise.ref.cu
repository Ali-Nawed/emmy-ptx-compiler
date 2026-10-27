extern "C" __global__
__launch_bounds__(256) void k_log_pointwise(const float* add_c1, const float* x, float* log) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 11) {
        int a0 = _gid;
        float in1__u0 = x[a0 * 3];
        float in0__u0 = add_c1[0];
        float v2__u0 = log((in0__u0 + (in1__u0 * in1__u0)));
        float in1__u1 = x[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float v2__u1 = log((in0__u1 + (in1__u1 * in1__u1)));
        float in1__u2 = x[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float v2__u2 = log((in0__u2 + (in1__u2 * in1__u2)));
        log[a0 * 3] = v2__u0;
        log[a0 * 3 + 1] = v2__u1;
        log[a0 * 3 + 2] = v2__u2;
    }
}
