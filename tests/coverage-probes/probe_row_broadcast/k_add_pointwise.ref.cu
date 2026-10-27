extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* w, const float* x, float* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 129) {
        int a0 = _gid / 43;
        int a1 = (_gid) % 43;
        float in1__u0 = w[a1 * 3];
        float in0__u0 = x[a0 * 129 + a1 * 3];
        float v0__u0 = in0__u0 + in1__u0;
        float in1__u1 = w[a1 * 3 + 1];
        float in0__u1 = x[a0 * 129 + (a1 * 3 + 1)];
        float v0__u1 = in0__u1 + in1__u1;
        float in1__u2 = w[a1 * 3 + 2];
        float in0__u2 = x[a0 * 129 + (a1 * 3 + 2)];
        float v0__u2 = in0__u2 + in1__u2;
        add[a0 * 129 + a1 * 3] = v0__u0;
        add[a0 * 129 + (a1 * 3 + 1)] = v0__u1;
        add[a0 * 129 + (a1 * 3 + 2)] = v0__u2;
    }
}
