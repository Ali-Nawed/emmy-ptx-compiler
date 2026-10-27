extern "C" __global__
__launch_bounds__(256) void k_to_pointwise(const int* x, float* to) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        int in0__u0 = x[a0 * 3];
        int in0__u1 = x[a0 * 3 + 1];
        int in0__u2 = x[a0 * 3 + 2];
        to[a0 * 3] = in0__u0;
        to[a0 * 3 + 1] = in0__u1;
        to[a0 * 3 + 2] = in0__u2;
    }
}
