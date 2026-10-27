extern "C" __global__
__launch_bounds__(256) void k_pow_1_pointwise(const int* y, const int* x, int* pow_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        int in1 = y[0];
        int in0 = x[0];
        int v1 = powf(in0, (in1 * in1));
        pow_1[0] = v1;
    }
}
