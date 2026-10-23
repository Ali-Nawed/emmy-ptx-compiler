extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const int* x, const int* y, int* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        int in0 = x[0];
        int in1 = y[0];
        int v0 = in0 * in1;
        mul[0] = v0;
    }
}
