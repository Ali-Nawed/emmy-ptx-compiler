extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const int* x, const int* y, int* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        int in0 = x[a0];
        int in1 = y[a0];
        int v1 = in1 * fmaxf(0.0f, in0);
        mul[a0] = v1;
    }
}
