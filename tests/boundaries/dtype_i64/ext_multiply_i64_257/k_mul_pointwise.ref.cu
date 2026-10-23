extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const long long* x, const long long* y, long long* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 257) {
        int a0 = _gid;
        long long in0 = x[a0];
        long long in1 = y[a0];
        long long v0 = in0 * in1;
        mul[a0] = v0;
    }
}
