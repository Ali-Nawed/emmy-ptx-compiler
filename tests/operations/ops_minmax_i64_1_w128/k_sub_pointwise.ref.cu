extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const long long* y, const long long* x, long long* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        long long in0 = x[0];
        long long in1 = y[0];
        long long v2 = fmaxf(in0, in1) - fminf(in0, in1);
        sub[0] = v2;
    }
}
