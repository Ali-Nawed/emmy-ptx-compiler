extern "C" __global__
__launch_bounds__(256) void k_sum_1_pointwise(const long long* x, long long* sum_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        long long in0 = x[a0];
        sum_1[a0] = in0;
    }
}
