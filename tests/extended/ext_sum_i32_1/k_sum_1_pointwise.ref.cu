extern "C" __global__
__launch_bounds__(256) void k_sum_1_pointwise(const int* x, long long* sum_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        int in0 = x[0];
        sum_1[0] = in0;
    }
}
