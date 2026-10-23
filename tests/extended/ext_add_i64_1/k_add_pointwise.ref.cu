extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const long long* x, const long long* y, long long* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        long long in0 = x[0];
        long long in1 = y[0];
        long long v0 = in0 + in1;
        add[0] = v0;
    }
}
