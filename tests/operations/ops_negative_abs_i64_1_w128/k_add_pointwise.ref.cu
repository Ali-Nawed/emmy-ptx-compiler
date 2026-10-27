extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const long long* x, const long long* y, long long* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        long long in0 = x[0];
        long long in1 = y[0];
        long long v2 = in1 + (0.0f - fabsf(in0));
        add[0] = v2;
    }
}
