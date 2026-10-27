extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const long long* y, const long long* x, long long* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        long long in0 = x[0];
        long long in1 = y[0];
        bool v0 = in0 > in1;
        long long v1 = ((v0) ? (in0) : (in1));
        where[0] = v1;
    }
}
