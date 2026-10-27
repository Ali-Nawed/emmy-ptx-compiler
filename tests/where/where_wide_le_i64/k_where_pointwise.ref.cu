extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const long long* add_c1, const long long* add_1_c1, const long long* x, const long long* y, long long* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 7) {
        int a0 = _gid;
        long long in1 = add_c1[0];
        long long in2 = x[a0];
        long long in0 = add_1_c1[0];
        long long in3 = y[a0];
        bool v2 = (in1 + in2) <= (in0 + in3);
        long long v3 = ((v2) ? (in2) : (in3));
        where[a0] = v3;
    }
}
