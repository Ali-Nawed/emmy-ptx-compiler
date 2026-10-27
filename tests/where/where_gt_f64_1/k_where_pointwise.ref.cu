extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const double* y, const double* x, double* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x[0];
        float in1 = y[0];
        bool v0 = in0 > in1;
        float v1 = ((v0) ? (in0) : (in1));
        where[0] = v1;
    }
}
