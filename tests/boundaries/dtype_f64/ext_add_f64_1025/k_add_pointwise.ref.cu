extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const double* x, const double* y, double* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1025) {
        int a0 = _gid;
        float in0 = x[a0];
        float in1 = y[a0];
        float v0 = in0 + in1;
        add[a0] = v0;
    }
}
