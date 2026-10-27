extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const double* y, const double* x, double* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        float in0 = x[0];
        float in1 = y[0];
        float v2 = fmaxf(in0, in1) - fminf(in0, in1);
        sub[0] = v2;
    }
}
