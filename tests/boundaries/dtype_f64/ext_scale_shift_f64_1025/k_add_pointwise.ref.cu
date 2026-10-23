extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const double* mul_c1, const double* add_c1, const double* x, double* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1025) {
        int a0 = _gid;
        float in1 = mul_c1[0];
        float in2 = x[a0];
        float in0 = add_c1[0];
        float v1 = in0 + (in1 * in2);
        add[a0] = v1;
    }
}
