extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const int* x, const int* y, int* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        int in0 = x[0];
        int in1 = y[0];
        int v2 = in1 + (0.0f - fabsf(in0));
        add[0] = v2;
    }
}
