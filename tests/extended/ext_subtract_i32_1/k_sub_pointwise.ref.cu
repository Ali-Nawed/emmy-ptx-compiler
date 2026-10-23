extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const int* x, const int* y, int* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        int in0 = x[0];
        int in1 = y[0];
        int v0 = in0 - in1;
        sub[0] = v0;
    }
}
