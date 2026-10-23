extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const int* x, const int* y, int* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 31) {
        int a0 = _gid;
        int in0 = x[a0];
        int in1 = y[a0];
        int v0 = in0 - in1;
        sub[a0] = v0;
    }
}
