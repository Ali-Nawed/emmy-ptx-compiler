extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const int* y, const int* x, long long* add) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        int in0 = x[a0];
        int in1 = y[a0];
        int v0 = in0 + in1;
        add[a0] = v0;
    }
}
