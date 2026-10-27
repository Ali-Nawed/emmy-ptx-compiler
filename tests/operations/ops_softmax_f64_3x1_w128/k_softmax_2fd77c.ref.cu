extern "C" __global__
__launch_bounds__(256) void k_softmax_2fd77c(const double* x, double* softmax) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 3) {
        int a0 = _gid;
        float in0 = x[a0];
        float v1 = expf((in0 - in0));
        float v2 = v1 / v1;
        softmax[a0] = v2;
    }
}
