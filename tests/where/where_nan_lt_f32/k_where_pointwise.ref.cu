extern "C" __global__
__launch_bounds__(256) void k_where_pointwise(const float* y, const float* x, float* where) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 5) {
        int a0 = _gid;
        float in0 = x[a0];
        float in1 = y[a0];
        bool v1 = (in0 / in0) < in1;
        float v3 = ((v1) ? (in1) : ((0.0f - in1)));
        where[a0] = v3;
    }
}
