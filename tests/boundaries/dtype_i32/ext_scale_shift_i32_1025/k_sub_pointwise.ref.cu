extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const int* sub_c1, const int* mul_c1, const int* x, int* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1025) {
        int a0 = _gid;
        int in0 = mul_c1[0];
        int in2 = x[a0];
        int in1 = sub_c1[0];
        int v1 = (in0 * in2) - in1;
        sub[a0] = v1;
    }
}
