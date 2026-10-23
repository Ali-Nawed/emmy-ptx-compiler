extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const long long* x, const long long* mul_c1, const long long* sub_c1, long long* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 1) {
        long long in0 = mul_c1[0];
        long long in2 = x[0];
        long long in1 = sub_c1[0];
        long long v1 = (in0 * in2) - in1;
        sub[0] = v1;
    }
}
