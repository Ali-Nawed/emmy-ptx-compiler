extern "C" __global__
__launch_bounds__(256) void k_and_1_pointwise(const long long* sub_c1, const long long* mul_c1, const long long* rshift_c1, const long long* and_1_c1, const long long* x, long long* and_1) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 43) {
        int a0 = _gid;
        long long in1__u0 = mul_c1[0];
        long long in4__u0 = x[a0 * 3];
        long long in3__u0 = sub_c1[0];
        long long in2__u0 = rshift_c1[0];
        long long in0__u0 = and_1_c1[0];
        long long v3__u0 = (((in1__u0 * in4__u0) - in3__u0) >> in2__u0) & in0__u0;
        long long in1__u1 = mul_c1[0];
        long long in4__u1 = x[a0 * 3 + 1];
        long long in3__u1 = sub_c1[0];
        long long in2__u1 = rshift_c1[0];
        long long in0__u1 = and_1_c1[0];
        long long v3__u1 = (((in1__u1 * in4__u1) - in3__u1) >> in2__u1) & in0__u1;
        long long in1__u2 = mul_c1[0];
        long long in4__u2 = x[a0 * 3 + 2];
        long long in3__u2 = sub_c1[0];
        long long in2__u2 = rshift_c1[0];
        long long in0__u2 = and_1_c1[0];
        long long v3__u2 = (((in1__u2 * in4__u2) - in3__u2) >> in2__u2) & in0__u2;
        and_1[a0 * 3] = v3__u0;
        and_1[a0 * 3 + 1] = v3__u1;
        and_1[a0 * 3 + 2] = v3__u2;
    }
}
