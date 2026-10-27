#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_tanh_pointwise(const __half* x, __half* tanh) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 13) {
        int a0 = _gid;
        __half2 _v_in0__u0 = *reinterpret_cast<const __half2*>(&x[a0 * 2]);
        __half in0__u0 = _v_in0__u0.x;
        __half in0__u1 = _v_in0__u0.y;
        __half v0__u0 = htanh(in0__u0);
        __half v0__u1 = htanh(in0__u1);
        __half2 _vs_v0__u0_81888f49 = __halves2half2(v0__u0, v0__u1);
        *reinterpret_cast<__half2*>(&tanh[a0 * 2]) = _vs_v0__u0_81888f49;
    }
}
