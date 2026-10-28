#include <cuda_fp16.h>
extern "C" __global__
__launch_bounds__(256) void k_matmul_7421ac(const __half* b, const __half* a, __half* matmul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 128) {
        int a0 = _gid / 8;
        int a1 = (_gid) % 8;
        float acc0 = 0.0f;
        for (int a2 = 0; a2 < 16; a2++) {
            __half in0 = a[a0 * 16 + a2];
            __half in1 = b[a2 * 8 + a1];
            __half v0 = in0 * in1;
            acc0 += __half2float(v0);
        }
        matmul[a0 * 8 + a1] = __float2half(acc0);
    }
}
