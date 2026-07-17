#include "xR64ABS.cuh"

// AutoGPTQ src [E,K/8,O], uint4b8 -> PTX A-native dst [E,O,K/8], s4.
// mma.m8n8k32 A lane lp owns K[8lp:8lp+8], low nibble first.
__global__ void xR64ABSRepackGPTQ(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    int E,
    int K,
    int O) {

    uint64_t n = (uint64_t)E * ((uint32_t)K >> 3) * (uint32_t)O;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t out = idx % (uint32_t)O;
    uint64_t t = idx / (uint32_t)O;
    uint32_t packet = t % ((uint32_t)K >> 3);
    uint32_t e = t / ((uint32_t)K >> 3);
    dst[((uint64_t)e * O + out) * ((uint32_t)K >> 3) + packet]
        = xr64_ld_ca_nc_u32(
            src + ((uint64_t)e * ((uint32_t)K >> 3) + packet) * O + out)
        ^ 0x88888888u;
}

void launch_xr64_abs_repack_gptq(
    const uint32_t* src,
    uint32_t* dst,
    int E,
    int K,
    int O,
    cudaStream_t stream) {

    uint64_t n = (uint64_t)E * ((uint32_t)K >> 3) * (uint32_t)O;
    xR64ABSRepackGPTQ<<<(n + 255u) / 256u, 256, 0, stream>>>(
        src, dst, E, K, O);
}
