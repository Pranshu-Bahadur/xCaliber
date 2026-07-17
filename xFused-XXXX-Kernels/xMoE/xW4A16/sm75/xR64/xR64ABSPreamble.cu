#include "xR64ABS.cuh"

// warp = 8 source tokens x K32
// lp0..3 = 8 FP16 -> one packed INT4 u32 per token
__global__ __launch_bounds__(XR64_ABS_CTA, 1)
void xR64ABSRouteQuantizeScatter(
    const uint16_t* __restrict__ X,
    const int32_t* __restrict__ topk_idx,
    const int32_t* __restrict__ topk_off,
    uint32_t* __restrict__ X4,
    float* __restrict__ SX,
    int N,
    int NS,
    int H,
    int TOPK) {

    uint32_t lane = threadIdx.x & 31u;
    uint64_t tile = (uint64_t)blockIdx.x * XR64_ABS_WARPS
                  + ((uint32_t)threadIdx.x >> 5);
    uint32_t H32 = (uint32_t)H >> 5;
    if (tile >= (uint64_t)(((uint32_t)N + 7u) >> 3) * H32) return;

    uint32_t kg = tile % H32;
    uint32_t token = ((uint32_t)(tile / H32) << 3) + (lane >> 2);
    uint32_t lp = lane & 3u;
    uint32_t r0 = 0u, r1 = 0u, r2 = 0u, r3 = 0u;

    if (token < (uint32_t)N)
        xr64_ld_ca_nc_v4(
            X + (uint64_t)token * (uint32_t)H + (kg << 5) + (lp << 3),
            r0, r1, r2, r3);

    float x0 = xr64_bf16((uint16_t)r0), x1 = xr64_bf16((uint16_t)(r0 >> 16));
    float x2 = xr64_bf16((uint16_t)r1), x3 = xr64_bf16((uint16_t)(r1 >> 16));
    float x4 = xr64_bf16((uint16_t)r2), x5 = xr64_bf16((uint16_t)(r2 >> 16));
    float x6 = xr64_bf16((uint16_t)r3), x7 = xr64_bf16((uint16_t)(r3 >> 16));

    float m = fmaxf(fmaxf(fabsf(x0), fabsf(x1)),
                    fmaxf(fabsf(x2), fabsf(x3)));
    m = fmaxf(m, fmaxf(fmaxf(fabsf(x4), fabsf(x5)),
                       fmaxf(fabsf(x6), fabsf(x7))));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1, 4));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2, 4));

    float inv = m > 0.0f ? 7.0f / m : 0.0f;
    uint32_t packed = xr64_q4(x0, inv) | (xr64_q4(x1, inv) << 4)
                    | (xr64_q4(x2, inv) << 8) | (xr64_q4(x3, inv) << 12)
                    | (xr64_q4(x4, inv) << 16) | (xr64_q4(x5, inv) << 20)
                    | (xr64_q4(x6, inv) << 24) | (xr64_q4(x7, inv) << 28);

    for (uint32_t k = 0; k < (uint32_t)TOPK; k++) {
        uint32_t e = 0u, p = 0u;
        if (token < (uint32_t)N && !lp) {
            e = xr64_ld_ca_nc_u32(topk_idx + (uint64_t)token * TOPK + k);
            p = xr64_ld_ca_nc_u32(topk_off + (uint64_t)token * TOPK + k);
        }
        e = __shfl_sync(0xffffffffu, e, 0, 4);
        p = __shfl_sync(0xffffffffu, p, 0, 4);
        if (token < (uint32_t)N) {
            X4[((uint64_t)e * NS + p) * ((uint32_t)H >> 3)
                + kg * 4u + lp] = packed;
            if (!lp)
                SX[((uint64_t)e * NS + p) * H32 + kg]
                    = m > 0.0f ? m * (1.0f / 7.0f) : 1.0f;
        }
    }
}

void launch_xr64_abs_preamble(
    const uint16_t* X,
    const int32_t* topk_idx,
    const int32_t* topk_off,
    uint32_t* X4,
    float* SX,
    int N,
    int NS,
    int H,
    int TOPK,
    cudaStream_t stream) {

    uint64_t tiles = (uint64_t)(((uint32_t)N + 7u) >> 3)
                   * ((uint32_t)H >> 5);
    xR64ABSRouteQuantizeScatter<<<
        (tiles + XR64_ABS_WARPS - 1u) / XR64_ABS_WARPS,
        XR64_ABS_CTA, 0, stream>>>(
            X, topk_idx, topk_off, X4, SX, N, NS, H, TOPK);
}
