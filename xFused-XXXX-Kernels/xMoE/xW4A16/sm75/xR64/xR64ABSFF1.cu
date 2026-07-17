#include "xR64ABS.cuh"

#ifndef XR64_ABS_FF1_KERNEL
#define XR64_ABS_FF1_KERNEL xR64ABSFF1
#endif

#define XR64_FF1_N8(Q, G0, G1, U0, U1) do {                                \
    uint32_t a = 0u;                                                        \
    if (nbase + (Q) * 8u + (lane >> 2) < live)                             \
        a = xr64_ld_ca_nc_u32(X4                                            \
            + ((uint64_t)e * NS + nbase + (Q) * 8u + (lane >> 2)) * H8    \
            + kg * 4u + (lane & 3u));                                      \
    float sx0 = 0.0f, sx1 = 0.0f;                                          \
    if (lane < 4u) {                                                        \
        uint32_t t = nbase + (Q) * 8u + ((lane & 3u) << 1);                \
        if (t < live)                                                       \
            sx0 = xr64_ld_ca_nc_f32(                                        \
                SX + ((uint64_t)e * NS + t) * H32 + kg);                   \
        if (t + 1u < live)                                                  \
            sx1 = xr64_ld_ca_nc_f32(                                        \
                SX + ((uint64_t)e * NS + t + 1u) * H32 + kg);              \
    }                                                                       \
    sx0 = __shfl_sync(0xffffffffu, sx0, lane & 3u);                         \
    sx1 = __shfl_sync(0xffffffffu, sx1, lane & 3u);                         \
    int32_t d0, d1;                                                         \
    xr64_mma_m8n8k32_s4(d0, d1, wg, a);                                    \
    G0 = fmaf((float)d0, swg * sx0, G0);                                   \
    G1 = fmaf((float)d1, swg * sx1, G1);                                   \
    xr64_mma_m8n8k32_s4(d0, d1, wu, a);                                    \
    U0 = fmaf((float)d0, swu * sx0, U0);                                   \
    U1 = fmaf((float)d1, swu * sx1, U1);                                   \
} while (0)

#define XR64_FF1_STORE(Q, G0, G1, U0, U1) do {                             \
    uint32_t t0 = nbase + (Q) * 8u + ((lane & 3u) << 1);                   \
    uint32_t tw = 0u;                                                       \
    if (lane < 4u && t0 < live)                                            \
        tw = xr64_ld_ca_nc_u32(                                             \
            route_weight + (uint64_t)e * NS + t0);                         \
    tw = __shfl_sync(0xffffffffu, tw, lane & 3u);                           \
    float y0 = xr64_gate<GELU_TANH>(G0) * U0                               \
             * xr64_bf16((uint16_t)tw);                                    \
    float y1 = xr64_gate<GELU_TANH>(G1) * U1                               \
             * xr64_bf16((uint16_t)(tw >> 16));                            \
    uint32_t f = ((uint32_t)threadIdx.x >> 5) * 8u + (lane >> 2);          \
    if (t0 < live && ibase + f < (uint32_t)I)                              \
        smem[((Q) * 8u + ((lane & 3u) << 1)) * XR64_ABS_PLANE + f] = y0;  \
    if (t0 + 1u < live && ibase + f < (uint32_t)I)                         \
        smem[((Q) * 8u + ((lane & 3u) << 1) + 1u)                         \
            * XR64_ABS_PLANE + f] = y1;                                   \
} while (0)

// n32 -> I1024 -> I256 -> K128 -> K32
// one W13/S13 fragment replays across four adjacent N8 consumers.
template <bool GELU_TANH>
__global__ __launch_bounds__(XR64_ABS_CTA, 1)
void XR64_ABS_FF1_KERNEL(
    const uint32_t* __restrict__ W13,
    const float* __restrict__ S13,
    const uint32_t* __restrict__ X4,
    const float* __restrict__ SX,
    const uint16_t* __restrict__ route_weight,
    const int32_t* __restrict__ count,
    uint32_t* __restrict__ Y4,
    float* __restrict__ SY,
    int NS,
    int I,
    int H) {

    extern __shared__ float smem[];
    uint32_t e = blockIdx.x;
    uint32_t live = count[e];
    if (!live) return;

    uint32_t lane = threadIdx.x & 31u;
    uint32_t H8 = (uint32_t)H >> 3;
    uint32_t H32 = (uint32_t)H >> 5;
    uint32_t H128 = ((uint32_t)H + 127u) >> 7;

    for (uint32_t nbase = 0; nbase < live; nbase += 32u) {
        for (uint32_t i1024 = 0; i1024 < (uint32_t)I; i1024 += 1024u) {
            for (uint32_t plane = 0; plane < 4u; plane++) {
                uint32_t ibase = i1024 + plane * XR64_ABS_PLANE;
                if (ibase >= (uint32_t)I) break;

                float g00 = 0.0f, g01 = 0.0f, u00 = 0.0f, u01 = 0.0f;
                float g10 = 0.0f, g11 = 0.0f, u10 = 0.0f, u11 = 0.0f;
                float g20 = 0.0f, g21 = 0.0f, u20 = 0.0f, u21 = 0.0f;
                float g30 = 0.0f, g31 = 0.0f, u30 = 0.0f, u31 = 0.0f;
                uint32_t feature = ibase
                    + (((uint32_t)threadIdx.x >> 5) << 3) + (lane >> 2);

                for (uint32_t g128 = 0; g128 < H128; g128++) {
                    float swg = 0.0f, swu = 0.0f;
                    if (feature < (uint32_t)I && !(lane & 3u)) {
                        swg = xr64_ld_cs_nc_f32(
                            S13 + ((uint64_t)e * ((uint32_t)I << 1)
                                + feature) * H128 + g128);
                        swu = xr64_ld_cs_nc_f32(
                            S13 + ((uint64_t)e * ((uint32_t)I << 1)
                                + I + feature) * H128 + g128);
                    }
                    swg = __shfl_sync(0xffffffffu, swg, lane & ~3u);
                    swu = __shfl_sync(0xffffffffu, swu, lane & ~3u);

                    for (uint32_t qk = 0; qk < 4u; qk++) {
                        uint32_t kg = (g128 << 2) + qk;
                        if (kg >= H32) break;

                        uint32_t wg = 0u, wu = 0u;
                        if (feature < (uint32_t)I) {
                            wg = xr64_ld_cs_nc_pf_u32(
                                W13 + ((uint64_t)e * ((uint32_t)I << 1)
                                    + feature) * H8 + kg * 4u + (lane & 3u));
                            wu = xr64_ld_cs_nc_pf_u32(
                                W13 + ((uint64_t)e * ((uint32_t)I << 1)
                                    + I + feature) * H8 + kg * 4u + (lane & 3u));
                        }

                        XR64_FF1_N8(0, g00, g01, u00, u01);
                        if (nbase + 8u < live)
                            XR64_FF1_N8(1, g10, g11, u10, u11);
                        if (nbase + 16u < live)
                            XR64_FF1_N8(2, g20, g21, u20, u21);
                        if (nbase + 24u < live)
                            XR64_FF1_N8(3, g30, g31, u30, u31);
                    }
                }

                XR64_FF1_STORE(0, g00, g01, u00, u01);
                if (nbase + 8u < live)
                    XR64_FF1_STORE(1, g10, g11, u10, u11);
                if (nbase + 16u < live)
                    XR64_FF1_STORE(2, g20, g21, u20, u21);
                if (nbase + 24u < live)
                    XR64_FF1_STORE(3, g30, g31, u30, u31);
                __syncthreads();

                uint32_t tiles = min(32u, live - nbase) * 8u;
                for (uint32_t tile = (uint32_t)threadIdx.x >> 5;
                     tile < tiles; tile += XR64_ABS_WARPS) {
                    uint32_t t = tile >> 3;
                    uint32_t fg = tile & 7u;
                    uint32_t f = (fg << 5) + lane;
                    float x = nbase + t < live && ibase + f < (uint32_t)I
                        ? smem[t * XR64_ABS_PLANE + f] : 0.0f;
                    float m = fabsf(x);
                    for (int d = 16; d; d >>= 1)
                        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
                    float inv = m > 0.0f ? 7.0f / m : 0.0f;
                    uint32_t q = xr64_q4(x, inv);
                    uint32_t src = lane & ~7u;
                    uint32_t packed = __shfl_sync(0xffffffffu, q, src)
                        | (__shfl_sync(0xffffffffu, q, src + 1u) << 4)
                        | (__shfl_sync(0xffffffffu, q, src + 2u) << 8)
                        | (__shfl_sync(0xffffffffu, q, src + 3u) << 12)
                        | (__shfl_sync(0xffffffffu, q, src + 4u) << 16)
                        | (__shfl_sync(0xffffffffu, q, src + 5u) << 20)
                        | (__shfl_sync(0xffffffffu, q, src + 6u) << 24)
                        | (__shfl_sync(0xffffffffu, q, src + 7u) << 28);
                    if (!(lane & 7u) && nbase + t < live
                        && ibase + f < (uint32_t)I)
                        Y4[((uint64_t)e * NS + nbase + t)
                            * ((uint32_t)I >> 3)
                            + ((ibase + (fg << 5)) >> 3) + (lane >> 3)]
                            = packed;
                    if (!lane && nbase + t < live
                        && ibase + (fg << 5) < (uint32_t)I)
                        SY[((uint64_t)e * NS + nbase + t)
                            * ((uint32_t)I >> 5)
                            + (ibase >> 5) + fg]
                            = m > 0.0f ? m * (1.0f / 7.0f) : 1.0f;
                }
                __syncthreads();
            }
        }
    }
}

void launch_xr64_abs_ff1(
    const uint32_t* W13,
    const float* S13,
    const uint32_t* X4,
    const float* SX,
    const uint16_t* route_weight,
    const int32_t* count,
    uint32_t* Y4,
    float* SY,
    int E,
    int NS,
    int I,
    int H,
    bool gelu_tanh,
    cudaStream_t stream) {

    if (gelu_tanh)
        XR64_ABS_FF1_KERNEL<true><<<
            E, XR64_ABS_CTA, XR64_ABS_SMEM_BYTES, stream>>>(
            W13, S13, X4, SX, route_weight, count, Y4, SY, NS, I, H);
    else
        XR64_ABS_FF1_KERNEL<false><<<
            E, XR64_ABS_CTA, XR64_ABS_SMEM_BYTES, stream>>>(
            W13, S13, X4, SX, route_weight, count, Y4, SY, NS, I, H);
}
