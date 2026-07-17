#include "xR64ABS.cuh"

#define XR64_FF2_N8(Q, O0, O1) do {                                        \
    uint32_t a = 0u;                                                        \
    if (nbase + (Q) * 8u + (lane >> 2) < live)                             \
        a = xr64_ld_ca_nc_u32(Y4                                            \
            + ((uint64_t)e * NS + nbase + (Q) * 8u + (lane >> 2)) * I8    \
            + kg * 4u + (lane & 3u));                                      \
    float sy0 = 0.0f, sy1 = 0.0f;                                          \
    if (lane < 4u) {                                                        \
        uint32_t t = nbase + (Q) * 8u + ((lane & 3u) << 1);                \
        if (t < live)                                                       \
            sy0 = xr64_ld_ca_nc_f32(                                        \
                SY + ((uint64_t)e * NS + t) * I32 + kg);                   \
        if (t + 1u < live)                                                  \
            sy1 = xr64_ld_ca_nc_f32(                                        \
                SY + ((uint64_t)e * NS + t + 1u) * I32 + kg);              \
    }                                                                       \
    sy0 = __shfl_sync(0xffffffffu, sy0, lane & 3u);                         \
    sy1 = __shfl_sync(0xffffffffu, sy1, lane & 3u);                         \
    int32_t d0, d1;                                                         \
    xr64_mma_m8n8k32_s4(d0, d1, w, a);                                     \
    O0 = fmaf((float)d0, sw * sy0, O0);                                    \
    O1 = fmaf((float)d1, sw * sy1, O1);                                    \
} while (0)

#define XR64_FF2_REDUCE(Q, O0, O1) do {                                    \
    uint32_t t0 = nbase + (Q) * 8u + ((lane & 3u) << 1);                   \
    uint32_t token0 = 0u, token1 = 0u;                                     \
    if (lane < 4u && t0 < live) {                                          \
        token0 = xr64_ld_ca_nc_u32(                                         \
            route_token + (uint64_t)e * NS + t0);                          \
        if (t0 + 1u < live)                                                 \
            token1 = xr64_ld_ca_nc_u32(                                     \
                route_token + (uint64_t)e * NS + t0 + 1u);                 \
    }                                                                       \
    token0 = __shfl_sync(0xffffffffu, token0, lane & 3u);                   \
    token1 = __shfl_sync(0xffffffffu, token1, lane & 3u);                   \
    if (t0 < live && feature < (uint32_t)H)                                \
        atomicAdd(Y + (uint64_t)token0 * H + feature, O0);                 \
    if (t0 + 1u < live && feature < (uint32_t)H)                           \
        atomicAdd(Y + (uint64_t)token1 * H + feature, O1);                 \
} while (0)

// n32 -> H1024 -> H256 -> K128 -> K32
// topk_W is already folded into Y4 by FF1; this is the routed expert sum.
__global__ __launch_bounds__(XR64_ABS_CTA, 1)
void xR64ABSFF2(
    const uint32_t* __restrict__ W2,
    const float* __restrict__ S2,
    const uint32_t* __restrict__ Y4,
    const float* __restrict__ SY,
    const int32_t* __restrict__ route_token,
    const int32_t* __restrict__ count,
    float* __restrict__ Y,
    int NS,
    int I,
    int H) {

    uint32_t e = blockIdx.x;
    uint32_t live = count[e];
    if (!live) return;

    uint32_t lane = threadIdx.x & 31u;
    uint32_t I8 = (uint32_t)I >> 3;
    uint32_t I32 = (uint32_t)I >> 5;
    uint32_t I128 = ((uint32_t)I + 127u) >> 7;

    for (uint32_t nbase = 0; nbase < live; nbase += 32u) {
        for (uint32_t h1024 = 0; h1024 < (uint32_t)H; h1024 += 1024u) {
            for (uint32_t plane = 0; plane < 4u; plane++) {
                uint32_t hbase = h1024 + plane * XR64_ABS_PLANE;
                if (hbase >= (uint32_t)H) break;

                uint32_t feature = hbase
                    + (((uint32_t)threadIdx.x >> 5) << 3) + (lane >> 2);
                float o00 = 0.0f, o01 = 0.0f;
                float o10 = 0.0f, o11 = 0.0f;
                float o20 = 0.0f, o21 = 0.0f;
                float o30 = 0.0f, o31 = 0.0f;

                for (uint32_t g128 = 0; g128 < I128; g128++) {
                    float sw = 0.0f;
                    if (feature < (uint32_t)H && !(lane & 3u))
                        sw = xr64_ld_cs_nc_f32(
                            S2 + ((uint64_t)e * H + feature) * I128 + g128);
                    sw = __shfl_sync(0xffffffffu, sw, lane & ~3u);

                    for (uint32_t qk = 0; qk < 4u; qk++) {
                        uint32_t kg = (g128 << 2) + qk;
                        if (kg >= I32) break;

                        uint32_t w = 0u;
                        if (feature < (uint32_t)H)
                            w = xr64_ld_cs_nc_pf_u32(
                                W2 + ((uint64_t)e * H + feature) * I8
                                    + kg * 4u + (lane & 3u));

                        XR64_FF2_N8(0, o00, o01);
                        if (nbase + 8u < live)
                            XR64_FF2_N8(1, o10, o11);
                        if (nbase + 16u < live)
                            XR64_FF2_N8(2, o20, o21);
                        if (nbase + 24u < live)
                            XR64_FF2_N8(3, o30, o31);
                    }
                }

                XR64_FF2_REDUCE(0, o00, o01);
                if (nbase + 8u < live)
                    XR64_FF2_REDUCE(1, o10, o11);
                if (nbase + 16u < live)
                    XR64_FF2_REDUCE(2, o20, o21);
                if (nbase + 24u < live)
                    XR64_FF2_REDUCE(3, o30, o31);
            }
        }
    }
}

void launch_xr64_abs_ff2(
    const uint32_t* W2,
    const float* S2,
    const uint32_t* Y4,
    const float* SY,
    const int32_t* route_token,
    const int32_t* count,
    float* Y,
    int E,
    int NS,
    int I,
    int H,
    cudaStream_t stream) {

    xR64ABSFF2<<<E, XR64_ABS_CTA, 0, stream>>>(
        W2, S2, Y4, SY, route_token, count, Y, NS, I, H);
}
