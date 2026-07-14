#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define MOE_PREAMBLE_CTA 256

struct alignas(8) MoeActScale {
    float XGS;
    uint32_t XGSINV2;
};

__device__ __forceinline__ uint32_t moe_ue4m3_ceil(float x) {
    if (!(x > 0.0f)) return 0u;
    if (x >= 448.0f) return 0x7eu;

    uint32_t bits = __float_as_uint(x);
    int e = int((bits >> 23) & 0xffu) - 120;
    if (e <= 0) {
        uint32_t m = __float2uint_ru(x * 512.0f);
        return m > 7u ? 7u : m;
    }

    uint32_t m = (bits >> 20) & 7u;
    m += (bits & 0x000fffffu) != 0u;
    if (m == 8u) {
        m = 0u;
        e++;
    }
    if (e >= 15) return m > 6u ? 0x7eu : (0x78u | m);
    return (uint32_t(e) << 3) | m;
}

__device__ __forceinline__ float moe_ue4m3_f32(uint32_t x) {
    uint32_t e = x >> 3;
    uint32_t m = x & 7u;
    if (!e) return float(m) * (1.0f / 512.0f);
    return __uint_as_float(((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ uint32_t moe_e2m1x2_bf16x2(
    uint32_t x,
    uint32_t scale
) {
    uint32_t q;
    asm volatile(
        "{\n\t"
        ".reg .b32 y;\n\t"
        ".reg .b16 lo, hi;\n\t"
        ".reg .f32 flo, fhi;\n\t"
        ".reg .b8 z;\n\t"
        "mul.rn.bf16x2 y, %1, %2;\n\t"
        "mov.b32 {lo, hi}, y;\n\t"
        "cvt.f32.bf16 flo, lo;\n\t"
        "cvt.f32.bf16 fhi, hi;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 z, fhi, flo;\n\t"
        "cvt.u32.u8 %0, z;\n\t"
        "}"
        : "=r"(q)
        : "r"(x), "r"(scale));
    return q;
}

// X is token-major BF16. Positive finite BF16 magnitudes preserve integer order.
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_act_absmax_partial(
    const uint32_t* __restrict__ X,
    uint64_t pairs,
    uint32_t* __restrict__ partial
) {
    __shared__ uint32_t warp_max[MOE_PREAMBLE_CTA >> 5];
    uint32_t mx = 0u;
    for (uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         p < pairs;
         p += (uint64_t)blockDim.x * gridDim.x) {
        uint32_t x = X[p];
        uint32_t lo = x & 0x7fffu;
        uint32_t hi = (x >> 16) & 0x7fffu;
        mx = max(mx, max(lo, hi));
    }
    for (int d = 16; d; d >>= 1)
        mx = max(mx, __shfl_down_sync(0xffffffffu, mx, d));
    if (!(threadIdx.x & 31)) warp_max[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x < 32) {
        mx = threadIdx.x < (MOE_PREAMBLE_CTA >> 5)
           ? warp_max[threadIdx.x] : 0u;
        for (int d = 16; d; d >>= 1)
            mx = max(mx, __shfl_down_sync(0xffffffffu, mx, d));
        if (!threadIdx.x) partial[blockIdx.x] = mx;
    }
}

__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_act_scale_finalize(
    const uint32_t* __restrict__ partial,
    uint32_t count,
    MoeActScale* __restrict__ scale
) {
    __shared__ uint32_t warp_max[MOE_PREAMBLE_CTA >> 5];
    uint32_t mx = 0u;
    for (uint32_t p = threadIdx.x; p < count; p += blockDim.x)
        mx = max(mx, partial[p]);
    for (int d = 16; d; d >>= 1)
        mx = max(mx, __shfl_down_sync(0xffffffffu, mx, d));
    if (!(threadIdx.x & 31)) warp_max[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (!threadIdx.x) {
        mx = max(max(max(warp_max[0], warp_max[1]),
                     max(warp_max[2], warp_max[3])),
                 max(max(warp_max[4], warp_max[5]),
                     max(warp_max[6], warp_max[7])));
        float amax = __uint_as_float(mx << 16);
        if (mx) {
            scale->XGS = 2688.0f / amax;
            uint32_t inv = __bfloat16_as_ushort(
                __float2bfloat16(amax * (1.0f / 2688.0f)));
            scale->XGSINV2 = inv | (inv << 16);
        } else {
            scale->XGS = 0.0f;
            scale->XGSINV2 = 0u;
        }
    }
}

/*
    one CTA = one X4[kt,n16] + Sx[kt,n16]

    warp w owns token n16*16+w and n16*16+w+8
    lane l owns one BF16x2 at K=2*l

    four source lanes -> one E2M1x8 packet
    32T vector store -> 128 contiguous X4 words
     4T vector store ->  16 contiguous Sx words
*/
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_act_pack(
    const uint32_t* __restrict__ X,
    const MoeActScale* __restrict__ scale,
    uint32_t* __restrict__ X4,
    uint32_t* __restrict__ Sx,
    int N,
    int NP,
    int H
) {
    __shared__ __align__(16) uint32_t tile[144];
    uint32_t n16 = blockIdx.x % ((uint32_t)NP >> 4);
    uint32_t kt = blockIdx.x / ((uint32_t)NP >> 4);
    uint32_t w = (uint32_t)threadIdx.x >> 5;
    uint32_t l = (uint32_t)threadIdx.x & 31u;

    for (uint32_t half = 0; half < 2; half++) {
        uint32_t token = (n16 << 4) + w + (half << 3);
        uint32_t x = token < (uint32_t)N
            ? X[((uint64_t)token * (uint32_t)H + (kt << 6)) / 2u + l]
            : 0u;
        uint32_t mx = max(x & 0x7fffu, (x >> 16) & 0x7fffu);
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1, 8));
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 2, 8));
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 4, 8));

        uint32_t sx = 0u;
        uint32_t qscale = 0u;
        if (!(l & 7u) && mx) {
            sx = moe_ue4m3_ceil(
                __uint_as_float(mx << 16) * scale->XGS * (1.0f / 6.0f));
            uint32_t bf = __bfloat16_as_ushort(__float2bfloat16(
                scale->XGS / moe_ue4m3_f32(sx)));
            qscale = bf | (bf << 16);
        }
        sx = __shfl_sync(0xffffffffu, sx, 0, 8);
        qscale = __shfl_sync(0xffffffffu, qscale, 0, 8);
        uint32_t q = mx ? moe_e2m1x2_bf16x2(x, qscale) : 0u;
        uint32_t q0 = __shfl_sync(0xffffffffu, q, l & ~3u);
        uint32_t q1 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 1u);
        uint32_t q2 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 2u);
        uint32_t q3 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 3u);
        if (!(l & 3u)) {
            uint32_t packet = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
            uint32_t q8 = l >> 2;
            tile[(w << 4) + ((q8 & 3u) << 2) + (half << 1) + (q8 >> 2)]
                = packet;
        }

        uint32_t s0 = __shfl_sync(0xffffffffu, sx, 0);
        uint32_t s1 = __shfl_sync(0xffffffffu, sx, 8);
        uint32_t s2 = __shfl_sync(0xffffffffu, sx, 16);
        uint32_t s3 = __shfl_sync(0xffffffffu, sx, 24);
        if (!l)
            tile[128u + (w << 1) + half]
                = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    }
    __syncthreads();

    if (threadIdx.x < 32)
        reinterpret_cast<uint4*>(X4 + ((uint64_t)kt * ((uint32_t)NP >> 4)
                                     + n16) * 128u)[threadIdx.x]
            = reinterpret_cast<uint4*>(tile)[threadIdx.x];
    if (threadIdx.x < 4)
        reinterpret_cast<uint4*>(Sx + ((uint64_t)kt * ((uint32_t)NP >> 4)
                                     + n16) * 16u)[threadIdx.x]
            = reinterpret_cast<uint4*>(tile + 128)[threadIdx.x];
}

/*
    Stream/SOL activation layout:

    X4[token,H64,p,{K0,K32}] = 8 u32 per token/K64
    Sx[token,H64]             = one UE4M3x4 word

    The quantization math is identical to moe_act_pack; only the destination
    permutation changes so the n256-outer kernels can gather routed tokens.
*/
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_act_pack_stream(
    const uint32_t* __restrict__ X,
    const MoeActScale* __restrict__ scale,
    uint32_t* __restrict__ X4,
    uint32_t* __restrict__ Sx,
    int N,
    int NP,
    int H
) {
    uint32_t n16 = blockIdx.x % ((uint32_t)NP >> 4);
    uint32_t kt = blockIdx.x / ((uint32_t)NP >> 4);
    uint32_t w = (uint32_t)threadIdx.x >> 5;
    uint32_t l = (uint32_t)threadIdx.x & 31u;

    for (uint32_t half = 0; half < 2; half++) {
        uint32_t token = (n16 << 4) + w + (half << 3);
        uint32_t x = token < (uint32_t)N
            ? X[((uint64_t)token * (uint32_t)H + (kt << 6)) / 2u + l]
            : 0u;
        uint32_t mx = max(x & 0x7fffu, (x >> 16) & 0x7fffu);
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1, 8));
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 2, 8));
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 4, 8));

        uint32_t sx = 0u;
        uint32_t qscale = 0u;
        if (!(l & 7u) && mx) {
            sx = moe_ue4m3_ceil(
                __uint_as_float(mx << 16) * scale->XGS * (1.0f / 6.0f));
            uint32_t bf = __bfloat16_as_ushort(__float2bfloat16(
                scale->XGS / moe_ue4m3_f32(sx)));
            qscale = bf | (bf << 16);
        }
        sx = __shfl_sync(0xffffffffu, sx, 0, 8);
        qscale = __shfl_sync(0xffffffffu, qscale, 0, 8);
        uint32_t q = mx ? moe_e2m1x2_bf16x2(x, qscale) : 0u;
        uint32_t q0 = __shfl_sync(0xffffffffu, q, l & ~3u);
        uint32_t q1 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 1u);
        uint32_t q2 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 2u);
        uint32_t q3 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 3u);
        if (!(l & 3u)) {
            uint32_t q8 = l >> 2;
            X4[((uint64_t)token * ((uint32_t)H >> 6) + kt) * 8u
               + ((q8 & 3u) << 1) + (q8 >> 2)]
                = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        }

        uint32_t s0 = __shfl_sync(0xffffffffu, sx, 0);
        uint32_t s1 = __shfl_sync(0xffffffffu, sx, 8);
        uint32_t s2 = __shfl_sync(0xffffffffu, sx, 16);
        uint32_t s3 = __shfl_sync(0xffffffffu, sx, 24);
        if (!l)
            Sx[(uint64_t)token * ((uint32_t)H >> 6) + kt]
                = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    }
}

/*
    one CTA = one expert
    one warp = one 32-token Xb word per pass
    ballot emits the bitplane
    header = {last live n8 + 1, routed token count}
*/
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_route_pack_topk(
    const int32_t* __restrict__ topk_idx,
    const uint16_t* __restrict__ topk_W,
    uint32_t* __restrict__ Xb,
    uint16_t* __restrict__ expert_topk_W,
    int N,
    int NP,
    int TOPK,
    int Xb_stride
) {
    __shared__ uint32_t warp_count[MOE_PREAMBLE_CTA >> 5];
    __shared__ uint32_t warp_last[MOE_PREAMBLE_CTA >> 5];
    uint32_t e = blockIdx.x;
    uint32_t warp = (uint32_t)threadIdx.x >> 5;
    uint32_t lane = (uint32_t)threadIdx.x & 31u;
    uint32_t count = 0u;
    uint32_t last = 0u;

    for (uint32_t word = warp; word < ((uint32_t)NP + 31u) >> 5;
         word += MOE_PREAMBLE_CTA >> 5) {
        uint32_t token = (word << 5) + lane;
        uint32_t hit = 0u;
        uint32_t weight = 0u;
        if (token < (uint32_t)N) {
            for (uint32_t k = 0; k < (uint32_t)TOPK; k++) {
                uint32_t yes = (uint32_t)topk_idx[token * TOPK + k] == e;
                hit |= yes;
                if (yes) weight = topk_W[token * TOPK + k];
            }
        }
        uint32_t bits = __ballot_sync(0xffffffffu, hit != 0u);
        if (!lane) {
            Xb[(uint64_t)e * (uint32_t)Xb_stride + 1u + word] = bits;
            count += __popc(bits);
            if (bits)
                last = (word << 2) + ((31u - __clz(bits)) >> 3) + 1u;
        }
        if (token < (uint32_t)NP)
            expert_topk_W[(uint64_t)e * (uint32_t)NP + token]
                = hit ? (uint16_t)weight : 0u;
    }

    if (!lane) {
        warp_count[warp] = count;
        warp_last[warp] = last;
    }
    __syncthreads();
    if (threadIdx.x < (MOE_PREAMBLE_CTA >> 5)) {
        count = warp_count[threadIdx.x];
        last = warp_last[threadIdx.x];
        for (int d = 4; d; d >>= 1) {
            count += __shfl_down_sync(0xffu, count, d);
            last = max(last, __shfl_down_sync(0xffu, last, d));
        }
        if (!threadIdx.x) {
            count = (last << 16) | (count & 0xffffu);
            asm volatile(
                "st.release.gpu.global.u32 [%0], %1;"
                :
                : "l"((uint64_t)__cvta_generic_to_global(
                      Xb + (uint64_t)e * (uint32_t)Xb_stride)),
                  "r"(count)
                : "memory");
        }
    }
}

/*
    Xb / expert_topk_W keep the FF1 contract unchanged.

    expert_token_idx[e,0]        = routed token count
    expert_token_idx[e,1+packed] = ascending-token select sidecar:

      Xb[e]       0..0..1..0..1..0
      packed      0        1
      token_idx   token_a  token_b

    Eight warps compact one 32-token word each per round.  The shared prefix
    is CTA-local and deterministic; no atomics touch the packed cursor.
*/
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_route_pack_topk_sidecar(
    const int32_t* __restrict__ topk_idx,
    const uint16_t* __restrict__ topk_W,
    uint32_t* __restrict__ Xb,
    uint16_t* __restrict__ expert_topk_W,
    uint16_t* __restrict__ expert_token_idx,
    int N,
    int NP,
    int TOPK,
    int Xb_stride
) {
    __shared__ uint32_t warp_count[MOE_PREAMBLE_CTA >> 5];
    __shared__ uint32_t warp_last[MOE_PREAMBLE_CTA >> 5];
    __shared__ uint32_t packed_base;
    __shared__ uint32_t last_live;
    uint32_t e = blockIdx.x;
    uint32_t warp = (uint32_t)threadIdx.x >> 5;
    uint32_t lane = (uint32_t)threadIdx.x & 31u;

    for (uint32_t p = (uint32_t)threadIdx.x; p <= (uint32_t)NP;
         p += MOE_PREAMBLE_CTA) {
        if (p < (uint32_t)NP)
            expert_topk_W[(uint64_t)e * (uint32_t)NP + p] = 0u;
        expert_token_idx[(uint64_t)e * ((uint32_t)NP + 1u) + p] = 0u;
    }
    if (!threadIdx.x) {
        packed_base = 0u;
        last_live = 0u;
    }
    __syncthreads();

    for (uint32_t first = 0u;
         first < (((uint32_t)NP + 31u) >> 5);
         first += MOE_PREAMBLE_CTA >> 5) {
        uint32_t word = first + warp;
        uint32_t token = (word << 5) + lane;
        uint32_t hit = 0u;
        uint32_t weight = 0u;
        if (word < (((uint32_t)NP + 31u) >> 5)
            && token < (uint32_t)N) {
            for (uint32_t k = 0; k < (uint32_t)TOPK; k++) {
                uint32_t yes = (uint32_t)topk_idx[token * TOPK + k] == e;
                hit |= yes;
                if (yes) weight = topk_W[token * TOPK + k];
            }
        }

        uint32_t bits = __ballot_sync(0xffffffffu, hit != 0u);
        if (!lane) {
            warp_count[warp] = __popc(bits);
            warp_last[warp] = bits
                ? (word << 2) + ((31u - __clz(bits)) >> 3) + 1u : 0u;
            if (word < (((uint32_t)NP + 31u) >> 5))
                Xb[(uint64_t)e * (uint32_t)Xb_stride + 1u + word] = bits;
        }
        if (token < (uint32_t)NP)
            expert_topk_W[(uint64_t)e * (uint32_t)NP + token]
                = hit ? (uint16_t)weight : 0u;
        __syncthreads();

        uint32_t rank = packed_base;
        for (uint32_t w = 0u; w < warp; w++) rank += warp_count[w];
        rank += __popc(bits & ((1u << lane) - 1u));
        if (hit)
            expert_token_idx[(uint64_t)e * ((uint32_t)NP + 1u) + 1u + rank]
                = (uint16_t)token;
        __syncthreads();

        if (!threadIdx.x) {
            for (uint32_t w = 0u; w < (MOE_PREAMBLE_CTA >> 5); w++) {
                packed_base += warp_count[w];
                last_live = max(last_live, warp_last[w]);
            }
        }
        __syncthreads();
    }

    if (!threadIdx.x) {
        uint32_t header = (last_live << 16) | (packed_base & 0xffffu);
        expert_token_idx[(uint64_t)e * ((uint32_t)NP + 1u)]
            = (uint16_t)packed_base;
        asm volatile(
            "st.release.gpu.global.u32 [%0], %1;"
            :
            : "l"((uint64_t)__cvta_generic_to_global(
                  Xb + (uint64_t)e * (uint32_t)Xb_stride)),
              "r"(header)
            : "memory");
    }
}

/*
    Deterministic expert packing for the contiguous path.

    topk_off[token,slot]       = expert-local packed row p
    expert_topk_W[expert,p]    = BF16 routing weight
    expert_token_idx[expert,0] = routed row count
    expert_token_idx[expert,1+p] = original token

    One CTA owns one expert, so the packed cursor is CTA-local.  Every
    assignment slot has one writer and no global atomic is required.
*/
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_route_pack_contiguous(
    const int32_t* __restrict__ topk_idx,
    const uint16_t* __restrict__ topk_W,
    uint16_t* __restrict__ topk_off,
    uint16_t* __restrict__ expert_topk_W,
    uint16_t* __restrict__ expert_token_idx,
    int N,
    int NP,
    int TOPK
) {
    __shared__ uint32_t warp_count[MOE_PREAMBLE_CTA >> 5];
    __shared__ uint32_t packed_base;
    uint32_t e = blockIdx.x;
    uint32_t warp = (uint32_t)threadIdx.x >> 5;
    uint32_t lane = (uint32_t)threadIdx.x & 31u;

    for (uint32_t p = (uint32_t)threadIdx.x; p <= (uint32_t)NP;
         p += MOE_PREAMBLE_CTA) {
        if (p < (uint32_t)NP)
            expert_topk_W[(uint64_t)e * (uint32_t)NP + p] = 0u;
        expert_token_idx[(uint64_t)e * ((uint32_t)NP + 1u) + p] = 0u;
    }
    if (!threadIdx.x) packed_base = 0u;
    __syncthreads();

    for (uint32_t first = 0u; first < ((uint32_t)N + 31u) >> 5;
         first += MOE_PREAMBLE_CTA >> 5) {
        uint32_t token = ((first + warp) << 5) + lane;
        uint32_t hit = 0u;
        uint32_t slot = 0u;
        uint32_t weight = 0u;
        if (token < (uint32_t)N) {
            for (uint32_t k = 0; k < (uint32_t)TOPK; k++) {
                uint32_t yes = (uint32_t)topk_idx[token * TOPK + k] == e;
                if (yes) {
                    hit = 1u;
                    slot = k;
                    weight = topk_W[token * TOPK + k];
                }
            }
        }

        uint32_t bits = __ballot_sync(0xffffffffu, hit != 0u);
        if (!lane) warp_count[warp] = __popc(bits);
        __syncthreads();

        uint32_t rank = packed_base;
        for (uint32_t w = 0u; w < warp; w++) rank += warp_count[w];
        rank += __popc(bits & ((1u << lane) - 1u));
        if (hit) {
            topk_off[token * TOPK + slot] = (uint16_t)rank;
            expert_topk_W[(uint64_t)e * (uint32_t)NP + rank]
                = (uint16_t)weight;
            expert_token_idx[(uint64_t)e * ((uint32_t)NP + 1u) + 1u + rank]
                = (uint16_t)token;
        }
        __syncthreads();

        if (!threadIdx.x) {
            for (uint32_t w = 0u; w < (MOE_PREAMBLE_CTA >> 5); w++)
                packed_base += warp_count[w];
        }
        __syncthreads();
    }

    if (!threadIdx.x)
        expert_token_idx[(uint64_t)e * ((uint32_t)NP + 1u)]
            = (uint16_t)packed_base;
}

/*
    FF1-native expert-contiguous activation layout.

    X4[e,n16,H64,q8,lp4,{n8_0x2,n8_1x2}]
      lane lp reads both N8 fragments with one v4.

    Sx[e,n32,H64,q8,{n16a_0,n16a_1,n16b_0,n16b_1}]
      scale lane reads four adjacent N8 scales with one v4.

    Quantization is performed once per original token/K64.  The resulting
    packets are scattered to its TOPK expert-local offsets.
*/
__global__ __launch_bounds__(MOE_PREAMBLE_CTA)
void moe_act_pack_expert_contiguous(
    const uint32_t* __restrict__ X,
    const MoeActScale* __restrict__ scale,
    const int32_t* __restrict__ topk_idx,
    const uint16_t* __restrict__ topk_off,
    uint32_t* __restrict__ X4,
    uint32_t* __restrict__ Sx,
    int N,
    int NP,
    int H,
    int TOPK
) {
    uint32_t n16 = blockIdx.x % ((uint32_t)NP >> 4);
    uint32_t kt = blockIdx.x / ((uint32_t)NP >> 4);
    uint32_t w = (uint32_t)threadIdx.x >> 5;
    uint32_t l = (uint32_t)threadIdx.x & 31u;

    for (uint32_t half = 0; half < 2; half++) {
        uint32_t token = (n16 << 4) + w + (half << 3);
        if (token >= (uint32_t)N) continue;

        uint32_t x = X[((uint64_t)token * (uint32_t)H + (kt << 6)) / 2u + l];
        uint32_t mx = max(x & 0x7fffu, (x >> 16) & 0x7fffu);
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 1, 8));
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 2, 8));
        mx = max(mx, __shfl_xor_sync(0xffffffffu, mx, 4, 8));

        uint32_t sx = 0u;
        uint32_t qscale = 0u;
        if (!(l & 7u) && mx) {
            sx = moe_ue4m3_ceil(
                __uint_as_float(mx << 16) * scale->XGS * (1.0f / 6.0f));
            uint32_t bf = __bfloat16_as_ushort(__float2bfloat16(
                scale->XGS / moe_ue4m3_f32(sx)));
            qscale = bf | (bf << 16);
        }
        sx = __shfl_sync(0xffffffffu, sx, 0, 8);
        qscale = __shfl_sync(0xffffffffu, qscale, 0, 8);
        uint32_t q = mx ? moe_e2m1x2_bf16x2(x, qscale) : 0u;
        uint32_t q0 = __shfl_sync(0xffffffffu, q, l & ~3u);
        uint32_t q1 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 1u);
        uint32_t q2 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 2u);
        uint32_t q3 = __shfl_sync(0xffffffffu, q, (l & ~3u) + 3u);
        uint32_t packet = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);

        uint32_t s0 = __shfl_sync(0xffffffffu, sx, 0);
        uint32_t s1 = __shfl_sync(0xffffffffu, sx, 8);
        uint32_t s2 = __shfl_sync(0xffffffffu, sx, 16);
        uint32_t s3 = __shfl_sync(0xffffffffu, sx, 24);
        uint32_t spacket = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);

        for (uint32_t k = 0; k < (uint32_t)TOPK; k++) {
            uint32_t e = 0u;
            uint32_t p = 0u;
            if (!l) {
                e = (uint32_t)topk_idx[token * TOPK + k];
                p = topk_off[token * TOPK + k];
            }
            e = __shfl_sync(0xffffffffu, e, 0);
            p = __shfl_sync(0xffffffffu, p, 0);

            if (!(l & 3u)) {
                uint32_t q8 = l >> 2;
                X4[(uint64_t)e * (uint32_t)NP * ((uint32_t)H >> 3)
                    + (uint64_t)(p >> 4) * ((uint32_t)H << 1)
                    + (kt << 7)
                    + (p & 7u) * 16u
                    + (q8 & 3u) * 4u
                    + ((p >> 3) & 1u) * 2u
                    + (q8 >> 2)] = packet;
            }
            if (!l) {
                Sx[(uint64_t)e * (uint32_t)NP * ((uint32_t)H >> 6)
                    + (uint64_t)(p >> 5) * (((uint32_t)H >> 6) << 5)
                    + kt * 32u
                    + (p & 7u) * 4u
                    + ((p >> 3) & 3u)] = spacket;
            }
        }
    }
}
