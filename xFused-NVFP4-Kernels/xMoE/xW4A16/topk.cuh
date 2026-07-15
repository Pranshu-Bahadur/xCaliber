#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define MOE_TOPK_CTA 256

/*
    finite BF16 logits[N,E] -> {topk_idx i32, topk_W BF16}[N,TOPK]

    CTA256 = 8 tokens, one warp/token
    lane l owns {2l,2l+1} + 64j
    even E <= 1024, TOPK <= min(E,8) => one u32 dead mask/lane

    key = {ordered BF16, 0xffff-expert}
      integer max preserves BF16 order and selects lower expert on ties
      redux.sync.max.u32 elects one winner warp-wide

    Softmax and sigmoid are monotonic, so only selected logits are converted
    to f32. Both paths normalize over TOPK; routed_scale is applied last.
*/
template <bool SIGMOID>
__global__ __launch_bounds__(MOE_TOPK_CTA)
void moe_topk_bf16(
    const uint16_t* __restrict__ logits,
    int32_t* __restrict__ topk_idx,
    uint16_t* __restrict__ topk_W,
    int N,
    int E,
    int TOPK,
    float routed_scale
) {
    uint32_t lane = (uint32_t)threadIdx.x & 31u;
    uint32_t token = (uint32_t)blockIdx.x * 8u
                   + ((uint32_t)threadIdx.x >> 5);
    if (token >= (uint32_t)N) return;

    uint32_t dead = 0u;
    uint32_t selected = 0u;
    for (uint32_t rank = 0u; rank < (uint32_t)TOPK; rank++) {
        uint32_t best = 0u;
        for (uint32_t e = lane << 1, slot = 0u;
             e < (uint32_t)E;
             e += 64u, slot += 2u) {
            uint32_t x;
            asm volatile(
                "ld.global.ca.u32 %0, [%1];"
                : "=r"(x)
                : "l"((uint64_t)__cvta_generic_to_global(
                      logits + (uint64_t)token * (uint32_t)E + e))
                : "memory");

            if (!(dead & (1u << slot))) {
                uint32_t raw = x & 0xffffu;
                uint32_t key = raw
                    ^ ((0u - (raw >> 15)) | 0x8000u);
                best = max(best,
                    (key << 16) | (0xffffu - e));
            }
            if (!(dead & (2u << slot))) {
                uint32_t raw = x >> 16;
                uint32_t key = raw
                    ^ ((0u - (raw >> 15)) | 0x8000u);
                best = max(best,
                    (key << 16) | (0xfffeu - e));
            }
        }

        uint32_t winner;
        asm volatile(
            "redux.sync.max.u32 %0, %1, 0xffffffff;"
            : "=r"(winner)
            : "r"(best));
        uint32_t expert = 0xffffu - (winner & 0xffffu);
        dead |= (uint32_t)(lane == ((expert >> 1) & 31u))
             << (((expert >> 6) << 1) | (expert & 1u));
        if (lane == rank) selected = winner;
    }

    float x = 0.0f;
    uint32_t expert = 0u;
    if (lane < (uint32_t)TOPK) {
        expert = 0xffffu - (selected & 0xffffu);
        uint32_t ordered = selected >> 16;
        uint32_t raw = ordered
            ^ (0xffffu ^ ((0u - (ordered >> 15)) & 0x7fffu));
        uint16_t h = (uint16_t)raw;
        asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(x) : "h"(h));
    }

    float top1 = __shfl_sync(0xffffffffu, x, 0);
    float weight = 0.0f;
    if (lane < (uint32_t)TOPK) {
        if constexpr (SIGMOID) {
            float z = -x * 1.4426950408889634f;
            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                         : "=f"(z) : "f"(z));
            z += 1.0f;
            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                         : "=f"(weight) : "f"(z));
        } else {
            float z = (x - top1) * 1.4426950408889634f;
            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                         : "=f"(weight) : "f"(z));
        }
    }

    float denom = weight;
    denom += __shfl_xor_sync(0xffffffffu, denom, 16);
    denom += __shfl_xor_sync(0xffffffffu, denom, 8);
    denom += __shfl_xor_sync(0xffffffffu, denom, 4);
    denom += __shfl_xor_sync(0xffffffffu, denom, 2);
    denom += __shfl_xor_sync(0xffffffffu, denom, 1);
    float inv;
    asm volatile("rcp.approx.ftz.f32 %0, %1;"
                 : "=f"(inv) : "f"(denom));

    if (lane < (uint32_t)TOPK) {
        weight *= inv * routed_scale;
        uint16_t out;
        asm volatile("cvt.rn.bf16.f32 %0, %1;"
                     : "=h"(out) : "f"(weight));
        topk_idx[(uint64_t)token * (uint32_t)TOPK + lane]
            = (int32_t)expert;
        topk_W[(uint64_t)token * (uint32_t)TOPK + lane] = out;
    }
}
