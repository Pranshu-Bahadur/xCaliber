#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <cfloat>
#include <cooperative_groups.h>
#include <type_traits>
#include "ptx.inl"


template <bool softmax>
__global__ void topk_kernel(
    const __nv_bfloat16* router_logits,
    int* topk_idx,
    __nv_bfloat16* topk_weights,
    const int K,
    const int N,
    const int E
){
    uint32_t smd = 0u;
    uint32_t rmem[32];
    const int lane = (threadIdx.x + (threadIdx.y << 3));
    if (((uint64_t(blockIdx.x) << 3) + threadIdx.z) >= N) {
        return;
    }
    #pragma unroll 8
    for (int k = 0; k < K; k++) {
        rmem[16 + k] = 0u;
    }
    for (int i = 0; i <= ((E + 255) >> 8); i++) {
        if (i && ((lane << 3) + ((i - 1) << 8)) < E) {
            for (int j = ((i-1) << 2); j < (i << 2); j++) {
                if (softmax) {
                        rmem[j] = softmax_bf16x2(rmem[j]);
                        if (!j) smd = 0u;
                        add_bf16x2x1(rmem[j], smd);
                }
                if (!softmax) {
                        rmem[j] = softmax_bf16x2(rmem[j] ^ 0x8000'8000u);
                        add_bf16x2(rmem[j], 0x3f80'3f80u);
                        rcp_bf16x2(rmem[j]);
                }
                uint16_t e_offset = (uint16_t)(0xffffu - ((lane << 3) + ((i - 1) << 8) + ((j & 3) << 1)));
                rmem[j ^ 4] =  (rmem[j] << 16) | e_offset;
                rmem[j] =  (rmem[j] & 0xffff'0000u) | (e_offset - 1u);
                #pragma unroll 2
                for (int candidate = 0; candidate < 2; candidate++) {
                    uint32_t key = candidate ? rmem[j] : rmem[j ^ 4];
                    for (int k = 0; k < min(K, (j << 1) + candidate + 1); k++) {
                        uint32_t tmp = rmem[16 + k];
                        rmem[16 + k] = max(tmp, key);
                        key = min(tmp, key);
                    }
                }
            }
        }
        if (((lane << 3) + (i << 8)) < E) {
            ldcg_b32v4(
                router_logits
                + (((uint64_t(blockIdx.x) << 3) + threadIdx.z) * E)
                + (lane << 3) + (i << 8),
                &rmem[i << 2]
            );
        }
    }
    if (softmax) {
        #pragma unroll 5
        for (int i = 16; i > 0; i >>= 1) {
            add_bf16x2(smd, __shfl_xor_sync(0xffff'ffffu, smd, i));
        }
        rcp_bf16x2(smd);
        smd = 0xffff'0000u & smd;
    }
    int p = 16;
    uint32_t key = rmem[p];
    for (int k = 0; k < K; k++) {
        uint32_t tmp = key;
        #pragma unroll 5
        for (int i = 16; i > 0; i >>= 1) {
            tmp = max(tmp, __shfl_xor_sync(0xffff'ffffu, tmp, i));
        }
        rmem[k] = tmp;
        if (key == tmp) {
            ++p;
            key = (p < K + 16) ? rmem[p] : 0u;
        }
    }
    if (lane < (K << 1)) {
        int k = (lane < K) ? lane : lane - K;
        if (lane >= K) {
            topk_idx[(((uint64_t(blockIdx.x) << 3) + threadIdx.z) * K) + (uint64_t)(k)] = int(0xffffu - (rmem[k] & 0xffffu));
        }
        else {
            if (softmax) rmem[k] = softmax_mul(rmem[k], smd);
            topk_weights[(((uint64_t(blockIdx.x) << 3) + threadIdx.z) * K) + (uint64_t)(k)] = u162bf16(uint16_t(rmem[k] >> 16));
        }
    }
}
