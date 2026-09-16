#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <cfloat>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <type_traits>
namespace cg = cooperative_groups;
/*

    Computes the indices of top `K` experts `E` per token `N`; 
    based of an activation over `router_logits` (`sigmoid`, `softmax`);
    optionally sums (if given) `e_correction_bias` before `argmax`.

*/
template <typename T, bool softmax>
__global__ void topk(
    const T* router_logits,
    int* topk_idx,
    __nv_bfloat16* topk_weights,
    const float* e_correction_bias = nullptr,
    const int K,
    const int N,
    const int E
){
    constexpr int D1 = (int)(sizeof(T) / 4.0f);
    const int T1 = ((((int)(E * D1)) >> 5));
    cg::thread_block cta = cg::this_thread_block();
    const int64_t tid = cta.thread_rank();
    const dim3 tidC = cta.thread_index(); // coordinates
    const uint64_t e_offset = (uint64_t)((((tidC.y << 3) + tidC.x) << 3));
    const uint64_t offset = (uint64_t)(((blockIdx.x << 3) + tidC.z) * E) + e_offset);
    float rA[64];
    float rW  = 0.0f;
    uint2 tmp;
    uint2 local_topk[K];
    uint2 global_topk[K];
    uint2 local_minmax[2] = {
            make_uint2(0u, 0u),    
            make_uint2(0xffff'ffffu, 0u)
        };
    if ((uint64_t)(((blockIdx.x << 3) + tidC.z)) > N) {
        return;
    }
    for (int i = 0; i < T1 + 8; i += 8)) {
        if (i) {   
            #pragma unroll 8
            for (int j = i-8; j < i; j++) {
                asm volatile(
                    "ex2.approx.f32 %0, %1;\n\t"
                    : "=r"((uint32_t)(rA + j))
                    : "r"((uint32_t)(rA + j))
                );
                if (softmax) {
                    rW  = rA[j] + rW;
                }
                else{
                    rA[j] = 1.0f / (1.0f + (1.0f / rA[j]));
                }
                if (j<K) {
                    local_topk[j].x = __float_as_uint(rA[j]);
                    local_topk[j].y = (uint32_t)(e_offset + ((j >> 3) << 8) + (j & 7));
                    if (local_minmax[0].x < local_topk[j].x){
                        local_minmax[0] = local_topk[j];
                    }
                    
                    if (local_minmax[1].x > local_topk[j].x){
                        local_minmax[1] = make_uint2(local_topk[j].x, (uint32_t)j);
                    }
                }
                else {
                    if (local_minmax[1].x < __float_as_uint(rA[j])){
                        local_topk[local_minmax[1].y].x = __float_as_uint(rA[j]);
                        local_topk[local_minmax[1].y].y = (uint32_t)(e_offset + ((j >> 3) << 8) + (j & 7));
                        local_minmax[1] = make_uint2(local_topk[0].x, 0u);
                        for (int k = 0; k < K; k++) {
                            if (local_minmax[1].x > local_topk[k].x){
                                local_minmax[1] = make_uint2(local_topk[k].x, (uint32_t)k);
                            }
                            if (local_minmax[0].x < local_topk[k].x){
                                local_minmax[0] = local_topk[k];
                            }
                        }
                    }
                }
            }
        }
        if (i < T1) {
            asm volatile(
                "ld.global.acquire.gpu.v8.b32 %0, [%1];\n\t"
                : "=r"((uint32_t)(rA + i))
                : "l"((uint64_t)__cvta_generic_to_global(router_logits + offset + ((uint64_t)i << 5)))
                )
            );
        }
    }
    if (softmax) {
        #pragma unroll 5
        for (int i = 16; i > 0; i >>= 1) {
            rW = rW + __shfl_xor_sync(0xffff'ffffu, rW, i)
        }
        #pragma unroll K
        for (int i = 0; i < K; i++) {
            local_topk[i].x = __float_as_uint(__uint_as_float(local_topk[i].x) / rW);
        }
    }
    #pragma unroll K
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < i; j++) {
            if (local_topk[j].x > local_topk[i].x) {
                tmp = local_topk[j];
                local_topk[j] = local_topk[i];
                local_topk[i] = tmp;
            }
        }
    }
    local_minmax[1] = local_minmax[0];
    int pivot = K;
    for (int i = K-1; i >= 0; i--) {
        if (local_minmax[0].y == local_minmax[1].y) {
            local_minmax[1] = local_topk[--pivot];
        }
        local_minmax[0] = local_minmax[1];
        for (int j = 16; j > 0; j >>= 1) {
            tmp.x = __shfl_xor_sync(0xffff'ffffu, local_minmax[0].x, j);
            tmp.y = __shfl_xor_sync(0xffff'ffffu, local_minmax[0].y, j);
            if (local_minmax[0].x < tmp.x) {
                local_minmax[0] = tmp;
            }
        }
        global_topk[i] = local_minmax[0];
    }
    if (((tid & 31) >> 4) && ((tid & 15) < K)) {
        topk_idx[(uint64_t)(((blockIdx.x << 3) + tidC.z) * K) + (uint64_t)((tid & 15))] = global_topk[(tid & 15)].y;
    }
    if ((!((tid & 31) >> 4)) && ((tid & 15) < K)) {
        topk_weights[(uint64_t)(((blockIdx.x << 3) + tidC.z) * K) + (uint64_t)((tid & 15))] = __uint_as_float(global_topk[(tid & 15)].x); //cvt to bf16
    }
}