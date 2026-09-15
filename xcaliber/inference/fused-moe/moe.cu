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
    constexpr int T1 = ((((int)(E * D1)) >> 5));
    cg::thread_block cta = cg::this_thread_block();
    const int64_t tid = cta.thread_rank();
    const dim3 tidC = cta.thread_index(); // coordinates
    const uint64_t e_offset = (uint64_t)(((tidC.y + tidC.z) << 3));
    const uint64_t offset = (uint64_t)(blockIdx.x * tidC.x * E) + e_offset);
    float rA[64];
    float rW  = 0.0f;
    uint2 local_topk[K];
    uint2 local_minmax[2] = {
                                make_uint2(0xffff'ffffu, 0u),
                                make_uint2(0u, 0u)
                            };
    // warp-level pre-emption: acc to N
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
                    local_topk[j].y = (uint32_t)(e_offset + j);
                    if (local_minmax[0].x < local_topk[j].x){
                        local_minmax[0] = local_topk[j];
                    }
                    
                    if (local_minmax[1].x > local_topk[j].x){
                        local_minmax[1] = local_topk[j];
                    }
                }
                else {
                    if (local_minmax[1].x < __float_as_uint(rA[j])){
                        local_topk[local_minmax[1].y - e_offset].x = __float_as_uint(rA[j]);
                        local_topk[local_minmax[1].y - e_offset].y = (uint32_t)(e_offset + j);
                        local_minmax[1] = local_minmax[0];
                        for (int k = 0; k < K; k++) {
                            if (local_minmax[1].x > local_topk[k].x){
                                local_minmax[1] = local_topk[k];
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
                ".reg .b32 tmp;\n\t"
                "ld.global.acquire.gpu.v8.b32 tmp, [%1];\n\t"
                : "=r"((uint32_t)(rA + i))
                : "l"((uint64_t)__cvta_global_to_generic(router_logits + offset + i))
                )
            );
        }
    }
    uint2 tmp;
    #pragma unroll 2
    for (int i = 0; i < 2; i++) {
        #pragma unroll 2
        for (int j = 1; j <= 2; j <<= 1) {
            tmp.x = __shfl_xor_sync(0xffff'ffffu, local_minmax[i].x, j, 4);
            tmp.y = __shfl_xor_sync(0xffff'ffffu, local_minmax[i].y, j, 4);
            if ((!i) && local_minmax[i].x < tmp.x) {
                local_minmax[i] = tmp;
            }
            if (i && local_minmax[i].x > tmp.x) {
                local_minmax[i] = tmp;
            }
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 1; j <= 2; j <<= 1) {
            tmp.x = __shfl_xor_sync(0xffff'ffffu, local_topk[i].x, j, 4);
            tmp.y = __shfl_xor_sync(0xffff'ffffu, local_topk[i].y, j, 4);
            if (local_minmax[1].x > tmp.x){
                local_topk[i] = tmp[k];
                for (int k = 0; k <= i; k++) {
                    if (local_minmax[1].x > local_topk[k].x){
                        local_minmax[1] = local_topk[k];
                    }
                    if (local_minmax[0].x < local_topk[k].x){
                        local_minmax[0] = local_topk[k];
                    }
                }
            }
        }
    }
}