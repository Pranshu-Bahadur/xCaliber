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
    constexpr float D1 = 4.0f / sizeof(T);
    constexpr int T1 = ((((int)(E / D1)) >> 5));
    cg::thread_block cta = cg::this_thread_block();
    const int64_t tid = cta.thread_rank();
    const dim3 tidC = cta.thread_index(); // coordinates
    const uint64_t offset = (uint64_t)(blockIdx.x * tidC.x * E) + (uint64_t)(((tidC.y + tidC.x) << 3));
    float rA[64];
    float rW  = 0.0f;
    // warp-level pre-emption: acc to N
    for (int i = 0; i < T1 + 8; i += 8)) {
        if (i){   
            #pragma unroll 8
            for (int j = i-8; j < 8; j++) {
            
                asm volatile(
                    "ex2.approx.f32 %0, %1;\n\t"
                    : "=r"((uint32_t)(rA + j))
                    : "r"((uint32_t)(rA + j))
                );

                if (softmax) {
                    rW  = rA[j] + rW;
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
}