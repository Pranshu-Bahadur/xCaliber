#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <cfloat>
#include <cooperative_groups.h>
#include <cute/tensor.hpp>


namespace cg = cooperative_groups;

/*

    Computes the indices of top `K` experts `E` per token `N`; 
    based of an activation over `router_logits` (`sigmoid`, `softmax`);
    optionally sums (if given) `e_correction_bias` before `argmax`.

*/

__global__ void topk(
    const T* router_logits,
    int32_t* topk_idx,
    const int32_t K,
    const int32_t N,
    const int32_t E
){

    constexpr float D1 = 4.0f / sizeof(T);
    cg::thread_block cta = cg::this_thread_block();

    const int64_t tid = cta.thread_rank();
    const dim3 tidC = cta.thread_index(); // coordinates
    
    // warp-level pre-emption: acc to N


    constexpr uint64_t offset = (uint64_t)(blockIdx.x * tidC.x * E) +
                                (uint64_t)(((tidC.y + tidC.x) << 3));
    uint32_t rA[96];

    float32_t rM[32] = {0.0f};


    for (int32_t i = 0; i < ((((int32_t)(E / D1)) >> 5) + 8); i += 8)) {


        if (i){
            //computation
            //
        }


        asm volatile(
            "ld.global.acquire.gpu.v8.b32 %0, [%1];\n\t"
            : "=r"(rA + i)
            : "l"((uint64_t)__cvta_global_to_generic(offset + i))
            )
        );


    }   
}