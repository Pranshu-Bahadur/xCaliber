//imports
template <bool softmax>
__global__ void topk_kernel(
    const __nv_bfloat16* router_logits,
    int* topk_idx,
    __nv_bfloat16* topk_weights,
    const float* e_correction_bias = nullptr, //@TODO add sub-case in sigmoid
    const int K,
    const int N,
    const int E
){
    __nv_bfloat162 rmem[32];
    ushort ltopk_idx[16];
    if ((uint64_t)(((blockIdx.x << 3) + threadIdx.x)) > N) {
        return;
    }
    rmem[30] = make_bfloat162(
                u162bf16(0u),
                u162bf16(0u)
            );
    rmem[29] = rmem[30];
    rmem[28] = rmem[30];
    rmem[27] = rmem[30];
    
    for (int i = 0; i <= (E >> 10); i++) { //the 10 needs to be mutable
        int KP = (int)((K + 1)/2);
        if (i) { // mutable ^
            for (int j = 0; j < (i << 2) && (j << 1) < KP; j++) {
                if (j >= ((i-1) << 2)) {       
                    if (softmax) {
                        rmem[j] = softmax_bf16x2(rmem[j]);
                        if (!j) rmem[31] = 0u;
                        rmem[31] = add_bf16x2x1(rmem[j], rmem[31]);
                    }
                    if (!softmax) {
                        rmem[j] = softmax_bf16x2(__hneg2(rmem[j]));
                        rmem[j] = add_bf16x2(rmem[j], make_bfloat162(
                            u162bf16(0x1u),
                            u162bf16(0x1u))
                        );
                        rmem[j] = rcp_bf16x2(rmem[j]);
                    }
                }
                if ((j << 1) < KP) {
                    uint32_t m1 = __hlt2_mask(rmem[30], rmem[j]);
                    uint32_t m2 = __hlt2_mask(rmem[29], rmem[j]);
                    if (m1==0xffff'ffffu && m1==m2) {
                        int m3 = __hgt(rmem[j].x, rmem[j].y);
                        rmem[30].x = (m3)? rmem[j].y : rmem[j].x;
                        rmem[30].y = (m3)? rmem[j].x : rmem[j].y;
                        rmem[29].x = rmem[30].y;
                        rmem[29].y = rmem[30].x;
                        ltopk_idx[j << 1] = (uint16_t)(((m3)? (j << 1) + 1 : (j << 1)) + (threadIdx.y << 2) + (i << 6));
                        ltopk_idx[(j << 1) + 1] = (uint16_t)(((m3)? (j << 1) : (j << 1) + 1) + (threadIdx.y << 2) + (i << 6));
                    }
                    if (m1==0xffff'0000u) {
                        rmem[30].x = rmem[j].x;
                        rmem[29].y = rmem[j].x;
                        ltopk_idx[j << 1] = (uint16_t)((j << 1) + (threadIdx.y << 2) + (i << 6));;
                        if (m2==0x0000'ffffu) {
                            rmem[30].y = rmem[j].y;
                            rmem[29].x = rmem[j].y;
                            ltopk_idx[(j << 1) + 1] = (uint16_t)(((j << 1) + 1) + (threadIdx.y << 2) + (i << 6));
                        }
                    }
                    if (m2==0x0000'ffffu) {
                        rmem[30].x = rmem[j].y;
                        rmem[29].y = rmem[j].y;
                        ltopk_idx[(j << 1) + 1] = (uint16_t)(((j << 1)+1) + (threadIdx.y << 2) + (i << 6));;
                        if (m1==0xffff'0000u) {
                            rmem[30].y = rmem[j].x;
                            rmem[29].x = rmem[j].x;
                            ltopk_idx[(j << 1)] = (uint16_t)(((j << 1)) + (threadIdx.y << 2) + (i << 6));
                        }
                    }
                }
            }
        }
        if (i < (E >> 10)) {
            ldcg_b32v4(
                (uint64_t)__cvta_generic_to_global(
                    (uint64_t)(
                        router_logits
                        + (((blockIdx.x << 3) + (threadIdx.x)) * E)
                        + (threadIdx.y << 2) + (i << 6)
                    )
                ),
                (uint32_t)(&rmem[i << 2])
            );
        }
        
    }
}