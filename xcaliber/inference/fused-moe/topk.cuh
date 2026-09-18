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
    uint32_t smd = 0u;
    uint32_t rmem[32];
    if ((uint64_t)(((blockIdx.x << 3) + threadIdx.z)) > N) {
        return;
    }
    #pragma unroll 8
    for (int k = 0; k < K; k++) {
        rmem[16 + k] = 0u;
    }
    for (int i = 0; i <= (E >> 8); i++) {
        if (i) {
            for (int j = ((i-1) << 2); j < (i << 2); j++) {
                if (softmax) {
                        rmem[j] = softmax_bf16x2(rmem[j]);
                        if (!j) smd = 0u;
                        add_bf16x2x1(rmem[j], smd);
                }
                if (!softmax) {
                        rmem[j] = softmax_bf16x2(rmem[j] ^ 0x8000'8000u);
                        add_bf16x2(rmem[j], 0x0001'0001u);
                        rcp_bf16x2(rmem[j]);
                }
                uint16_t e_offset = (uint16_t)((0xffffu - ((threadIdx.x + (threadIdx.y << 3)) + ((i - 1) << 8))) + ((j & 3) << 1));
                rmem[j ^ 4] =  (rmem[j] << 16) | e_offset;
                rmem[j] =  (rmem[j] & 0xffff'0000u) | e_offset - 1u;
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
        if (i < (E >> 8)) {
            ldcg_b32v4(
                (uint64_t)__cvta_generic_to_global(
                    (uint64_t)(
                        router_logits
                        + (((blockIdx.x << 3) + (threadIdx.z)) * E)
                        + (threadIdx.x + (threadIdx.y << 3)) + (i << 8)
                    )
                ),
                &rmem[i << 2]
            );
        }
        if (softmax) {
            smd = 0xffff'0000u & smd;
            #pragma unroll 5
            for (int i = 16; i > 0; i >>= 1) {
                add_bf16x2(smd, __shfl_xor_sync(0xffff'ffffu, smd, i));
            }
            rcp_bf16x2(smd);
            smd = 0xffff'0000u & smd;
            #pragma unroll 8
            for (int i = 0; i < K; i++) {
                softmax_mul(rmem[16 + i], smd);
            }
        }
    }
}