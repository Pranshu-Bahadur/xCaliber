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
    uint32_t rmem[32];
    for (int i = 0; i < (E >> 10); i++) { //the 10 needs to be mutable
        int KP = (int)((K + 1)/2);
        if (i) { // mutable ^
            for (int j = (i-4); j < 4 && (i-4) < KP; j++) { //the 4 needs to be mutable
                if (softmax) {
                    rmem[j] = softmax_bf16x2(rmem[j]);
                    rmem[31] = add_bf16x2x1(rmem[j], rmem[31]);
                }
                else {
                    rmem[j] = softmax_bf16x2(rmem[j] ^ 0x10001000u);
                    rmem[j] = add_bf16x2(rmem[j], make_uint2(0x1u, 0x1u));
                    rmem[j] = rcp_bf16x2(rmem[j]);
                }
                if (rmem[j] && j < K/2) {

                }
            }
        }
        ldcg_b32v4(
            (uint64_t)__cvta_generic_to_global(
                (uint64_t)(
                    router_logits
                    + (((blockIdx.x << 3) + (threadIdx.x)) * E)
                    + (threadIdx.y << 2) + (i << 6)
                )
            ),
            rmem[i << 2]
        );
    }
}