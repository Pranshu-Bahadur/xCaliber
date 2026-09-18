template <bool softmax>
__global__ void topk_kernel(
    const __nv_bfloat16* router_logits,
    int* topk_idx,
    __nv_bfloat16* topk_weights,
    const float* e_correction_bias = nullptr, //@TODO
    const int K,
    const int N,
    const int E
){
    uint32_t rmem[32];
    for (int i = 0; i < (E >> 10); i++) { //the 10 needs to be mutable
        if (i) {
            for (int j = i-4; j < 4 && i < K; j++) { //the 4 needs to be mutable
                if (softmax) {
                    rmem[j] = softmax_bf16x2(rmem[j]);
                    rmem[31] = add_bf16x2x1(rmem[j], rmem[31]);
                }
                else {
                    rmem[j] = softmax_bf16x2(rmem[j] ^ 0x10001000u);
                    rmem[j] = add_bf16x2(rmem[j], make_uint2(0x1u, 0x1u));
                    rmem[j] = rcp_bf16x2(rmem[j]);
                }
                if (rmem[j]) {

                }
            }
        }
    }
}