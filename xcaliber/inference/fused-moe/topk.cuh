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
    for (int i = 0; i < (E >> 10); i++) {
        if (i) {
            for (int j = 0; j < 8; i++) {
                if (softmax) {
                    rmem[j] = softmax_bf16x2(rmem[j]);
                    rmem[31] = add_bf16x2x1(rmem[j], rmem[31]);
                }
                else {
                    rmem[j] = softmax_bf16x2(rmem[j] ^ 0x10001000u);
                    //@TODO hadd2
                }
            }
        }
    }
}