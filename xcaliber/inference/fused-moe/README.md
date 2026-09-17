# xCalibur: SuperSonicMoE
aim: reach sol perf
---
> The MoE operator can be defined through the following sub-operators.

1. `topk: router_logits o (N, E):(1, N) -> topk_idx o (N, K):(K, 1), topk_weights o (N, K):(K, 1)`

    computes the indices of top `K` experts `E` per token `N`; 
    based of an activation over `router_logits` (`sigmoid`, `softmax`);
    optionally sums (if given) `e_correction_bias` before `argmax`.

    topk operation itself is recursive.

    * each thread can hold at least 2 elements (max) -> $\frac{n^2}{2} \times \cdots \times \frac{(n-k-1)^2}{2}$ `for-each topk`
    * quicksort-esq

    CTA 256 $\rightarrow$ `(8, 4, 8)`

---

2. [gather/permute]
3. topk -> 1hot
4. [quantize]
4. ff1, megatron trick, act, [quantize], [scatter]
5. ff2, reduce

optional: topk_idx2crd/f2

arch
1. sm89: bf16
2. sm120: bf16, fp8, nvf4