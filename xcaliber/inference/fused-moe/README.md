xCalibur: SuperSonicMoE

> Background story: The SuperSonicMoE operator aims to be the spiritual successor to SonicMoE. However, the operator strives to be a successor in name only, as (much like a problem child) it tends to disagree with its ancestor at nearly every design decision (mostly). Jokes aside, xCalibur has nothing but the utmost respect for SonicMoE. Our only hope is that we live up to the name.

## Problem statement:

    Mixture of experts (MoE), make up ~90% of LLMs (this is before skynet, ofc., 2026). However, relative to other layers that make up our baby terminators, MoE has had little love by the GPU kernel ninjas. Primarily because it's hella boring or maybe it's because the kages don't care as much. Eitherway, it's a critical bottleneck, perfect for xCalibur to give it a go.

    Ok, now that we've set the stage, it's time for us to lock-in.

## Formulation:

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
4. ff1, megatron trick, act, [quantize], [scatter]
5. ff2, reduce

optional: topk_idx2crd/f2

arch
1. sm89: bf16
2. sm120: bf16, fp8, nvf4