# xCalibur: SuperSonicMoE
target: sm120a, sm89

> Background story: The SuperSonicMoE operator aims to be the spiritual successor to SonicMoE. However, the operator strives to be a successor in name only, as (much like a problem child) it tends to disagree with its ancestor at nearly every design decision (mostly). Jokes aside, xCalibur has nothing but the utmost respect for SonicMoE. Our only hope is that we live up to the name.

## Problem statement:

> Mixture of experts (MoE), make up ~90% of LLMs (this is before skynet, ofc., 2026). However, relative to other layers that make up our baby terminators, MoE has had little love by the GPU kernel ninjas. Primarily because it's hella boring or maybe it's because the kages don't care as much. Eitherway, it's a critical bottleneck, perfect for xCalibur to give it a go.

Ok, now that we've set the stage, it's time for us to lock-in.

## Formulation:

> The MoE operator can be defined through the following sub-operators.

1. topk

> Computes the indices of top `K` experts `E` per token `N`; based of an activation over `router_logits` (`sigmoid`, `softmax`);

`topk : router_logits o (N, E):(1, N)` $\rightarrow$ `(topk_idx o (N, K):(1, K), topk_weights o (N, K):(1, K))`

2. gather

@TODO: add 1hot encoding (SonicMoE is right, due to L2 cache reuse) //still evaluating

this needs to be done in a way such that decoding is easy (explore cute)

