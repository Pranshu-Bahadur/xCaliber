# xCalibur: SuperSonicMoE
aim: reach sol perf
---


1. `topk: router_logits o (N, E):(1, N) -> topk_idx o (N, K):(K, 1)`
---

variations:
- [ ] bf16 [e_correction_bias]
- [ ] sigmoid
- [ ] softmax
- [ ] float32

`N>E -> (E, 1)`

---

2. [quantize]
3. [gather/permute]
4. ff1, megatron trick, act, [quantize], [scatter]
5. ff2, reduce

optional: topk_idx2crd/f2

arch
1. sm120: bf16, fp8, nvf4
