# xCaliber xR64: Fused MoE NVFP4 Co-Design for sm120x

MoE layers constitute 90% of LLM weights. Solving for sparse MoE is still an open problem. 

We attempt to contribute to optimization efforts, by Co-designing the Sparse MoE operator for sm120. Specifically, we aim to produce an unified operator (sequence of sub-kernels) that target NVFP4 weights.

Upon the general challenges in Sparse MoE, our targeted architecture and floating-point format raises a few more.

1. ARCH: sm120x does not support tcgen05 and consequently does not have tensor-memory (despite being from the Blackwell family of NVIDIA GPUs). Furthermore wgmma support was discontinued for sm1xx, therefore neither the Hopper or tcgen05 paradigms are viable for this arch family, which makes this niche and low-roi in general. Typically, the solution here is to refurbish Ampere era kernels.

2. NVFP4: this floating point format is not backward compatible, which prevents us to "plug-and-play" Ampere era kernels (Marlin for instance dequantizes NVFP4->bf16 on the fly for backward comptiblility). This solves, the problem of compatibility, but leaves compute on the table. Mainly being 16-bit mma throughput being 3.5x slower than NVFP4 mma.

Co-design overview:

We define our Sparse MoE operator through a sequence of kernel launches - fusing multiple sub-operations for maximizing throughput.

Notation:
N: number of tokens (uint32_t) [typically for decode 8, 32, 64, etc; note: for training this can be 16384, 32768, etc]

> Our main focus is production-grade arch inference

E: number of experts in the MoE layer (uint16_t) [ex: 128, 256, etc]

FF1 Preamble:
- K1 absmax quantize activations (bf16 -> NVFP4)
- K2 scatter perm for coalesced loads downstream (mem bound mitigate)

FF1
- Gate & Up proj, SwiGLU/GELU, topk_W 
- mathy call: topk_W mul can be done before FF2 [arXiv 2603.07685](https://arxiv.org/pdf/2603.07685)
- Stage for FF2 (quantize outputs)

FF2
- Down Proj, reduce

note: PDL just opened up (i think pi does ff1ff2 in a single launch)

@TODO: change makes it super straightforward, reviewing L2 prefetch for weights (for interleaved / uncoalsced weight loads) - tradeoff intuition: the weights dont stream and if scattered act loads was good enough before, weights should be kewl 

[x] (curr-prev) K2 one-hot encode topk_indx evaluate perm expand instead: intuition FF2 uses E, N*TOPK if we're not doing topk why not just quantize -> perm for FF1? (note its been a month, so i dont recall this tradeoff); -> yeah, staging (since nvf4) is cleaner for inference (also L2 prefetch didnt cause justifiable gain) (yeah TeamWombat did this at MLSys too)


@TODO: prolly drawing upon Colfax NVFP4 + cute dsl is the better play at this point
(can't afford the registers, because CTA 1024 i think)

qn: how can we optimize for throughput and latency, while maintaining backward compatibility and portability
    

K0 topk (deferred): 
          - GIVEN:  router_logits (bf16, f32); Shape (N, E)
                    TOPK (uint32) (assumption: power of 2)
          - OPTIONAL: EXPERT_SCALE (bf16, f32)
                      ROUTED_SCALE (bf16, f32)
          - RETURNS:
                    topk_idx (uint32_t); Shape ()
          - MODES: SIGMOID, SOFTMAX
qn (deffered - reduce scope and lock-in): is there a sicker formulation? 
    @TODO [evaluate ROI for HexLift-4 (FF2) in my submission file somewhere from the cholesky gpumode comp](https://www.gpumode.com/leaderboard/776?tab=rankings)
    - this is a decomposition of f32/bf16 to multiple 4bit elements (using s2f6)



@TODO: Dual-Sparse MoE
@TODO: topk re-work and support for modes (+streams for k1, k2); current implementation is meh (can be better imo)
@TODO: ofc cute dsl support
@TODO: finish this doc for the team!



(@TODO add PrimeIntellect's fused kernels for ref) - justifying the absmax choice (monomoe (?))



The following are specfications about the arch:

```
The sm120 has the following specifications:
- 96 GB of GDDR7 memory with ~1.6 TB/s of memory bandwidth
- 24,064 CUDA cores
- 188 Streaming Multiprocessors (SMs)
- 12 Graphics Processing Clusters (GPCs)
- 752 fifth-generation Tensor Cores (4 per SM)
- L1 cache size: 128 KB/SM
- L2 cache size: 128 MB
- Peak FP4 Tensor TFLOP/s with FP32 Accumulate: 2015.2
- Max SM Clock Rate: 2.43 Ghz
```

[Ref-1: Colfax's Blog: Optimizing an NVFP4 Blockscaled GEMM on RTX PRO 6000 Blackwell GPU (SM120)](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)

Note: 5090 follows a similar specfication (@TODO check this)

