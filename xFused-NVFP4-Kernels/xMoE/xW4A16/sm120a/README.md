# xCaliber xR64: Fused MoE NVFP4 Co-Design for sm120x

MoE layers can constitute the majority of LLM weights, and efficient sparse MoE inference remains an open optimization problem.

xR64 attempts to contribute to this space by co-designing a sparse MoE operator specifically for **sm120x** and **NVFP4 weights**.

Rather than treating each GEMM or routing stage independently, we define the operator as a sequence of cooperating sub-kernels whose layouts, representations, and hand-off contracts are designed together.

> **Primary target:** production-grade inference on consumer/prosumer Blackwell architectures.

---

## 1. Motivation

Our target introduces two additional constraints beyond the usual sparse-MoE problems.

### 1.1 Architecture: sm120x

sm120x does not support `tcgen05` and consequently does not expose Tensor Memory despite belonging to the Blackwell family.

`wgmma` support was also discontinued for sm1xx.

Therefore, neither the Hopper `wgmma` paradigm nor the Blackwell `tcgen05` paradigm is directly available. In practice, sm120 kernels often need to reuse or adapt Ampere-style execution strategies.

This makes sm120x comparatively niche and lower-ROI for specialized kernel development.

### 1.2 NVFP4

NVFP4 is not backward-compatible with older MMA paths.

Existing Ampere-era kernels can regain compatibility by dequantizing NVFP4 weights to BF16 before compute. Marlin-style approaches are an example of this general strategy.

However, doing so leaves substantial compute throughput unused because native NVFP4 MMA throughput is significantly higher than 16-bit MMA throughput.

xR64 therefore targets **native NVFP4 execution rather than NVFP4-as-storage-only**.

---

# 2. Operator Overview

The current xR64 formulation is:

```text
router / top-k
    │
    │  [currently deferred from core scope]
    ▼

K1 — Activation Quantization
    BF16 activations
        ↓
    absmax quantization
        ↓
    NVFP4 + scales

        │
        ▼

K2 — Token / Expert Permutation
    reorder routed activations
        ↓
    downstream-coalesced representation

        │
        ▼

FF1 — Gate + Up Projection
    NVFP4 MMA
        ↓
    SwiGLU / GELU
        ↓
    routing-weight multiply
        ↓
    quantize + stage directly
    in FF2-consumable representation

        │
        ▼

FF2 — Down Projection
    NVFP4 MMA
        ↓
    routed-token reduction
        ↓
    output
```

The important design principle is that these stages are **not independent kernels**.

Each producer should emit data in the representation naturally required by the next consumer, minimizing runtime transforms and unnecessary intermediate traffic.

---

# 3. Notation

### `N` — Number of Tokens

```text
type: uint32_t
```

Typical inference/decode values:

```text
8, 32, 64, ...
```

Training or prefill workloads may instead contain:

```text
16384, 32768, ...
```

Our primary focus is inference.

### `E` — Number of Experts

```text
type: uint16_t
```

Typical values:

```text
128, 256, ...
```

---

# 4. Current xR64 Pipeline

## K1 — Activation Quantization

Convert incoming BF16 activations into native NVFP4 representation.

```text
BF16
  ↓
absmax
  ↓
NVFP4 + block scales
```

Why this exists as a preamble rather than being repeated inside FF1:

* quantization work is paid once,
* FF1 consumes native NVFP4 directly,
* representation can be designed around downstream MMA consumption,
* the output can immediately feed the permutation stage.

---

## K2 — Scatter / Permutation

Permute routed activations into a representation suitable for downstream expert computation.

Primary objective:

> Convert irregular routed-token access into a representation that FF1 can consume with substantially cleaner/coalesced memory behavior.

Current formulation:

```text
quantize
   ↓
permute by routed expert
   ↓
FF1-ready activation representation
```

This replaces the previous one-hot/top-k expansion formulation.

### Previous formulation

An earlier xR64 version used:

```text
topk_idx
   ↓
one-hot encode / expand
   ↓
expert-oriented representation
```

The current formulation is simpler for inference because NVFP4 staging already creates an explicit intermediate representation.

The permutation can therefore happen directly during activation staging.

This is also consistent with approaches explored by other sparse-MoE implementations.

---

# 5. FF1 — Gate + Up Projection

FF1 performs:

```text
Gate projection
Up projection
      ↓
SwiGLU / GELU
      ↓
routing-weight multiply
      ↓
quantize / stage for FF2
```

The routing weight can be applied before FF2 rather than requiring a separate downstream operation.

Reference:

[arXiv:2603.07685](https://arxiv.org/pdf/2603.07685)

The intended FF1 output is not merely an activation tensor.

It should be emitted directly in the **packet/layout grammar expected by FF2**.

```text
FF1 compute
    ↓
activation
    ↓
routing weight
    ↓
absmax / NVFP4 quantization
    ↓
FF2-native staged representation
```

This producer → consumer contract is a central xR64 co-design principle.

---

# 6. FF2 — Down Projection

FF2 consumes the staged NVFP4 representation produced by FF1.

```text
FF1 staged NVFP4
        ↓
Down projection
        ↓
routed-token reduction
        ↓
output
```

FF2 therefore should not need to reconstruct or heavily transform FF1 output before compute.

---

# 7. Launch / Fusion Model

xR64 currently treats the overall MoE operation as a **sequence of cooperating kernel launches**, with aggressive fusion inside each stage.

Potential structure:

```text
K1 quantize
   ↓
K2 permute
   ↓
FF1
   ↓
FF2
```

Programmatic Dependent Launch (PDL) may allow some of these stages to be chained with reduced launch overhead.

Prime Intellect appears to use a more aggressively fused FF1/FF2 formulation and should be reviewed as a reference.

### TODO

* evaluate PDL for K1 → K2 → FF1 → FF2 dependencies,
* inspect Prime Intellect FF1/FF2 launch structure,
* determine whether FF1 + FF2 should remain separate logical kernels or become one launch.

---

# 8. Weight-Movement Strategy

Earlier experiments evaluated explicit L2 prefetching for weights.

Observed result:

> Explicit L2 prefetch did not produce enough improvement to justify the additional complexity.

The current intuition is therefore simpler:

```text
activations:
    explicitly permute because routed access is irregular

weights:
    tolerate/interleave expert-weight loads
    unless measurement demonstrates a real bottleneck
```

If scattered activation loads were already manageable after permutation, irregular expert-weight traffic may likewise be acceptable without an explicit prefetch subsystem.

This needs to be benchmarked rather than assumed.

### Open lane

Evaluate:

* direct global loads,
* interleaved expert-weight loads,
* cache behavior,
* lightweight prefetching,
* whether CuTe-generated tiling changes the tradeoff.

---

# 9. Implementation Direction

The current implementation direction is likely:

```text
Colfax-style native NVFP4 building blocks
                +
           CuTe DSL
                +
      xR64 representation contracts
```

Rather than continuing to maintain increasingly bespoke handwritten machinery, CuTe DSL may provide a cleaner way to express and compose the underlying layouts.

The goal is not to abandon the xR64 representation model.

The goal is to express it more systematically.

---

# 10. Deferred Scope

These are intentionally **not required for the first integration pass**.

## K0 — Top-K Router

### Inputs

```text
router_logits: BF16 or FP32
shape:         (N, E)

TOPK: uint32_t
assumption: power of 2
```

Optional:

```text
EXPERT_SCALE: BF16 or FP32
ROUTED_SCALE: BF16 or FP32
```

### Outputs

```text
topk_idx: uint32_t
```

### Modes

```text
SIGMOID
SOFTMAX
```

### TODO

* redesign current top-k implementation,
* support routing modes cleanly,
* evaluate parallelism / launch overlap,
* evaluate stream usage where appropriate.

---

## Dual-Sparse MoE

Future extension.

```text
TODO: define dual-sparse formulation
```

---

## HexLift-4 / Low-Bit FF2 Decomposition

Potential FF2 research direction.

The idea is to decompose FP32/BF16 values into multiple low-bit components and recover the result through multiple 4-bit MMA contributions.

Related implementation exists in the previous GPUMode Cholesky work.

Reference:

[GPUMode submission / leaderboard](https://www.gpumode.com/leaderboard/776?tab=rankings)

Possible decomposition basis:

```text
s2f6
```

This is deferred until the baseline native-NVFP4 pipeline is locked.

---

# 11. Historical Design Decisions

## One-Hot Expansion

Previous:

```text
topk_idx
   ↓
one-hot encode
   ↓
expand
   ↓
FF1
```

Current:

```text
BF16 activation
   ↓
NVFP4 quantize
   ↓
expert permutation
   ↓
FF1
```

The new formulation removes unnecessary intermediate structure and fits the inference pipeline more naturally.

---

## Explicit L2 Prefetch

Previously explored for expert weights.

Result:

```text
No sufficiently large gain to justify complexity.
```

Current default:

```text
direct / naturally cached weight traffic
```

Revisit only if benchmarks show weight movement becoming dominant.

---

# 12. Current Team Attack Surface

The current formulation intentionally exposes several independent optimization lanes.

### Benchmarking

Measure:

```text
K1
K2
FF1
FF2
end-to-end
```

and establish the cost/benefit of each representation transition.

### CuTe DSL

Express the native NVFP4 FF1/FF2 layouts and producer-consumer contracts cleanly.

### Weight Movement

Characterize expert-weight traffic and determine whether explicit caching/prefetch is worthwhile.

### Launch Chaining

Evaluate PDL and other mechanisms for reducing launch overhead between dependent stages.

### Top-K

Deferred until the core NVFP4 MoE path is stable.

---

# 13. Architecture Reference

Target reference platform: RTX PRO 6000 Blackwell / sm120.

```text
Memory:                    96 GB GDDR7
Memory bandwidth:          ~1.6 TB/s
CUDA cores:                24,064
Streaming Multiprocessors: 188
Graphics Processing
Clusters:                  12
Tensor Cores:              752
                            (4 / SM)

L1 cache:                  128 KB / SM
L2 cache:                  128 MB

Peak FP4 Tensor TFLOP/s
with FP32 accumulate:       2015.2

Maximum SM clock:          2.43 GHz
```

Reference:

[Colfax Research — Optimizing an NVFP4 Blockscaled GEMM on RTX PRO 6000 Blackwell GPU (SM120)](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)

### TODO

Check which assumptions and architectural characteristics transfer directly to RTX 5090 / other sm120 consumer parts.

---

# 14. References / TODO Reading

* [Colfax — Optimizing an NVFP4 Blockscaled GEMM on RTX PRO 6000 Blackwell GPU](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/)
* [arXiv:2603.07685](https://arxiv.org/pdf/2603.07685)
* Prime Intellect fused MoE kernels
* MonoMoE / related absmax quantization work
* Team Wombat MLSys work
* CuTe DSL native NVFP4 examples

---

# 15. Immediate TODO

* [ ] Lock the K1 → K2 → FF1 → FF2 contract.
* [ ] Define exact tensor/layout shapes at every boundary.
* [ ] Document FF1 → FF2 staged NVFP4 representation.
* [ ] Establish benchmark harness and baseline.
* [ ] Evaluate CuTe DSL implementation direction.
* [ ] Benchmark expert-weight access before adding prefetch complexity.
* [ ] Review PDL for dependent-stage launch chaining.
* [ ] Add Prime Intellect fused kernels as reference.
* [ ] Add absmax / MonoMoE justification.
* [ ] Verify RTX 5090 architectural assumptions.
* [ ] Defer router/top-k redesign until the core path is locked.
