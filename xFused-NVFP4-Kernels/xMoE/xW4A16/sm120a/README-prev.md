# xCalibrr W4A16 MoE

## variants

```text
variant   checkpoint weight layout                 role
xR64      conventional HF / ModelOpt NVFP4         adoption path
xR58      customized xCalibrr MMA-native packets   performance path
```

`xR64` reads conventional packed E2M1 weights, K16 E4M3 scales, and FP32
global scales. It keeps existing HF/ModelOpt checkpoint layouts and assembles
the MMA fragment in registers. No xCalibrr weight repack beyond ordinary
loader-side expert stacking is required.

`xR58` stores W13/S13 and W2/S2 in the exact packets consumed by the CTA.
That is the measured winner: up to `3.449x` faster than the earlier W4A4 path
and `84.02%` measured end-to-end GPU utilization on Kimi N256 uniform.
The custom checkpoint loader/converter and framework integration remain WIP.

Both are real xCalibrr kernels. They share TOPK routing, expert-contiguous
activation compression, native NVFP4 FF1/FF2, and direct token reduction.
`xR64` is the compatibility lane; `xR58` is the full co-designed speed lane.
"Conventional" refers to the persistent checkpoint weights; transient X4/Sx
and Y4/SY stay operator-native in both variants because xCalibrr produces and
consumes them internally.
The published speedup is xR58 versus the earlier W4A4 path, not xR58 versus
xR64. A matched xR64/xR58 sweep is still required for that claim.

## xR58 final call

Expert-contiguous activation compression + contiguous FF1 + direct FF2
`cp.reduce` is the current winner.

- router-wired board: `126 / 126 PASS` (`63 softmax + 63 sigmoid`)
- top-k: `29 regs`, `0 spills`; swept indices and BF16 weights are exact
- router tax vs stored precomputed-top-k sweep: `+1.07% softmax`,
  `+1.27% sigmoid` geometric mean
- Kimi fixed-seed vs previous W4A4 path: `3.449x / 1.528x / 2.145x`
  for `uniform / zipf / burst`
- Kimi N256 uniform: `84.02%` measured end-to-end GPU utilization
- FF1: `56 regs`, `0 spills`
- FF2: `64 regs`, `0 spills`
- FF2 still reduces straight into token-major `Y` with
  `cp.reduce.async.bulk.global.shared::cta...add.noftz.bf16`

## board

```text
finite BF16 logits[N,E]
                 |
                 v
moe_topk_bf16<softmax | sigmoid>
  -> topk_idx i32[N,TOPK] + topk_W BF16[N,TOPK]
                 |
                 v
one CTA / expert: deterministic rank p, no global atomic
  topk_off[token,k]          = p
  expert_topk_W[e,p]         = topk_W[token,k]
  expert_token_idx[e,0]      = count
  expert_token_idx[e,1 + p]  = token
                 |
                 v
X BF16 -> global absmax -> NVFP4 scatter
  X4[e,n16,H64,q8,lp4,{n8a_x2,n8b_x2}]
  Sx[e,n32,H64,q8,{a0,a1,b0,b1}]
                 |
                 v
FF1: adjacent packed n16 pairs, no activation PF
                 |
                 v
Y4/SY -> FF2 -> shared issue tiles
                 |
                 v
expert_token_idx[e,1+p] -> token -> cp.reduce -> Y[token,H]
```

`NP = round_up(N, 32)`. X4 is n16-packed, but Sx/SY are n32 scale-quad
layouts; NP16 corrupts the expert stride when `N < 32`.

## router

```text
CTA256 = 8 tokens                  warp w = token 8*block + w
lane l = {2l, 2l+1} + 64j         one u32 dead mask / lane

BF16 raw -> ordered bits -> {ordered, 0xffff-expert}
                                |
                                v
                       redux.sync.max.u32
                                |
                select rank -> set dead bit -> next rank
```

Softmax and sigmoid preserve logit order, so the rank path stays integer.
Only the selected `TOPK` logits enter `ex2.approx` / `rcp.approx`; selected
weights are normalized to sum to `routed_scale`. Lower expert wins equal-logit
ties. Current contract is even `E <= 1024`, `TOPK <= min(E,8)`.

```text
gate       cases    topk errors   max BF16 |dw|   e2e tax vs precomputed
softmax    63/63         0           0.0                +1.07%
sigmoid    63/63         0           0.0                +1.27%
```

This tax is the geometric mean of matched model / N / routing / seed rows. It
is an end-to-end comparison against the stored sweep, not isolated top-k
latency. Sigmoid is `0.20%` slower than softmax geometrically.

## kimi fixed seed

```text
route     direct ms   previous ms   speedup
uniform    7.039680     24.277664     3.449x
zipf      17.884800     27.326048     1.528x
burst      9.856512     21.141216     2.145x
```

## precomputed top-k baseline

This historical 63-case table predates the integrated logits -> top-k node.
Each cell is `uniform / zipf / burst` end-to-end milliseconds, 20 iterations.

```text
model               N=8                         N=256                       N=512
kimi_k2             1.1226 / 1.0092 / 0.9337   6.8839 / 10.7387 / 7.0850  7.0271 / 17.5898 / 10.5931
qwen3_30b_a3b       0.1310 / 0.1234 / 0.1203   0.2416 / 0.9774 / 0.6512   0.3724 / 1.7180 / 0.8894
qwen3_235b_a22b     0.4771 / 0.4307 / 0.4156   0.9414 / 3.5225 / 2.1369   1.4231 / 6.7291 / 3.6542
minimax_m2          0.3917 / 0.3344 / 0.3193   1.4089 / 2.9671 / 2.1820   1.4476 / 5.3662 / 2.9113
gpt_oss_120b        0.5672 / 0.5569 / 0.5508   1.2611 / 3.4047 / 2.1636   1.2822 / 6.9415 / 2.9609
gpt_oss_20b         0.5529 / 0.5485 / 0.5487   1.2189 / 3.9673 / 1.9064   1.9318 / 7.3018 / 4.1153
gemma4_26b_a4b      0.1759 / 0.1597 / 0.1576   0.3261 / 1.2593 / 0.7391   0.5017 / 2.3151 / 1.3791
```

## BF16 output delta

This is the emitted BF16 tensor delta versus the previous W4A4 path, not an
"error" against a full-precision oracle.

```text
route     bit-exact    mean abs delta   max abs delta   max rel delta
uniform    76.4546%       0.0130605          0.5          1.5748%
zipf       89.8793%       0.0074010          2.0          1.4151%
burst      71.7215%       0.0168524          1.5          1.7241%
```

The delta is in the same range as the earlier async BF16 reduction variants.
Route/offset mismatches, nonfinite outputs, and padded-token writes are all
zero across the full sweep.

## xR58 GPU utilization

`overall_gpu_util_pct` is the kernel-duration-weighted Nsight Compute
`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` value across
the six compute kernels in one captured graph launch. It is a Speed-of-Light
throughput metric, not an `nvidia-smi` busy sample. The async `Y` memset is not
an NCU kernel and is excluded from the weighting.

```text
Kimi          N      route     overall GPU   DRAM      SM active   tensor pipe
headline     256     uniform      84.02%      83.96%      87.21%       7.15%
scale        512     uniform      82.57%      82.45%      85.72%       7.19%
skew         512     zipf         29.23%      29.16%      29.48%       2.99%
skew         512     burst        46.41%      46.31%      48.46%       4.70%
```

All 63 cases are DRAM-bound by the measured SoL counters. Across the full
model mix, mean overall utilization is `64.12% / 24.91% / 31.66%` for
`uniform / zipf / burst`. Balanced large shapes reach the 80s; skew leaves a
hot-expert tail, while low-expert-count shapes cannot occupy all 188 SMs.

## files

- `topk.cuh`: warp-local BF16 softmax / sigmoid top-k
- `preamble.cuh`: deterministic packing + expert-contiguous NVFP4 scatter
- `xR58FF1FF2/`: customized-layout FF1 + FF2 speed path
- `xR64FF1FF2/`: conventional-layout HF/ModelOpt-compatible FF1 + FF2
- `benchmarks/benchmark.cu`: one-graph logits -> top-k -> MoE harness
- `benchmarks/results_topk_full.csv`: all 126 router-wired xR58 rows
- `benchmarks/results_gpu_util.csv`: 63 xR58 utilization rows
- `benchmarks/results_gpu_util_kernels.csv`: 378 raw per-kernel rows
- `benchmarks/results_models.csv`: historical 63-row precomputed-top-k baseline

```bash
./benchmark E N I H TOPK {uniform|zipf|burst} seed iters output.csv \
  [softmax|sigmoid]
```
