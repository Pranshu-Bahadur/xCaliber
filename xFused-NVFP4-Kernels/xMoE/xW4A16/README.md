# contiguous direct

## final call

Expert-contiguous activation compression + contiguous FF1 + direct FF2
`cp.reduce` is the current winner.

- full board: `63 / 63 PASS`
- Kimi fixed-seed vs previous W4A4 path: `3.449x / 1.528x / 2.145x`
  for `uniform / zipf / burst`
- FF1: `56 regs`, `0 spills`
- FF2: `64 regs`, `0 spills`
- FF2 still reduces straight into token-major `Y` with
  `cp.reduce.async.bulk.global.shared::cta...add.noftz.bf16`

## board

```text
topk_idx[N,TOPK] + topk_W[N,TOPK]
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
expert_token_idx[e,p] -> token -> cp.reduce -> Y[token,H]
```

`NP = round_up(N, 32)`. X4 is n16-packed, but Sx/SY are n32 scale-quad
layouts; NP16 corrupts the expert stride when `N < 32`.

## kimi fixed seed

```text
route     direct ms   previous ms   speedup
uniform    7.039680     24.277664     3.449x
zipf      17.884800     27.326048     1.528x
burst      9.856512     21.141216     2.145x
```

## full sweep

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

## files

- `preamble.cuh`: deterministic packing + expert-contiguous NVFP4 scatter
- `xR57F1_contiguous_graph.cu`: contiguous FF1, no activation prefetch
- `xR57F2_direct_reduce_graph.cu`: direct inverse-map `cp.reduce` FF2
- `benchmark.cu`: one-graph end-to-end harness
- `results_models.csv`: all 63 rows
- `results_models_summary.csv`: 21 model/token summaries
- `results_output_delta_vs_previous_w4a4.csv`: BF16 output deltas
