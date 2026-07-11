# xCaliber End-to-End Co-Design

Team board from router output to reduced MoE output.

```text
hardware      RTX PRO 6000 Blackwell Server Edition / sm_120a
live body     kernel.cu
deep board    notes.md
evidence      experiments/README.md
```

## Final Call

```text
model router
    -> parallel activation + routing preambles
    -> CTA1024 expert grid
    -> fused FF1 -> SwiGLU -> topk_W -> FF2
    -> reduce expert partials
```

Target path:

```text
BF16 X -> global X scale -> X4/SX once -> W4A4 FF1
                                         -> BF16 SwiGLU
                                         -> topk_W
                                         -> W4A16 FF2
```

Compress each activation once before expert dispatch. Reuse it across TOPK=8
experts, W1/W3, and every `I128'` tile. Do not recompress inside each expert.
The BF16-X/WDEQUANT path stays as the like-for-like comparison until the first
complete numerical micro-tile settles the consumer choice.

## Math

For token `t`, router slot `k`, and `e = topk_idx[t,k]`:

```text
D1[t,e] = FF1_W1_e(X[t])
D3[t,e] = FF1_W3_e(X[t])
A [t,e] = silu(D1[t,e]) * D3[t,e]
R [t,e] = topk_W[t,k] * A[t,e]
P [t,e] = FF2_W2_e(R[t,e])

Y[t] = sum selected-expert P[t,e]
```

The fused boundary is `FF1 -> SwiGLU -> topk_W -> FF2`. Router and input
compression are preambles. Final expert reduction ownership remains open.

## CUDA Graph

```text
router logits                              X[N,H] BF16
     |                                          |
     v                                          v
framework select_experts                 GRAPH A: ACT
     |                                    global absmax
     |                                          |
topk_idx + topk_W                         XGS / XGSINV
     |                                          |
     v                                    BF16 -> X4/SX
GRAPH B: ROUTE                                  |
topk_idx -> Xb                                  |
     |                                          |
     +---------------- graph join --------------+
                            |
                            v
                  E x CTA1024 expert body
                            |
             FF1 -> SwiGLU -> topk_W -> FF2
                            |
                            v
                       reduce -> Y
```

Graph A and Graph B are independent after their inputs are ready. Capture them
on separate streams/child graphs and join before the expert grid. A half-GPU
launch budget per branch is the first experiment, not a fixed SM partition.

## Router Contract

The framework owns model-specific router semantics. xCaliber consumes:

```text
topk_idx [N,TOPK]      selected expert IDs
topk_W   [N,TOPK]      matching router weights
TOPK     8             N*8 live token/expert memberships
```

This keeps softmax/sigmoid, precision, correction bias, and selection policy
out of the expert kernel.

## Graph A: Activation Preamble

One scale domain covers all `X[N,H]`, not one expert:

```text
X_absmax = max(abs(X[t,h])) for all t,h
Q4MAX    = 6
S8MAX    = 448
XGSINV   = Q4MAX * S8MAX / X_absmax
z        = X * XGSINV
```

Then compress each token/K16 block:

```text
a_j  = max(abs(z_j))
SX_j = UE4M3_encode(a_j / Q4MAX)
X4_j = E2M1_encode(z_j / SX_j)

Xhat_j = X4_j * SX_j / XGSINV
```

Graph nodes and checkpoint:

```text
A0  X -> X_absmax, XGS, XGSINV
A1  X + XGSINV -> X4 + SX

X4[kt,n8,br,l] u32
SX[kt,n8,g]     u32

token / K64     BF16 128B -> X4 32B + SX 4B
compression     3.56x
```

This removes repeated absmax/packing from every routed expert. Zero input,
UE4M3 round-up, saturation, and exact checkpoint numerics remain open.

## Graph B: Routing Preamble

Convert token-major `topk_idx` into expert-major bitplanes:

```text
Xb[e][0]       routed count / empty sentinel
Xb[e][1+w]     token-membership word
w              token >> 5
bit            token & 31
stride         1 + ceil(N/32) words / expert
```

```text
for token t
  for k in 0..7
    e = topk_idx[t,k]
    Xb[e][1 + (t>>5)] |= 1u << (t&31)
    Xb[e][0]++
```

Xb preserves literal token positions; no token compaction. `Xb[e][0] == 0`
lets an expert CTA return before touching weights or activations.

Xb membership does not identify router slot `k`. Before wiring `topk_W`, lock
one route:

```text
expert/token -> slot sidecar
slot bits beside Xb
TOPK=8 scan of topk_idx
```

## Expert Grid

```text
blockIdx.x     expert e
grid           E expert CTAs
CTA            1024 threads / 32 warps
residency      1 CTA / SM
cluster        1 CTA / 1 SM / 1 expert

188 SM, E=384  at most 3 scheduling waves
empty expert   return; next queued CTA takes the SM
```

One CTA is expert-stationary while traversing routed tokens and `I128'` tiles.

```text
8 x 128T cohorts     W13/S13 ownership
4 x 256T bands       same n256 tokens, different K64 stripes
32 warps             future M16xN8 consumers
```

There is no RR fabric.

## Fused Expert Body

Target tile lifetime:

```text
for I128' tile
  for kt in H/K64
    load W13/S13 for paired W1 + W3
    load routed X4/SX
    accumulate D1 + D3

  A = silu(D1) * D3
  R = topk_W[token,slot] * A
  feed R directly to matching W2/FF2 tile
  publish weighted Y partial
```

`D1`, `D3`, SwiGLU, and routed `R` stay on chip. No full `[N,I]` intermediate
is written to global memory.

### FF1

```text
W13            fused W1/W3 packed E2M1
S13            UE4M3 scales; one packet per 16 W packets
W13GS[e,0/1]   W1/W3 global factors
X4/SX          packed activation + four K16 scales
MMA            native dense block-scaled W4A4 K64
post factor    W13GS / XGSINV
```

W1/W3 share output coordinates so SwiGLU does not need a global transpose.
W13/S13 plane semantics and final `kt`/issue-bundle bases remain open.

### FF2

```text
input           BF16 topk-weighted SwiGLU tile
stored weight   W2 in 4-bit checkpoint form
logical path    W4A16 down projection
implementation  stream/dequant W2 for the selected MMA
```

W2/S2/W2GS layout, CTA ownership, accumulator residency, and expert reduction
are not in `kernel.cu` yet. They need their own board; W13 cannot be inherited
silently because FF2 reverses the projection direction.

## Live vs Target

The graph above is the co-design target. Current `kernel.cu` is the expert-body
substrate and still accepts BF16 `X`; X4/SX and W2-family arguments are absent.

```text
LIVE
  CTA1024 / expert
  W13 + S13 + W13GS inputs
  BF16 X + XGSINV + Xb + topk_W inputs
  empty-expert exit
  weight + activation producer draft

NOT LIVE
  global X4/SX preamble
  complete FF1 MMA consumer
  SwiGLU + slot-resolved topk_W
  W2/FF2
  output reduction
```

## Memory Call

On BF16 scattered activation rows, current-panel L2 warming followed by TMA to
SMEM beat no-prefetch TMA by `4.41%` at N256 and `6.05%` at N512.

```text
selected BF16 method   pf32x4_current_tma_smem
baseline               no_pf_tma_smem
explicit PF wait       none
```

That confirms L2 absorption. X4/SX is a 36B payload, so its final prefetch and
transfer geometry must be remeasured instead of copied from the BF16 path.

## Calls

```text
LOCKED
  framework router boundary
  TOPK -> expert-major Xb control plane
  one tensor-global X scale, not per expert
  CTA1024 / expert / SM scheduling
  dense K64 W13/S13 physical bytes
  FF1 -> SwiGLU -> topk_W -> FF2 order

SELECTED
  parallel ACT + ROUTE preambles
  compress X once globally to X4/SX
  native W4A4 FF1 -> W4A16 FF2 direction
  L2 absorption for scattered activation traffic

COMPARE
  BF16 X + on-the-fly WDEQUANT on the same routed tile

OPEN
  Xb -> topk slot recovery
  XGS zero/round/saturation convention
  W13/S13 semantic indexing
  X4/SX producer + consumer physical layout
  W2/S2/W2GS checkpoint layout
  fused I128' -> FF2 on-chip schedule
  cross-expert reduction
  numerical and end-to-end performance proof
```

## Next Proof

One vertical slice through every boundary:

```text
N8 / TOPK=8 routing fixture
topk_idx -> Xb + slot recovery
global XGSINV -> one X4/SX K64 tile
one expert / one I128' W1+W3 tile
SwiGLU -> topk_W
one W2/FF2 output tile
reference-matched weighted partial
```

Measure numerical error, registers/spills, SMEM/TMEM, L2/DRAM sectors, and
end-to-end time before widening.

## Docs

```text
co-design.md             shared end-to-end board
notes.md                 layouts, fragment math, bank maps, decisions
experiments/README.md     benchmark contract and evidence
kernel.cu                live expert-body substrate
ptx.cuh                  reviewed PTX helpers
```

```text
LOCKED       implementation contract
CONFIRMED    measured or numerically proven
SELECTED     current direction
OPEN         unresolved before completion
```
