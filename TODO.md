# xCaliber TODO / kernel.cu Source Board


Current scope:

```text
CTA1024
1 CTA = 1 SM = 1 expert / cluster
ff1 W13/M13/S13 L2 prefetch + staging substrate
target consumer = 32 x m16n8k128 mma.sp
NVFP4 sparse 4:8 pairwise
consumer path not implemented yet
```

Do not drift back to the old CTA256 / I64 split board except as lineage.

## Co-Design Summary

Goal:

```text
per expert / per SM:
  build W13/M13/S13 + X4/SX substrate
  then consume post-staging as 32 x m16n8k128 mma.sp

dtype:
  NVFP4 sparse 4:8 pairwise

macro:
  ff1 W13/M13/S13 -> SwiGLU -> topk_W -> store / later ff2 path
```

CTA / expert:

```text
CTA1024 = 32 warps
1 CTA = 1 SM = 1 expert cluster
blockIdx.x = expert e

sentinel:
  Xb says whether this expert has live routed work
  empty expert -> CTA returns early
```

MMA target:

```text
32 warps -> 32 x m16n8k128 mma.sp

N=8:
  MMA n8 axis
  not TOPK

TOPK:
  routing sparsity / expert hit control
```

Sparse NVFP4 contract:

```text
W13:
  fused W1/W3 sparse weights
  [E, H128, I<<4]
  per original i/H128 = 16 u32 = 64B
  W1 32B + W3 32B

M13:
  pairwise sparse metadata sidecar
  [E, H128, I<<2]
  per original i/H128 = 4 u32 = 16B

S13:
  K/H-side NVFP4 scale sidecar
  [E, H128, I<<1]
  per original i/H128 = 2 u32 = 8B

X4/SX:
  sparse/quant activation payload + scale side
```

Panel unit:

```text
kt += 2
H256 = H128(kt+0) + H128(kt+1)

prefetch is full-I x H256 per expert/SM
staging then carves the post-loop operand substrate
```

Traffic per kt / SM:

```text
W13 = 256KB
M13 =  64KB
S13 =  32KB
WMS = 352KB
```

rr fabric:

```text
rr = lane 00..31
r  = warp rank 00..31

rr00/01 -> W13 L2 prefetch, hp kt+0 / kt+1
rr02/03 -> M13/S13 L2 prefetch, hp kt+0 / kt+1
rr04..07 + rr16..31 -> staged gmem->smem/tmem/rmem tx lanes

L2 is the rendezvous:
  producer rr lane does not need to equal tx rr lane
```

Correctness target:

```text
after staging loop:
  final substrate must be aligned for 32 x m16n8k128 mma.sp
  W13 packets + M13 metadata + S13 scales must match sparse 4:8 pairwise K=128 consumption

consumer not implemented yet:
  current work is verifying source/dst indexing, bank shape, and staging order
```

## Current Lock

- [x] `kernel.cu` is the source board.
- [x] `TODO.md` is a blackboard, not compile proof.
- [x] CTA1024 is current.
- [x] `blockIdx.x == expert e` in current addressing.
- [x] `kt` advances by H256: `for kt in 0..(H>>8), kt += 2`.
- [x] W13/M13/S13 layouts are checkpoint-native and unchanged.
- [x] Current prefetch unit is full-I x H256.
- [x] Current S13 smem tx is not the full S13 prefetch unit.

## rr32x32 Map

kernel.cu:

```cpp
cg::coalesced_group rr32x32 =
    cg::labeled_partition(cta1024, ((cta1024.thread_rank() ^ 32) & 31));
```

Since `&31` kills bit 5:

```text
tid = warp*32 + lane
label = ((tid ^ 32) & 31) = lane

rr32x32.meta_group_rank() = lane 00..31
rr32x32.thread_rank()     = rank within same-lane group = warp 00..31
```

Horizontal:

```text
rr/lane  physical tids across rank
-------  ----------------------------------------------------------
rr00     000, 032, 064, 096, ..., 992      rank00..31 = warp00..31
rr01     001, 033, 065, 097, ..., 993      rank00..31 = warp00..31
rr02     002, 034, 066, 098, ..., 994      rank00..31 = warp00..31
rr03     003, 035, 067, 099, ..., 995      rank00..31 = warp00..31
...
rr31     031, 063, 095, 127, ..., 1023     rank00..31 = warp00..31
```

Use:

```text
rr lane -> stream / phase
rr rank -> stripe id across 32 physical warps
```

## Sentinel Preemption

kernel.cu:

```text
if lane%4 == 0:
  rmem[0] = Xb[e*(8 + ((N+31)>>5)) + (lane >> 2)]

__shfl_sync(mask, rmem[0], 0, 4)
if popc(rmem[0]) == 0: return
```

Meaning:

```text
8 Xb words / expert
one word per 4-lane quad
empty expert word -> CTA-local early return
```

Open:

```text
confirm final Xb contract:
  whole expert empty? per TOPK bitplane word? per token chunk?
```

## Layouts

### W13

kernel.cu:

```text
W13: [E, (H+127)>>7, I<<4] u32
```

Per original `i` / H128:

```text
u32:   00           01           02           03           04           05           06           07
W1:    h000..015    h016..031    h032..047    h048..063    h064..079    h080..095    h096..111    h112..127

u32:   08           09           10           11           12           13           14           15
W3:    h000..015    h016..031    h032..047    h048..063    h064..079    h080..095    h096..111    h112..127
```

Packet:

```text
1 W13 packet = 1 u32 = 4B = I16
per i/W1/H128  = 8 packets  = 32B
per i/W3/H128  = 8 packets  = 32B
per i/W13/H128 = 16 packets = 64B
```

### M13

kernel.cu:

```text
M13: [E, (H+127)>>7, I<<2] u32
```

Per original `i` / H128:

```text
u32 00 = m1 h000..063
u32 01 = m1 h064..127
u32 02 = m3 h000..063
u32 03 = m3 h064..127

per i/M13/H128 = 4 packets = 16B
M13 = W13 / 4
```

### S13

kernel.cu:

```text
S13: [E, (H+127)>>7, I<<1] u32
```

Per original `i` / H128:

```text
u32 00 = s1 h000..127
u32 01 = s3 h000..127

per i/S13/H128 = 2 packets = 8B
S13 = W13 / 8
```

Important:

```text
S13 is H/K-side scale sidecar.
S13 has nothing to do with n8 structurally.
One S13 packet corresponds to 8 W13 packets for same i/proj/H128.
```

## L2 Prefetch Board

Loop:

```text
for kt = 0; kt < (H >> 8); kt += 2

hp0 = kt + 0
hp1 = kt + 1
```

### W13 Prefetch

kernel.cu:

```text
if rr < 2:
  prefetch W13 + e*(H>>7)*(I<<4) + (kt+rr)*(I<<4) + (rank<<10), 4096B
```

Horizontal:

```text
rr   hp      ranks        bytes/rank  total
--   ------  -----------  ----------  -----
00   kt+0    rank00..31   4096B       128KB
01   kt+1    rank00..31   4096B       128KB
```

Per issuing thread / rr rank:

```text
4096B = 1024 u32 packets
1 u32 packet = 4B = I16

W13 fused split:
  4096B -> I512' = (I256, I256)
          W1 side  W3 side

derived original-i view only:
  1024 packets / 16 packets per original i = i64
```

Rank map:

```text
rr00 hp=kt+0:
  r00 I512' (I256,I256) | r01 I512' (I256,I256) | ... | r31 I512' (I256,I256)

rr01 hp=kt+1:
  r00 I512' (I256,I256) | r01 I512' (I256,I256) | ... | r31 I512' (I256,I256)
```

Totals:

```text
W13 / H128 / SM = 32 ranks * 4096B = 128KB
W13 / H256 / SM = 2 * 128KB        = 256KB
```

### M13/S13 Prefetch

kernel.cu:

```text
if 2 <= rr < 4:
  hp = kt + ((rr ^ 2) & 1)

  prefetch M13 + e*(H>>7)*(I<<2) + hp*(I<<2) + (rank<<8), 1024B
  prefetch S13 + e*(H>>7)*(I<<1) + hp*(I<<1) + (rank<<7),  512B
```

Horizontal:

```text
rr   hp      ranks        M13/rank  S13/rank  total
--   ------  -----------  --------  --------  -----
02   kt+0    rank00..31   1024B     512B      48KB
03   kt+1    rank00..31   1024B     512B      48KB
```

Per issuing thread / rr rank:

```text
M13 1024B = 256 u32 packets = same W13 rank coverage sideband
S13  512B = 128 u32 packets = same W13 rank coverage sideband

derived original-i view:
  M13: 256 packets / 4 packets per original i = i64
  S13: 128 packets / 2 packets per original i = i64
```

Totals:

```text
M13 / H128 / SM = 32 * 1024B = 32KB
S13 / H128 / SM = 32 *  512B = 16KB

M13 / H256 / SM = 64KB
S13 / H256 / SM = 32KB
```

### Full Traffic Unit

Per kt step / SM:

```text
W13 = 256KB
M13 =  64KB
S13 =  32KB
WMS = 352KB
```

Across 148 SM:

```text
W13 = 148 * 256KB = 37.0 MiB
WMS = 148 * 352KB = 50.9 MiB
```

## Current S13 Smem Tx Sketch

kernel.cu shape:

```text
for k = 0 only:
  for rri = 0, 4:
    for rrip = 0..3:
      if rr == 2 and ((rri << 2) + rrip) == rank:
        cp.async.bulk.wait_group 1

    if rr == 2:
      rr32x32.sync()
      copy 32B to smem + rank*64 + 00
      copy 32B to smem + rank*64 + 32
```

Wait election currently:

```text
rri=0 -> ranks 00,01,02,03 wait_group 1
rri=4 -> ranks 16,17,18,19 wait_group 1
```

Panel coverage currently:

```text
smem tx predicate is rr==2
rr02 maps hp=kt+0

rr03 prefetches hp=kt+1
rr03 does not smem-tx anything yet
```

Important mismatch:

```text
wait_group is elected-rank only
smem tx predicate is rr==2 only
=> current sketch issues smem tx from all ranks, not just elected ranks
```

### S13 Smem Tx Address

kernel.cu:

```text
src = S13 + e*(H>>7)*(I<<1) + hp*(I<<1) + (rank << 4)
dst = smem + (rank << 6)

tx0: dst + 00 <- src + 0B,     32B
tx1: dst + 32 <- src + 1024B,  32B
```

Pointer arithmetic:

```text
rank << 4 is u32 arithmetic
rank step = 16 u32 = 64B

32B tx = 8 u32
S13 per original i = 2 u32
32B tx = 4 original i worth of S13 payload
```

Source horizontal:

```text
row0:
  r00 i000..003 | r01 i008..011 | r02 i016..019 | ... | r31 i248..251

row1 = row0 + 1024B = row0 + 256 u32 = row0 + i128:
  r00 i128..131 | r01 i136..139 | r02 i144..147 | ... | r31 i376..379
```

Overlap/gap:

```text
row1(r00..15) overlaps row0(r16..31)
row0 skips i004..007, i012..015, ...
row1 skips the same 4i gaps in its band
```

So current S13 smem tx is not yet a unique contiguous S13 tile.

## Smem Bank Board

Current dst:

```text
dst0 = smem + rank*64 + 00
dst1 = smem + rank*64 + 32
```

Bank rule:

```text
bank = (byte >> 2) & 31
```

Destination map:

```text
rank   dst row0          dst row1          banks row0  banks row1
----   ---------------   ---------------   ----------  ----------
00     smem+0000..0031   smem+0032..0063   00..07      08..15
01     smem+0064..0095   smem+0096..0127   16..23      24..31
02     smem+0128..0159   smem+0160..0191   00..07      08..15
03     smem+0192..0223   smem+0224..0255   16..23      24..31
...
31     smem+1984..2015   smem+2016..2047   16..23      24..31
```

Good consumer:

```text
rank pair = (2p, 2p+1)

lane 00..07 -> even rank row0 -> banks 00..07
lane 08..15 -> even rank row1 -> banks 08..15
lane 16..23 -> odd  rank row0 -> banks 16..23
lane 24..31 -> odd  rank row1 -> banks 24..31

=> lanes 00..31 -> banks 00..31
```

Bad consumer:

```text
lane k -> smem + k*64 + const
=> component-across-ranks
=> 16-way bank-hot
```

Verdict:

```text
dst bank shape is good for rank-pair / row-pair consumer.
source tile is not yet good.
```

## Pressing Next

- [ ] Lock S13 smem source tile before more bank work.

```text
choose one:
  A) all ranks issue, source formula must be unique/contiguous
  B) elected ranks issue, smem tx predicate must include elected rank
  C) two-stage rri/rrip walk, source/dst must include rri/rrip offsets
```

- [ ] Decide what `+1024B` is supposed to mean.

```text
currently:
  +1024B = +256 u32 = +i128 within same hp

possible intents:
  adjacent 32B row within same small tile
  paired I128 band within same H128
  next H128 panel

note:
  next H128 panel stride is `(I<<1)` u32, not fixed +1024B.
```

- [ ] Keep W13 prefetch board in packet notation.

```text
do not describe 4096B as direct original-i rows first.
primary:
  4096B -> I512' (I256, I256)
derived:
  i64 only because W13 has 16 u32 per original i / H128
```

- [ ] Decide M13 smem landing discipline.

```text
current:
  M13 is prefetched with S13 in rr02/rr03
  M13 has no smem tx sketch yet

need:
  same rank-pair row-pair shape?
  separate sideband bucket?
  direct rmem path?
```

- [ ] Decide W13 consumer landing.

```text
current:
  W13 L2 prefetch exists
  W13 L2 -> smem/tmem/rmem consumer does not exist

constraint:
  consumer should preserve I512' (I256,I256) / rank mental model
```

- [ ] Recompute mbarrier `expect_tx` only after S13/M13 tx shape is real.
- [ ] Keep `wait_group` cadence explicit as a traffic governor.
- [ ] Add OOB/tail byte handling only after the main ideal-shape board is stable.
- [ ] Fix syntax/hygiene after the board is stable.

## Syntax / Compile Hygiene

Known current sketch issues:

```text
line 3:   namespace alias form
line 26:  `thread_block` name
line 30:  `uchar` name / smem byte count typo 32728 vs comment 32768
line 64:  unused asm immediate operand
line 91:  missing asm operand comma before second "n"
line 98:  `rrip` undeclared
line 110: malformed if parens
line 117: smem tx asm operands are not wired correctly
line 118: [%0 + 32] / [%1 + 1024] PTX operand form needs rewrite
line 117: mbar operand `%2` is referenced but not passed
```

Do not let syntax cleanup mutate the board.

## Parked

- [ ] 2SM / expert variant profile.
- [ ] W13GS global scale path.
- [ ] X4/SX activation staging.
- [ ] TOPK bitplane loop outside current kernel sketch.
- [ ] tcgen05 / mma consumer board.
- [ ] REAP kernel for NVFP4.
- [ ] SparseGPT kernel for NVFP4.
- [ ] proxy fences / mbar wait semantics.
- [ ] non-ideal I/H tail handling.

```text
NVFP4 parked kernel tracks:
  REAP      -> pruning / reconstruction path, decide W/M/S packet contract
  SparseGPT -> calibration / one-shot sparse quant path, decide output layout target

keep separate from current ff1 W13/M13/S13 prefetch board until kernel shape is real.
```
