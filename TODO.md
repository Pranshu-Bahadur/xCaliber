# sm12x xRR48 TODO / Live Blackboard

This repo is the live blackboard. Cleaned/finalized slices can move to
`Fused-Sparse-NVFP4-Kernels` later.

Current focus:

```text
kernel.cu
CTA=1024
1 CTA = 1 SM = 1 expert
rr32x32 issue fabric
full-I x H256 W/M/S prefetch unit
S13 smem landing + bank map
```

Do not drift back to the old CTA256/I64 board except as lineage.

## Current Lock

- [x] CTA1024 is the current board.
- [x] Current prefetch unit is intentionally full-I x H256.
- [x] H256 was chosen because RTX PRO 6000 has enough L2 to make this regime plausible.
- [x] `kernel_bank_map.txt` is the current bank-conflict artifact.
- [x] W13/M13/S13 source layouts remain checkpoint-native and unchanged.
- [x] `rr32x32.thread_rank()` is the warp-rank inside a lane-label group; this is part of the board, not an open question.

```text
rr32x32 = labeled_partition(cta1024, ((thread_rank ^ 32) & 31))

because:
  ((tid ^ 32) & 31) == (tid & 31)

therefore:
  meta_group_rank = lane id 0..31
  thread_rank     = physical warp id 0..31

important:
  rr rank 00..31 are one lane across 32 physical warps,
  not 32 lanes inside one physical warp.
```

## rr32x32 Horizontal Board

```text
physical:
  tid = warp*32 + lane

partition:
  label = (tid ^ 32) & 31 = lane

rr view:
  meta_group_rank = lane
  thread_rank     = warp
```

Compressed horizontal map:

```text
rr/lane  physical tids across rank
-------  ----------------------------------------------------------
rr00     000, 032, 064, 096, ..., 992      rank 00..31 = warp 00..31
rr01     001, 033, 065, 097, ..., 993      rank 00..31 = warp 00..31
rr02     002, 034, 066, 098, ..., 994      rank 00..31 = warp 00..31
rr03     003, 035, 067, 099, ..., 995      rank 00..31 = warp 00..31
...
rr31     031, 063, 095, 127, ..., 1023     rank 00..31 = warp 00..31
```

Use:

```text
rr lane bucket -> selects stream / phase
rr rank        -> walks 32 full-I stripes
```

## Pressing / Next

- [ ] Fix syntax/hygiene only after board is stable.

```text
known syntax/hygiene:
  namespace alias line
  missing includes / aliases
  rrip declaration
  malformed if parens around rr32x32.meta_group_rank()
  inline asm operand separators
  [%0 + 32] / [%1 + 1024] PTX operand form
  mbar operand should be shared address object, not raw array pointer
```

- [ ] Lock current S13 shared landing as a real consumer contract.

```text
current good pattern:
  copy issue:
    one lane / physical warp
    2 x 32B rows per rr rank

  consumer:
    rank-pair / row-pair
    full 32-bank coverage

bad pattern:
  component-across-ranks
  16-way bank-hot
```

- [ ] Decide whether M13 should share the same rank-pair row-pair smem discipline.
- [ ] Decide whether M13+S13 stay in the same sideband bucket or split.

```text
current direction:
  rr0,1 -> W13
  rr2,3 -> M13 + S13 prefetch
  rr2   -> S13 smem fill sketch

open:
  rr2 only for S13 fill?
  rr2/3 both fill?
  M13 smem fill next to S13 or separate?
```

- [ ] Write W13 L2->smem/tmem/rmem stage after W13 prefetch board is stable.
- [ ] Decide W13 consumer landing shape before changing W13 bytes/thread.
- [ ] Recompute mbar `expect_tx` after final M13/S13 smem fill shape.
- [ ] Keep `wait_group` cadence explicit: traffic governor vs deep pipe.
- [ ] Add OOB/tail byte handling for non-ideal H/I only after the main board is stable.

## Current Throughput / Traffic Board

Current intended unit:

```text
full-I x H256

I = 2048
H256 = 2 x H128
proj = W1/W3
```

Per SM / per kt group:

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

Interpretation:

```text
WMS = W13 + M13 + S13

W13 is the main traffic.
M13/S13 are sideband, but still 96KB / SM for full-I x H256.
```

## Current Horizontal Prefetch / Smem-Tx Board

### Prefetch Buckets

```text
rr   lane  ranks       stream      hp      bytes/rank  total
--   ----  ----------  ----------  ------  ----------  ------
00   00    warp00..31  W13         kt+0    4096B       128KB
01   01    warp00..31  W13         kt+1    4096B       128KB
02   02    warp00..31  M13 + S13   kt+0    1024+512B   48KB
03   03    warp00..31  M13 + S13   kt+1    1024+512B   48KB
```

Total per kt step:

```text
W13 = 256KB
M13 =  64KB
S13 =  32KB
WMS = 352KB
```

### W13 Prefetch Horizontal

```text
W13 base:
  W13 + e*(H>>7)*(I<<4) + hp*(I<<4)

rank chunk:
  rank << 10 u32 = rank * 4096B

rank -> original i range:
  rank00 -> i0000..0063
  rank01 -> i0064..0127
  rank02 -> i0128..0191
  rank03 -> i0192..0255
  ...
  rank31 -> i1984..2047
```

Horizontal:

```text
rr00 W13 hp=kt+0:
  r00 i0000..0063 | r01 i0064..0127 | ... | r31 i1984..2047

rr01 W13 hp=kt+1:
  r00 i0000..0063 | r01 i0064..0127 | ... | r31 i1984..2047
```

### M13/S13 Prefetch Horizontal

```text
M13 base:
  M13 + e*(H>>7)*(I<<2) + hp*(I<<2)

M13 rank chunk:
  rank << 8 u32 = rank * 1024B
  rank00 -> i0000..0063
  rank01 -> i0064..0127
  ...
  rank31 -> i1984..2047

S13 base:
  S13 + e*(H>>7)*(I<<1) + hp*(I<<1)

S13 rank chunk:
  rank << 7 u32 = rank * 512B
  rank00 -> i0000..0063
  rank01 -> i0064..0127
  ...
  rank31 -> i1984..2047
```

Horizontal:

```text
rr02 hp=kt+0:
  M13 r00..r31 | S13 r00..r31

rr03 hp=kt+1:
  M13 r00..r31 | S13 r00..r31
```

### Current S13 L2 -> Smem Tx

```text
predicate:
  meta_group_rank == 2

dst:
  row0 = smem + rank*64 + 00
  row1 = smem + rank*64 + 32

tx:
  row0 <- 32B
  row1 <- 32B

rank:
  rr32x32.thread_rank() = physical warp id
```

Destination horizontal:

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

Good consumer horizontal:

```text
rank pair = (2p, 2p+1)

lane 00..07 -> even rank row0 -> banks 00..07
lane 08..15 -> even rank row1 -> banks 08..15
lane 16..23 -> odd  rank row0 -> banks 16..23
lane 24..31 -> odd  rank row1 -> banks 24..31

result:
  lanes 00..31 -> banks 00..31
```

Bad consumer:

```text
lane k -> smem + k*64 + const
=> component-across-ranks
=> 16-way hot
```

Source tx board to finish:

```text
current source expression:
  src0 = S13 + e*(H>>7)*(I<<1) + hp*(I<<1) + (rank << 4)
  src1 = src0 + 1024B

note:
  rank << 4 is u32 arithmetic = rank * 64B

open:
  verify intended source tiling for src1 = src0 + 1024B
  decide rank gating / rri use so the source rows do not get ambiguous
  preserve the dst bank-clean rank-pair shape
```

## W13 Layout

Keep this layout pinned. Horizontal view only.

```text
W13: [E, (H+127)>>7, I<<4]

per original i / H128:

u32:   00           01           02           03           04           05           06           07
W1:    h000..015    h016..031    h032..047    h048..063    h064..079    h080..095    h096..111    h112..127

u32:   08           09           10           11           12           13           14           15
W3:    h000..015    h016..031    h032..047    h048..063    h064..079    h080..095    h096..111    h112..127
```

Packet:

```text
1 W13 u32 packet = 8 stored nvfp4 values = 16 sparse H positions
per i/proj/H128 = 8 W13 packets = 32B
per i/W1+W3/H128 = 16 W13 packets = 64B
```

Current full-I x H256:

```text
I2048 x H128:
  2048 * 64B = 128KB

I2048 x H256:
  2 * 128KB = 256KB
```

## M13 Layout

Keep this layout pinned. Horizontal view only.

```text
M13: [E, H128, I<<2]

per original i / H128:
  u32 00: m1 h000..063
  u32 01: m1 h064..127
  u32 02: m3 h000..063
  u32 03: m3 h064..127

ratio:
  M13 = W13 / 4
```

Packet:

```text
per i/H128:
  4 u32 = 16B
```

Current full-I x H256:

```text
I2048 x H128:
  2048 * 16B = 32KB

I2048 x H256:
  2 * 32KB = 64KB
```

## S13 Layout

Keep this layout pinned. Horizontal view only.

```text
S13: [E, H128, I<<1]

one u32:
  4x u8 scales
  128 sparse cols

per original i / H128:
  u32 00: s1 h000..127
  u32 01: s3 h000..127

ratio:
  S13 = W13 / 8
```

Important:

```text
S13 has nothing to do with n8 structurally.
S13 is H/K-side scale sidecar.

per original i/proj/H128:
  1 S13 packet corresponds to 8 W13 packets.
```

Packet:

```text
per i/H128:
  2 u32 = 8B
```

Current full-I x H256:

```text
I2048 x H128:
  2048 * 8B = 16KB

I2048 x H256:
  2 * 16KB = 32KB
```

## Smem Bank Board

Current S13 landing:

```text
rank = rr32x32.thread_rank()
dst0 = smem + rank*64 + 00
dst1 = smem + rank*64 + 32
```

Bank map:

```text
even rank:
  row0 -> banks 00..07
  row1 -> banks 08..15

odd rank:
  row0 -> banks 16..23
  row1 -> banks 24..31
```

Good consumer:

```text
consume adjacent rank pairs:

lane 00..07 -> even rank row0 -> banks 00..07
lane 08..15 -> even rank row1 -> banks 08..15
lane 16..23 -> odd  rank row0 -> banks 16..23
lane 24..31 -> odd  rank row1 -> banks 24..31
```

Bad consumer:

```text
lane k -> smem + k*64 + const

result:
  component-across-ranks
  16-way bank-hot
```

Artifact:

```text
kernel_bank_map.txt
```

## Orchestration Board

```text
stage 0:
  padded sentinel / Xb pre-emption
  return before heavy state when empty

stage 1:
  mbar init
  smem allocated
  rmem indexing regs live

stage 2:
  W13 gmem->L2 prefetch
    rr0,rr1
    full-I x H256

  M13/S13 gmem->L2 prefetch
    rr2,rr3
    full-I x H256 sideband

stage 3:
  S13 L2->smem sketch
  rank-pair bank-clean consumer contract

stage 4:
  M13 L2->smem
  W13 L2->smem/tmem/rmem
  dequant / MMA feed

stage 5:
  FF1
  SwiGLU
  topkW
  store / later FF2
```

## Active Review Items

- [ ] Decide whether S13 smem fill should use rr2 only or rr2+rr3.
- [ ] Finish S13 smem source tiling around `src1 = src0 + 1024B`.
- [ ] Add M13 smem fill map and bank proof.
- [ ] Add W13 destination map and bank proof.
- [ ] Define S13 rank-pair consumer code before changing the smem layout.
- [ ] Define M13 consumer before choosing final sideband bucket layout.
- [ ] Decide whether `wait_group 1` is traffic governance or an unfinished placeholder.
- [ ] Recompute exact `mbarrier.init` count and `expect_tx` once fill shape is fixed.
- [ ] Keep OOB/tail bytes parked until main path compiles as a sketch.
- [ ] Do not run local `nvcc` unless explicitly requested.

## Syntax / Compile Hygiene

Keep these separate from architecture review.

- [ ] Fix namespace alias.
- [ ] Add `cuda_bf16.h`, `stdint.h`, and type aliases.
- [ ] Declare `rrip`.
- [ ] Fix malformed `if` around `meta_group_rank == (2+k)`.
- [ ] Fix inline PTX operand list separators.
- [ ] Review whether `[%0 + 32]` and `[%1 + 1024]` are legal PTX forms.
- [ ] Pass mbar shared address correctly into `cp.async.bulk.shared...mbarrier`.
- [ ] Add `memory` clobbers where needed.

## Parked

- [ ] Final W13 tmem shape.
- [ ] Final M13/S13/SX tmem shape.
- [ ] Proxy fence placement.
- [ ] `tcgen05` commit/wait.
- [ ] 2CTA / 2SM variant.
- [ ] FF2 / PDL / CLC.
- [ ] Public README / cleaned Fused repo transfer.

## Completed / Carried

- [x] W13/M13/S13 checkpoint-native layouts are the correct baseline.
- [x] CTA-local sentinel/pre-emption belongs before heavy state.
- [x] M13 = W13 / 4.
- [x] S13 = W13 / 8.
- [x] S13 packet = one u32 = 4x u8 = 128 sparse cols.
- [x] S13 is K/H-side scale sidecar, not n8-side data.
- [x] `rr32x32.meta_group_rank() -> lane`, `rr32x32.thread_rank() -> warp` is the intended fabric.
- [x] `rr32x32` bank-map artifact updated for the current kernel.
- [x] Comments should stay as maps/formulas, not prose blobs.
