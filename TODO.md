# sm12x TODO / Co-Design Board

- [x] Padded Sentinel based premption (8*32-bit reg can hold information of total tokens, for now repeated 8 times)

- []



## Pressing / Next

- [x] Keep sm12x sketch inline for now.

```text
xRR48_sm12x.cu:
  kernel orchestration
  layout boards
  loop order
  raw PTX

no helpers yet:
  lower register pressure
  easier to see exact index math
  move only real repeated PTX later
```

- [x] Add clean sideband bring-up file.

```text
xRR48_sm12x_sideband.cu:
  no W13
  no ldmatrix
  no tmem
  M13: [E,H128,I<<2] -> smem by TMA + mbar
  S13: [E,H128,I<<1] -> smem by LDGSTS
```

- [x] Lock current `I` tile name: `I64 == I128'`.

```text
current W13 sketch:
  one i-step = 8 warps x 512B
             = 4096B
             = 1024 u32
             = 64 original-i rows x 16 W13 packets

therefore:
  current panel = I64 x H128
  I64 == I128'
  because:
    I64 Up
    I64 Gate
```

- [x] Fix W13 loop bound / name.

```text
old:
  for i < ((I << 4) >> 7)
  addr += i << 10

problem:
  bound says 128 u32 chunks
  addr step says 1024 u32 chunks

now:
  i < (I >> 6)
  addr = i64 << 10
```

- [x] Fix M13 active-lane index.

```text
current active lanes:
  lane = threadIdx.x & 31
  active if lane < 2
  8 warps x 2 lanes x 16B = 256B

current index:
  (warp + lane) << 2

problem:
  w0.l1 == w1.l0
  w1.l1 == w2.l0
  aliasing

candidate:
  (((threadIdx.x >> 5) << 1) + (threadIdx.x & 1)) << 2
  16 lanes -> 16 contiguous 16B packets
```

- [x] Restore CTA-local topk sentinel before state.

```text
topk_bitplanes[e][0] == 0:
  load sentinel
  CTA sync
  return before rmem/smem/mbar/tmem
```

- [x] Wire `mtbar[0]` init.

```text
mtbar[0]:
  stage mbar

init:
  thread0
  mbarrier.init.layout::v0.shared::cta.b64
  sync
```

- [x] Fill M13 16B issuer smem indexing.

```text
M13:
  4 stages x 256B = 1024B
  1024B = I64 x 4 M13 packets

m0 smem+0000..0255 -> i00..15
m1 smem+0256..0511 -> i16..31
m2 smem+0512..0767 -> i32..47
m3 smem+0768..1023 -> i48..63
```

- [x] Explain `for (m=0; m<4; m++)`.

```text
M13 / i row:
  4 u32 = 16B
  {m1 h000..063, m1 h064..127, m3 h000..063, m3 h064..127}

M13 / I64:
  64 rows x 16B = 1024B

16B issue wave:
  lane0,lane1 / warp
  16 lanes x 16B = 256B

therefore:
  4 waves x 256B = 1024B
  m0 -> i00..15
  m1 -> i16..31
  m2 -> i32..47
  m3 -> i48..63
```

- [ ] Write the throughput board before writing more code.

```text
W13:
  gmem->L2 prefetch
  512B / warp
  8 warps
  4096B / CTA step

M13/S13:
  LDGSTS or TMA
  16B / active thread
  active 2 lanes / warp
  8 warps
  256B / CTA step
```

## Current Sketch Board

```text
CTA=256
8 warps
warp = threadIdx.x >> 5
lane = threadIdx.x & 31

W13:
  lane0 / warp issues 512B L2 prefetch
  8 lanes total
  4096B panel

M13:
  lane0,lane1 / warp issue 16B TMA
  mbar expect_tx = 1024B
  4 waves x 256B

S13:
  lane0,lane1 / warp issue 16B LDGSTS
  2 waves x 256B
```

## W13 Board

### Source

```text
W13: [E, (H+127)>>7, I<<4]

per original i / H128:
  w1 h000..015
  w1 h016..031
  w1 h032..047
  w1 h048..063
  w1 h064..079
  w1 h080..095
  w1 h096..111
  w1 h112..127
  w3 h000..015
  w3 h016..031
  w3 h032..047
  w3 h048..063
  w3 h064..079
  w3 h080..095
  w3 h096..111
  w3 h112..127
```

### gmem->L2, current sketch

```text
i64 panel:
  base = W13
       + e * ((H+127)>>7) * (I<<4)
       + kt * (I<<4)
       + i64 * 1024

warp issue:
  w0 lane0 -> base + 0000 u32 -> 512B
  w1 lane0 -> base + 0128 u32 -> 512B
  w2 lane0 -> base + 0256 u32 -> 512B
  w3 lane0 -> base + 0384 u32 -> 512B
  w4 lane0 -> base + 0512 u32 -> 512B
  w5 lane0 -> base + 0640 u32 -> 512B
  w6 lane0 -> base + 0768 u32 -> 512B
  w7 lane0 -> base + 0896 u32 -> 512B

total:
  8 x 512B = 4096B
  1024 u32
  I64 x 16 packets
```

### Horizontal view

```text
one W13 prefetch panel = I64 x H128 = I128' x H128

             u32 000..127   128..255    256..383    384..511    512..639    640..767    768..895    896..1023
             ------------   ---------    ---------    ---------    ---------    ---------    ---------    ----------
issuer       w0.l0          w1.l0        w2.l0        w3.l0        w4.l0        w5.l0        w6.l0        w7.l0
bytes        0000..0511     0512..1023   1024..1535   1536..2047   2048..2559   2560..3071   3072..3583   3584..4095
meaning      I00..07        I08..15      I16..23      I24..31      I32..39      I40..47      I48..55      I56..63
```

### Vertical view

```text
W13 checkpoint
  |
  | 8 lane0 issuers
  | 512B / warp
  v
L2
  |
  | later: TMA / LDGSTS / shaped smem fill
  | open: exact tx path
  v
smem / tmem / rmem
  |
  | W1/W3 H128 panel
  v
FF1
```

## M13 Board

### Source

```text
M13: [E, H128, I<<2]

per original i / H128:
  m1 h000..063
  m1 h064..127
  m3 h000..063
  m3 h064..127

ratio:
  M13 = W13 / 4
```

### 16B issuer direction

```text
active:
  lane0,lane1 per warp
  16 active threads
  16B/thread
  256B total

correct lane id:
  mid = (warp << 1) + lane01
  lane01 = lane & 1

issue:
  mid00 -> u32 00..03
  mid01 -> u32 04..07
  mid02 -> u32 08..11
  mid03 -> u32 12..15
  ...
  mid15 -> u32 60..63
```

### TODO

- [x] Correct M13 source to W13 H128 regime.
- [x] Add M13 TMA smem stage in `xRR48_sm12x_sideband.cu`.
- [x] Add M13 mbar init / expect_tx / wait wiring for fixed 1024B panel.
- [ ] Review if M13 should stay 16B issued TMA or use larger tx.
- [ ] Write M13 smem bank layout before tmem/rmem handoff.

## S13 Board

### Source

```text
S13: [E, H128, I<<1]

one u32:
  4x u8 scales
  128 sparse cols

per original i / H128:
  s1 i h000..127
  s3 i h000..127

ratio:
  S13 = W13 / 8
```

### TODO

- [x] Correct S13 source to W13 H128 regime.
- [x] Add S13 LDGSTS smem stage in `xRR48_sm12x_sideband.cu`.
- [x] Add S13 raw smem stage:

```text
S13 / I64:
  64 rows x 8B = 512B

LDGSTS:
  16 lanes x 16B = 256B

s0 smem+1024..1279 -> i00..31
s1 smem+1280..1535 -> i32..63
```

- [ ] Keep `ldmatrix` parked until M13/S13 smem hold is reviewed.
- [ ] Pack S13 `ldmatrix` regs into F233 row order:

```text
raw:
  s13_0 -> i00..15 subtile
  s13_1 -> i16..31 subtile
  s13_2 -> i32..47 subtile
  s13_3 -> i48..63 subtile

target:
  row00 = {s1 i00, s1 i32, s3 i00, s3 i32}
  row01 = {s1 i01, s1 i33, s3 i01, s3 i33}
  ...
  row31 = {s1 i31, s1 i63, s3 i31, s3 i63}
```

- [ ] Review final S13 tmem layout against PTX F233 sparse SFA pane.

## TMA / LDGSTS Throughput Board

```text
W13:
  target large tx
  512B / warp prefetch
  4096B / CTA prefetch panel
  good match for later bulk movement

M13/S13:
  small sideband
  16B/thread LDGSTS candidate
  2 active lanes/warp
  256B / CTA issue

open:
  verify 512B prefetch legality / split if needed
  verify wait-group cadence
  verify overlap with mbar/tmem alloc
  verify smem bank map before tcgen05 handoff
```

## Orchestration Board

```text
stage 0:
  sentinel pre-emption

stage 1:
  mbar init
  tmem alloc
  taddr/mbar pass

stage 2:
  W13 gmem->L2 prefetch
  M13/S13 LDGSTS or TMA sideband

stage 3:
  smem->tmem / rmem handoff

stage 4:
  FF1
  SwiGLU
  topkW
  FF2
```

## Code Hygiene

- [ ] Add includes / aliases (`cuda_bf16.h`, `stdint.h`, `uint32`, `uint16`, `uchar`).
- [x] Fix `threadIdx >> 5` -> `threadIdx.x >> 5`.
- [ ] Keep index math inline, comments as boards.
- [ ] No register-heavy helper abstractions in the kernel.
- [ ] Helpers only for PTX repetition / state ops.
- [ ] No local nvcc until requested.

## Parked

- [ ] Exact mbar tx counts.
- [ ] Proxy fence placement.
- [ ] tcgen05 commit/wait.
- [ ] Final W13 tmem shape.
- [ ] Final M13/S13/SX sideband tmem shape.
- [ ] 2CTA variant.

## Completed / Carried From xRR48

- [x] W13/M13/S13 checkpoint-native layouts are the right starting point.
- [x] CTA-local sentinel pre-emption belongs before heavy state.
- [x] M13 = W13 / 4.
- [x] S13 = W13 / 8.
- [x] S13 scale packet = one u32 = 4x u8 = 128 sparse cols.
- [x] Comments must be maps, not prose blobs.
