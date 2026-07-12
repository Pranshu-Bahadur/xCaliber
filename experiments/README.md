# Activation L2 Prefetch Experiment

Activation transport lineage plus the full CTA1024 FF1 checkpoint for
`kernel.cu` on the NVIDIA RTX PRO 6000 Blackwell Server Edition.

## Overall Winner - Paired No-Workspace N16

```text
loop       live n16 pair -> i1024 -> I256 plane -> H2048 panel -> k64
reuse      one W13/S13 packet feeds up to four routed N8 MMAs
partials   registers through H; no global W1/W3 partial workspace
output     SwiGLU * topk_W -> BF16 cp.reduce directly into X[I,NP]
staging    two dynamic 2*N16 activation stages; 77,824B max
ptxas      58 registers; zero stack; zero spills; 1 CTA/SM
```

Corrected exact-live sweep:

```text
E=384 I=2048 H=7168 TOPK=8 seeds=3 repeats=20
N8   live={8,64}
N256 live={128,384}
N512 live={128,384}
18 rows; all PASS; equal output checksum
```

This table includes the final 2-entry literal `XbMaskLUT`.

```text
+-----+------+----------+----------+---------+----------------+
| N   | live | N16x2 ms | serial   | vs old  | vs partial-best|
+-----+------+----------+----------+---------+----------------+
|   8 |    8 |    0.547 |    0.522 |   0.96x |          1.42x |
|   8 |   64 |    0.780 |    0.770 |   0.99x |          1.14x |
| 256 |  128 |   11.511 |   22.041 |   1.91x |          4.46x |
| 256 |  384 |   13.976 |   23.373 |   1.67x |          5.60x |
| 512 |  128 |   23.021 |   44.083 |   1.92x |          9.57x |
| 512 |  384 |   25.955 |   47.062 |   1.81x |         10.84x |
+-----+------+----------+----------+---------+----------------+
```

Redundant-load call:

```text
serial W replay   active_n16/live_experts
paired W replay   ceil(active_n16_per_expert/2)

N256/live128      16x -> 8x
N512/live128      32x -> 16x

activation replay remains 4*(I/1024) = 8x
activation is ~3% of streamed W/S/X bytes on the large boards
do not spend the remaining register budget on plane pairing first
```

Raw data:

```text
ff1_w4a4_swiglu_topk_reduce_n16x2_combined.csv
ff1_w4a4_swiglu_topk_reduce_n16x2_summary.csv
ff1_w4a4_swiglu_topk_reduce_combined.csv          serial lineage
ff1_w4a4_swiglu_topk_reduce_summary.csv           serial lineage
```

## Global-Partial Checkpoint - Lineage

The section below is the superseded global-partial transport design. Keep it
for transport lineage; do not use it as the current full-kernel winner.

Corrected CTA1024 schedule:

```text
kt outer -> i1024 -> n256 -> n8
4 x m16n8k64 / n8
BF16 expert partial load + store for kt<last
last kt -> SwiGLU -> topk_W -> red.global BF16 Y
W13GSXGSINV = W13GS * XGSINV
```

Sweep:

```text
E=384 I=2048 H=7168 TOPK=8
N={8,256,512} pool={128,384} seeds=3 repeats=20
72 rows; all PASS; equal checksum / four-mode cell
```

```text
+-----+------+------------------------------+----------+----------+
| N   | pool | paired winner                | med ms   | vs TMA0  |
+-----+------+------------------------------+----------+----------+
|   8 |  128 | no-PF cp.async.ca            |    0.911 |   +4.10% |
|   8 |  384 | no-PF cp.async.ca            |    0.922 |   +4.30% |
| 256 |  128 | PF32 cp.async.ca             |   34.638 |   +0.64% |
| 256 |  384 | PF32 cp.async.ca             |   55.771 |   +0.14% |
| 512 |  128 | PF32 cp.async.ca             |  141.020 |   +0.33% |
| 512 |  384 | PF32 cp.async.ca             |  222.017 |   +0.07% |
+-----+------+------------------------------+----------+----------+
```

Call:

```text
N=8
  no-PF cp.async.ca is real: +4.1..4.3% vs TMA0

N>=256
  no material transport winner
  every PF/TMA/cp.async result is within 0.7% paired
  keep TMA0 as standard baseline until partial traffic is removed

why
  N256 pool384 partial load+store ~= 25.95GB / launch
  N512 pool384 partial load+store ~= 51.79GB / launch
  activation X4+SX is only ~= 16MB / 31MB
  global BF16 partial churn hides the activation transport edge
```

Runtime SMEM:

```text
S planes            4 * blockDim.x * 4B
activation pingpong 2 * min(N,256) * (32B X4 + 4B SX)
N8                  16960B
N256/N512           34816B
```

`ptxas`: 64 registers/thread, one 4B spill slot, 1 CTA/SM. The spill is one
address value parked across the MMA body; W/B/C payloads are not spilled.

Raw data:

```text
ff1_w4a4_scattered_combined.csv   72 measured rows
ff1_w4a4_scattered_summary.csv    paired three-seed summary
ff1_w4a4_scattered_exp.py         all-in-one source/build/sweep/validation
```

## Transport-Only Final Call (Lineage)

```text
OVERALL BEST         pf32x4_current_tma_smem
transport            TMA global -> raw SMEM
prefetch             current K64 as 4 x 32B, no explicit wait
target               nearly-full expert pool / multi-wave ACTCOMP

standard baseline     no_pf_tma_smem
primary cache state   no explicit L2 scrub
low-pressure fallback no_pf_tma_smem
experimental control  pf32x4_current_cta_nc_actcomp_smem
```

TMA is the standard transport and the only overall baseline. Every reported
`vs TMA0` result compares equal work against `no_pf_tma_smem` for the same
route seed, path, output contract, and cache state.

The live call:

```text
one-wave / light pressure
  start from TMA0
  CTA .nc can avoid the raw-SMEM stage
  do not force 4x32B PF

nearly-full pool / three scheduling waves
  current-panel PF wins
  use 4x32B no-wait for the kernel-native mapping

transport
  implement TMA-P32
  retain CTA-P32 only as the full-kernel control
```

At no flush and `N=256,pool=384`, route-paired scoring selects TMA `4x32B`
at `+4.41%` versus TMA0. TMA `2x64B` has the lowest raw median at `9.925ms`,
only `0.08%` from TMA `4x32B` at `9.933ms`. At `N=512,pool=384`, TMA
`2x64B`, TMA `4x32B`, and CTA `4x32B` are within `0.06%` raw time. The
kernel-native `4x32B` shape gives up nothing material, and TMA is the standard
transport. Final call: `pf32x4_current_tma_smem`.

## Hypothesis Confirmed

The scattered-load L2 absorption hypothesis won.

```text
hypothesis
  prefetch ownership does not need to match consumer ownership
  scattered token rows can be hinted as complete K64 sector groups
  later scattered TMA / CTA loads can consume those same resident L2 lines

PF producer
  four 256T bands
  4 x 32B hints cover X[token, K64] = 128B

consumer
  different thread ownership
  token row IDs remain scattered
  each selected row still contains one contiguous 128B K64 payload

L2 absorbs
  the HBM miss leg for matching resident sectors

L2 does not absorb
  downstream sector service, address scatter, or consumer synchronization
```

The hint and consumer only need the same line address. The issuing thread,
warp, and expert CTA do not own the unified L2 line. Multiple experts may hint
the same X row; the later consumer still performs a real TMA or global load.

No explicit prefetch wait is required for correctness. If the hint lands late,
the consumer resolves the ordinary miss. If it lands early enough and survives,
the consumer skips the HBM leg.

Measured proof against standard TMA0, no explicit scrub:

```text
+-----+------+----------+----------+----------+
| N   | pool | TMA0 ms  | TMA-P32  | vs TMA0  |
+-----+------+----------+----------+----------+
| 256 |  384 |   10.380 |    9.933 |   +4.41% |
| 512 |  384 |   12.668 |   11.875 |   +6.05% |
+-----+------+----------+----------+----------+

P32 survives flush={0,128,512}MB:
  best paired reduction vs TMA0 = +4.41% .. +6.28%

verdict:
  scattered rows stayed scattered
  current-panel L2 residency still won
```

## Baseline Contract

Aliases used below:

```text
TMA0       no_pf_tma_smem
TMA-P128   pf128_current_tma_smem
TMA-P64    pf64x2_current_tma_smem
TMA-P32    pf32x4_current_tma_smem
TMA-P32W   pf32x4_current_wait0_tma_smem

CTA0       no_pf_cta_nc_actcomp_smem
CTA-P128   pf128_current_cta_nc_actcomp_smem
CTA-P32    pf32x4_current_cta_nc_actcomp_smem
CTA-P32W   pf32x4_current_wait0_cta_nc_actcomp_smem

DIR-P128   pf128_current_direct
DIR-P32    pf32x4_current_direct
DIR-P64L1  pf64x4_current_lead1_direct
```

Primary score:

```text
per seed:
  reduction_vs_TMA0 = 100 * (1 - method_ms / TMA0_ms)

reported:
  median(method_ms across seeds)
  median(reduction_vs_TMA0 across seeds)

winner:
  highest median paired reduction_vs_TMA0
  median method_ms is the tiebreaker
```

Transport-local no-PF reductions remain in the summary CSV only as diagnostic
columns. They cannot crown the overall winner.

Fair groups:

```text
pure / wdequant:
  direct warp consumer
  vs TMA raw-SMEM distribution to the same 32-warp consumer

actcomp packed producer:
  CTA .nc -> rmem -> absmax/NVFP4 pack -> packed X4/SX SMEM
  vs TMA -> raw SMEM -> rmem -> same pack -> same packed X4/SX SMEM

actcomp warp diagnostic:
  warp-replicated direct sink
  no equal-work packed-SMEM TMA baseline
  excluded from every overall winner table
```

## Primary Result

This is the closest measured state to steady-state inference:

```text
flush_mb       0
route seeds    3
repeats        20 / method / seed
warmup         one untimed launch / method
timed churn    full W13/S13 stream remains inside the event
```

ACTCOMP fair producer, no explicit scrub:

```text
+-----+------+---------+----------+---------+----------+
| N   | pool | TMA0 ms | winner   | win ms  | vs TMA0  |
+-----+------+---------+----------+---------+----------+
|   8 |  128 |   2.035 | CTA0     |   1.889 |   +7.19% |
|   8 |  384 |   2.103 | CTA0     |   2.006 |   +4.61% |
| 256 |  128 |   3.422 | CTA-P128 |   3.244 |   +5.21% |
| 256 |  384 |  10.380 | TMA-P32  |   9.933 |   +4.41% |
| 512 |  128 |   4.185 | CTA0     |   3.952 |   +5.51% |
| 512 |  384 |  12.668 | TMA-P64  |  11.877 |   +6.15% |
+-----+------+---------+----------+---------+----------+
```

Selected full-pool methods:

```text
+-----+-----------+--------+----------+----------------+
| N   | transport | PF     | ms       | vs TMA0        |
+-----+-----------+--------+----------+----------------+
| 256 | TMA       | none   |   10.380 | baseline       |
| 256 | TMA       | 2x64B  |    9.925 |         +4.39% |
| 256 | TMA       | 4x32B  |    9.933 |         +4.41% |
| 256 | CTA .nc   | none   |   10.282 |         +0.90% |
| 256 | CTA .nc   | 4x32B  |   10.017 |         +3.50% |
+-----+-----------+--------+----------+----------------+
| 512 | TMA       | none   |   12.668 | baseline       |
| 512 | TMA       | 2x64B  |   11.877 |         +6.15% |
| 512 | TMA       | 4x32B  |   11.875 |         +6.05% |
| 512 | CTA .nc   | none   |   12.251 |         +2.99% |
| 512 | CTA .nc   | 4x32B  |   11.871 |         +5.99% |
+-----+-----------+--------+----------+----------------+
```

Read:

```text
N256 full pool
  paired TMA baseline selects 4x32B
  2x64B owns a 0.08% raw-median edge

N512 full pool
  TMA-P64, TMA-P32, and CTA-P32 are tied within 0.06%

explicit wait
  loses to no-wait in every selected ACTCOMP cell

conclusion
  implement TMA-P32
  current-panel residency wins without departing from the standard transport
```

## Cache-State Robustness

The 512MB scrub is a cold-state stress control, not a model of kernel
boundaries. CUDA does not automatically flush L2 at every launch.

Kernel-native `4x32B`, best of TMA-P32 and CTA-P32:

```text
+-----+----------+---------+----------+---------+----------+
| N   | flush MB | TMA0 ms | P32 path | P32 ms  | vs TMA0  |
+-----+----------+---------+----------+---------+----------+
| 256 |        0 |  10.380 | TMA      |   9.933 |   +4.41% |
| 256 |      128 |  10.453 | CTA .nc  |   9.748 |   +6.28% |
| 256 |      512 |  10.440 | CTA .nc  |   9.914 |   +4.77% |
| 512 |        0 |  12.668 | TMA      |  11.875 |   +6.05% |
| 512 |      128 |  12.658 | TMA      |  11.920 |   +5.76% |
| 512 |      512 |  12.661 | CTA .nc  |  11.882 |   +5.98% |
+-----+----------+---------+----------+---------+----------+
```

The raw transport edge flips by cache state. The P32 gain does not. The
no-flush primary result selects TMA-P32, TMA is the standard path, and CTA-P32
does not win by enough to justify a separate implementation. TMA-P32 is the
final call; CTA-P32 remains a control.

## Cold Stress Sweep

The corrected v2 sweep uses `flush=512MB`. It is retained as a robustness and
cross-path stress test, not the primary inference state.

```text
+----------+-----+------+---------+------------+---------+----------+
| path     | N   | pool | TMA0 ms | winner     | win ms  | vs TMA0  |
+----------+-----+------+---------+------------+---------+----------+
| actcomp  |   8 |  128 |   2.065 | CTA0       |   1.911 |   +6.70% |
| actcomp  |   8 |  384 |   2.105 | CTA0       |   1.996 |   +5.06% |
| actcomp  | 256 |  128 |   3.422 | CTA-P128   |   3.250 |   +4.74% |
| actcomp  | 256 |  384 |  10.432 | CTA-P32    |   9.775 |   +5.98% |
| actcomp  | 512 |  128 |   4.203 | CTA0       |   3.984 |   +5.18% |
| actcomp  | 512 |  384 |  12.666 | TMA-P64    |  11.926 |   +5.85% |
+----------+-----+------+---------+------------+---------+----------+
| pure     |   8 |  128 |   1.614 | DIR-P128   |   1.553 |   +3.80% |
| pure     |   8 |  384 |   1.706 | DIR-P128   |   1.680 |   +1.57% |
| pure     | 256 |  128 |   5.181 | TMA0       |   5.181 |   +0.00% |
| pure     | 256 |  384 |  11.296 | DIR-P64L1  |  10.441 |   +7.93% |
| pure     | 512 |  128 |   8.520 | TMA0       |   8.520 |   +0.00% |
| pure     | 512 |  384 |  14.116 | TMA-P32W   |  13.200 |   +6.79% |
+----------+-----+------+---------+------------+---------+----------+
| wdequant |   8 |  128 |   2.930 | DIR-P32    |   2.842 |   +3.27% |
| wdequant |   8 |  384 |   2.955 | DIR-P32    |   2.843 |   +3.88% |
| wdequant | 256 |  128 |  32.660 | TMA0       |  32.660 |   +0.00% |
| wdequant | 256 |  384 |  31.637 | TMA0       |  31.637 |   +0.00% |
| wdequant | 512 |  128 |  56.991 | TMA0       |  56.991 |   +0.00% |
| wdequant | 512 |  384 |  51.028 | TMA0       |  51.028 |   +0.00% |
+----------+-----+------+---------+------------+---------+----------+
```

WDEQUANT is compute-dominated at N256/N512, so transport changes are buried by
the dequant sink. That table must not be read as proof that activation
transport is irrelevant.

## Run Board

```text
GPU                 RTX PRO 6000 Blackwell Server Edition
SMs                 188
target              sm_120a
grid                E = 384 CTAs
CTA                 1024 threads / 32 warps
residency           runtime-checked: exactly 1 CTA / SM

N                   {8,256,512}
TOPK                8 unique experts / token
expert_pool         {128,384}
route seeds         3
I                   2048
H                   7168
K panel             K64 BF16 = 128B / token

primary flush       0MB
stress flush        {128,512}MB outside timed event
timed repeats       20 / method / route seed
```

The kernel stops at transport or transform sinks. It does not execute MMA,
SwiGLU, `topk_W`, or output reduction.

## Routing

The host creates deterministic uniform-hash routing:

```text
token t -> eight unique experts
total assignments = N * TOPK
```

Candidate expert IDs are shuffled across `[0,E)`. Empty CTAs appear
throughout the grid and return immediately, allowing scheduler backfill.

```text
+-----+------+--------------+--------------------+
| N   | pool | live experts | tokens/live expert |
+-----+------+--------------+--------------------+
|   8 |  128 | 51..53       | 1..3               |
|   8 |  384 | 58..60       | 1..2               |
| 256 |  128 | 128          | 6..27              |
| 256 |  384 | 382..383     | 1..14              |
| 512 |  128 | 128          | 18..50             |
| 512 |  384 | 384          | 3..25              |
+-----+------+--------------+--------------------+
```

`N=8` stresses empty-expert pre-emption. `pool=128` fits in one scheduling
wave. `pool=384` reaches three waves on 188 SMs.

This is synthetic routing, not a trace from model logits.

## Xb Contract

```text
Xb[e][0]                     routed-token count / empty sentinel
Xb[e][1..ceil(N/32)]         literal routed-token bitplanes

n256=0                       token 0..255
n256=1                       token 256..511
```

The checker compares the header against the full bitplane popcount and traps
on mismatch. Sparse token positions are never compacted into low issuer lanes.

## Prefetch Map

Kernel-native ownership:

```text
t        = threadIdx.x
q        = t & 255
band     = t >> 8
token    = (n256<<8) + q
live     = Xb[e,n256,q]

band 0   tids 000..255   X[token, kt*64 +  0..15] BF16   32B
band 1   tids 256..511   X[token, kt*64 + 16..31] BF16   32B
band 2   tids 512..767   X[token, kt*64 + 32..47] BF16   32B
band 3   tids 768..1023  X[token, kt*64 + 48..63] BF16   32B

four bands cover one scattered token K64 exactly:
  4 x 32B = 128B
```

Prefetch producer identity is independent of later consumer identity. All
expert CTAs hint through the unified GPU L2; duplicate expert hints to the
same X line do not create additional unique cache lines.

The prefetch is a weak, non-blocking residency hint. It does not complete or
synchronize the later load. `wait_group` is tested only as a performance
throttle.

## Consumer Maps

### TMA Baseline

Token rows are scattered, but each token's K64 is contiguous:

```text
X[token0, kt*64 : kt*64+64] -> contiguous 128B -> raw SMEM slot 0
X[token1, kt*64 : kt*64+64] -> contiguous 128B -> raw SMEM slot 1
...
```

TMA does not perform one tensor-wide gather. One elected thread issues one
128B global-to-SMEM copy for each live token row. The CTA then reuses raw SMEM.

### CTA ACTCOMP

```text
q       = t >> 2       token position 0..255
p       = t & 3        K octet owner

one quad              4T x 2 x 16B = one token K64
one CTA pass          every live token loaded once
output                X4 32B/token + SX 4B/token
```

Both TMA and CTA paths perform the same absmax, NVFP4 pack, scale pack, packed
SMEM publication, and readback checksum.

### Warp Diagnostic

The pure/WDEQUANT consumer uses:

```text
g       = lane >> 2    token row 0..7
p       = lane & 3     16B subrange owner

load0   K[ 0+8p ..  7+8p]
load1   K[32+8p .. 39+8p]
```

One warp consumes eight scattered token rows. For ACTCOMP this warp-replicated
sink does not publish the matched packed-SMEM output, so it is diagnostic only.

## Path Sinks

```text
pure
  stream bytes into a dependency-preserving checksum

wdequant
  stream W13/S13/X
  expand E2M1x8 packets
  apply scale/global-scale arithmetic
  no MMA

actcomp
  load BF16 K64
  absmax per 16
  pack NVFP4 X4 + UE4M3 SX
  publish/read packed SMEM for fair producer methods
  no MMA
```

## Timing And Traffic

```text
1. initialize W13, S13, X, Xb, scales, and output
2. launch each method once untimed
3. optionally run the scrub kernel before a timed repeat
4. start CUDA event
5. execute one complete kernel launch
6. stop CUDA event
7. report median wall time across repeats
8. report median live-CTA clock64 cycles as secondary evidence
```

The scrub is outside the event. `flush=0` allocates no scrub buffer and
launches no scrub kernel.

Traffic columns are logical requested bytes, not measured HBM bytes. The
validator recomputes every byte count and GB/s column from the routing and
layout contracts.

Natural in-kernel L2 pressure at `I=2048,H=7168`:

```text
one (kt,i) bundle / live CTA     W13 64KiB + S13 8KiB = 72KiB
188 resident CTAs               13.2MiB requested / bundle
64MiB traffic volume            about five global bundles
one complete 188-CTA wave       5.78GiB requested W13/S13
maximum unique X                N512 * H7168 * BF16 = 7MiB
```

Real inference therefore sees rolling replacement, not a clean per-kernel
flush. The activation hint only needs to survive its short PF-to-consumer
interval.

## Correctness

```text
PTX audit
  reject malformed cache-op order
  reject unsupported bulk-PF eviction modifiers
  require 16B-aligned PF sizes
  keep CUDA/Python method tables identical

runtime
  require one resident CTA / SM
  trap on Xb header/popcount mismatch
  preserve literal sparse token positions

CSV
  rectangular header
  N*TOPK assignment count
  byte and GB/s recomputation
  equal checksum inside every fair output contract
```

## Reproduce

```bash
colab exec -s xCalibrr -f experiments/l2pfact_exp.py --timeout 3600
```

`colab exec -f` does not forward script arguments. The profile is selected by
`COLAB_SWEEP_PROFILE` near the top of the script:

```text
full                corrected all-path 512MB stress sweep
flush_sensitivity   ACTCOMP sweep at flush={0,128,512}MB
```

Local dry-run and PTX audit:

```bash
python3 experiments/l2pfact_exp.py \
  --quick --dry-run \
  --workdir /tmp/xcaliber_kernel_sim \
  --csv /tmp/kernel_sim_results.csv
```

## Future Work

### Full Kernel

- Implement TMA-P32 first; retain TMA0 and CTA-P32 as controls.
- Measure whether ACTCOMP packed-SMEM reuse pays for TMA's raw-SMEM stage.
- Randomize method order and run a long no-scrub decode sequence after warmup.
- Replace synthetic routing with captured TOPK traces.

### Fixed-Address 64MiB Rewrite

Test a software-managed global staging window:

```text
source panel j       -> fixed global stage[0..64MiB) -> PF / consumer
source panel j+1     -> overwrite the same addresses -> PF / consumer
```

Required scoreboard:

```text
baseline             TMA0 reading the original source
copy source          streaming / evict-first
copy destination     L2 residency priority
consumers            staged TMA0, TMA-P32, CTA-P32
report               copy-only, consume-only, copy+consume
budget               resident stage + transient source footprint
```

The source must exceed 64MiB; current X is at most 7MiB.

This is not SiMRA Multi-RowCopy. SiMRA broadcasts one physical DDR4 source row
to as many as 31 destination rows through raw `ACT/PRE` timing violations.
CUDA exposes neither physical DRAM-row mapping nor raw DRAM commands. The
experiment is an ordinary global-to-global rewrite with fixed destination
addresses.

References: [SiMRA paper](https://arxiv.org/abs/2405.06081),
[SiMRA artifact](https://github.com/CMU-SAFARI/SiMRA-DRAM).

## Limits

- One GPU and one W13/S13 shape.
- Synthetic uniform routing.
- `N<=512`; one or two literal N256 Xb planes.
- No MMA, SwiGLU, `topk_W`, or output reduction.
- Fixed method order in existing artifacts.
- Cache operators and prefetches remain performance hints.
- Absolute milliseconds move with route and GPU state; paired same-seed
  reductions are the decision metric.

## Artifacts

- [`l2pfact_exp.py`](l2pfact_exp.py): CUDA source generator, runner, validator,
  TMA-standard summarizer, and ASCII report.
- [`l2pfact_flush_sensitivity_combined.csv`](l2pfact_flush_sensitivity_combined.csv):
  486 raw ACTCOMP rows for `flush={0,128,512}MB`.
- [`l2pfact_flush_sensitivity_summary.csv`](l2pfact_flush_sensitivity_summary.csv):
  TMA-standard cache-state summary.
- [`l2pfact_sweep_v2_combined.csv`](l2pfact_sweep_v2_combined.csv): 936 raw,
  corrected all-path stress rows.
- [`l2pfact_sweep_v2_summary.csv`](l2pfact_sweep_v2_summary.csv):
  TMA-standard all-path summary.
- [`l2pfact_sweep_combined.csv`](l2pfact_sweep_combined.csv) and
  [`l2pfact_sweep_summary.csv`](l2pfact_sweep_summary.csv): archived N128
  pre-producer lineage. They are not used for the live decision.

Summary CSVs expose both:

```text
median_time_reduction_vs_tma_pct
median_time_reduction_vs_transport_no_pf_pct
```

Only the first is an overall score.
