// SM120a W4A4 FF1 -> SwiGLU -> topk_W -> BF16 bulk-reduce proof.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
//     experiments/ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe.cu -o /tmp/ff1_w4a4_actreg_pipe
//
// Two-panel pipeline proof:
//   /tmp/ff1_w4a4 8 16 1024 4096 3 experiments/ff1_w4a4.csv
//
// Full CTA wave:
//   /tmp/ff1_w4a4 188 256 1024 512 20 experiments/ff1_w4a4.csv

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CTA 1024
#define TOPK 8
#define ACTREG 1

// One routed bit -> one all-ones/all-zero MMA fragment mask.
__device__ __constant__ uint32_t XbMaskLUT[2] = {0u, 0xffffffffu};

#define CUDA_CHECK(call) do {                                                \
    cudaError_t e__ = (call);                                                \
    if (e__ != cudaSuccess) {                                                \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                     cudaGetErrorString(e__));                               \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

/*
    Checkpoint contract

    W13[e,kt,i1024,plane4,t1024,v4]
      one plane = 16KB = CTA x one lane-native m16n8k64 A fragment
      A rows 0..7 = W1 I8, A rows 8..15 = W3 matching I8
      e2m1 sign bits are preflipped for the exp(-A) SwiGLU formulation

    S13[e,kt,i1024,plane4,half2,q256,pad512]
      plane span = 1024 u32 to preserve the live I<<2 checkpoint stride
      half0 -> p0 / W1 row g, half1 -> p1 / W3 row g+8

    X4[kt,n16,lane,{n0lo,n0hi,n1lo,n1hi}] u32
      one K64 / N16 = 32 lanes x 16B = 512B
      one ld.shared.v4 gives both N8 B fragments to every MMA lane

    Sx[kt,n16,g,{n0,n1}] u32
      one K64 / N16 = 8 token rows x two UE4M3x4 words = 64B

    Four 256T prefetch bands each own eight K64 rows per H2048 stage.
    PF(h+1) overlaps TMA(h); every sector is hinted once, with no PF wait.

    routed N16 pairing
      scan Xb for the next two live N16 tiles, adjacency not required
      load one W13/S13 packet -> 2 N8 MMAs per live N16
      W/S replay = sum_e ceil(live_n16[e]/2), no global partial workspace

    dynamic SMEM / two arbitrary live N16 tiles
      stage_k64 = min(H64,32)
      stage      = stage_k64*2*(512B X4 + 64B Sx)
      total      = 2*stage + CTA*4B fragment-native BF16 output

    sync
      one cold __syncthreads  mbarrier initialization only
      barrier 1 / 512T        complete P32 issue before matching TMA
      ready[stage] / 32 warps release/acquire payload reuse
      TMA state               release/acquire epoch publication
      output / four lanes     warp publish -> one 16B cp.reduce issuer

    post13 = BF16x2(W13GS[0/1] * XGSINV)

    topk_W[e,NP] BF16
      routed expert-major sidecar emitted with Xb

    X[I,NP] BF16
      zero before launch; this proof reduces topk-weighted SwiGLU directly
*/
__global__ __launch_bounds__(CTA, 1)
void ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe(
    const uint32_t* __restrict__ W13,
    const uint32_t* __restrict__ S13,
    const __nv_bfloat16* __restrict__ W13GS,
    const uint32_t* __restrict__ X4,
    const uint32_t* __restrict__ Sx,
    __nv_bfloat16 XGSINV,
    const uint32_t* __restrict__ Xb,
    const __nv_bfloat16* __restrict__ topk_W,
    __nv_bfloat16* __restrict__ X,
    int Xb_stride,
    int NP,
    int I,
    int H
) {

    extern __shared__ __align__(16) unsigned char smem[];
#if !ACTREG
    __shared__ __align__(16) uint64_t mbar[2];
    __shared__ __align__(16) uint64_t mstate[2];
    __shared__ uint32_t mepoch[2];
    __shared__ __align__(16) uint64_t ready[2];
#endif

    uint32_t live16, live16b;
#if !ACTREG
    uint32_t epoch = 0, ready_epoch0 = 0, ready_epoch1 = 0;
#endif
    uint32_t xbm0, xbm1, xbm2, xbm3;
    uint32_t a0, a1, a2, a3;
    uint32_t b00, b01, b10, b11;
#if ACTREG
    uint32_t c00, c01, c10, c11;
#endif
    uint32_t scaleA, scaleB0, scaleB1;
#if ACTREG
    uint32_t scaleC0, scaleC1;
#endif
    uint32_t tw, out, post13, xgsinv2;
    uint16_t selector0 = 0;
    float d00, d01, d10, d11;
    float s00, s01, s02, s03;
    float s10, s11, s12, s13;
    float t00, t01, t02, t03;
    float t10, t11, t12, t13;
    float post1, post3, x1, x3, o0, o1, topk0, topk1;
    int n16b;
#if ACTREG
#define ACT_OUT_OFFSET 0u
#else
    uint32_t stage_k64 = ((uint32_t)H >> 6) < 32u
                       ? ((uint32_t)H >> 6) : 32u;
    uint32_t stage_x4_bytes = stage_k64 << 10;
    uint32_t stage_bytes = stage_k64 * 1152u;
#define ACT_OUT_OFFSET (stage_bytes << 1)
#endif

    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(live16)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       Xb + (uint64_t)blockIdx.x * (uint32_t)Xb_stride))
                 : "memory");
    if (!live16) return;

    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(post13)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       W13GS + (uint64_t)blockIdx.x * 2u))
                 : "memory");
    xgsinv2 = __bfloat16_as_ushort(XGSINV);
    xgsinv2 |= xgsinv2 << 16;
    asm volatile(
        "{\n\t"
        ".reg .b32 zero;\n\t"
        "mov.b32 zero, 0;\n\t"
        "fma.rn.bf16x2 %0, %0, %1, zero;\n\t"
        "}"
        : "+r"(post13)
        : "r"(xgsinv2));
    post1 = __uint_as_float((post13 & 0xffffu) << 16);
    post3 = __uint_as_float(post13 & 0xffff0000u);

#if !ACTREG
    if (!threadIdx.x) {
        mepoch[0] = 0u;
        mepoch[1] = 0u;
    }
    if (threadIdx.x < 2) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 1;\n\t"
            "mbarrier.init.shared::cta.b64 [%1], 32;\n\t"
            "fence.mbarrier_init.release.cluster;\n\t"
            "fence.proxy.async.shared::cta;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(mbar + threadIdx.x)),
              "r"((uint32_t)__cvta_generic_to_shared(ready + threadIdx.x))
            : "memory"
        );
    }
    __syncthreads();
#endif

    for (int n16 = 0; n16 < (NP >> 4); n16++) {
        asm volatile("ldu.global.u32 %0, [%1];"
                     : "=r"(live16)
                     : "l"((uint64_t)__cvta_generic_to_global(
                           Xb + (uint64_t)blockIdx.x * (uint32_t)Xb_stride
                           + 1u + ((uint32_t)n16 >> 1)))
                     : "memory");
        live16 = (live16 >> ((n16 & 1) << 4)) & 0xffffu;
        if (!live16) continue;
        live16b = 0u;
        for (n16b = n16 + 1; n16b < (NP >> 4); n16b++) {
            asm volatile("ldu.global.u32 %0, [%1];"
                         : "=r"(live16b)
                         : "l"((uint64_t)__cvta_generic_to_global(
                               Xb + (uint64_t)blockIdx.x * (uint32_t)Xb_stride
                               + 1u + ((uint32_t)n16b >> 1)))
                         : "memory");
            live16b = (live16b >> ((n16b & 1) << 4)) & 0xffffu;
            if (live16b) break;
        }
        xbm0 = XbMaskLUT[(live16
                        >> (((uint32_t)threadIdx.x & 31u) >> 2)) & 1u];
        xbm1 = XbMaskLUT[(live16
                        >> (8u + (((uint32_t)threadIdx.x & 31u) >> 2))) & 1u];
        xbm2 = XbMaskLUT[(live16b
                        >> (((uint32_t)threadIdx.x & 31u) >> 2)) & 1u];
        xbm3 = XbMaskLUT[(live16b
                        >> (8u + (((uint32_t)threadIdx.x & 31u) >> 2))) & 1u];

        for (int i = 0; i < (I >> 10); i++) {
            // Four I256 planes are independent. Serializing them keeps only
            // one plane's paired four-N8 consumer set live.
            for (int plane = 0; plane < 4; plane++) {
                s00 = 0.0f; s01 = 0.0f; s02 = 0.0f; s03 = 0.0f;
                s10 = 0.0f; s11 = 0.0f; s12 = 0.0f; s13 = 0.0f;
                t00 = 0.0f; t01 = 0.0f; t02 = 0.0f; t03 = 0.0f;
                t10 = 0.0f; t11 = 0.0f; t12 = 0.0f; t13 = 0.0f;

#if !ACTREG
                epoch++;
                if (!threadIdx.x) {
                    if (ready_epoch0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .pred done;\n\t"
                            "stage_wait_%=:\n\t"
                            "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 "
                            "done, [%0], %1;\n\t"
                            "@!done bra stage_wait_%=;\n\t"
                            "}"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(ready)),
                              "r"((ready_epoch0 - 1u) & 1u)
                            : "memory");
                    }
                    asm volatile(
                        "{\n\t"
                        ".reg .b64 state;\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
                        "state, [%0], %4;\n\t"
                        "st.shared.u64 [%1], state;\n\t"
                        "st.release.cta.shared::cta.u32 [%2], %3;\n\t"
                        "}"
                        :
                        : "r"((uint32_t)__cvta_generic_to_shared(mbar)),
                          "r"((uint32_t)__cvta_generic_to_shared(mstate)),
                          "r"((uint32_t)__cvta_generic_to_shared(mepoch)),
                          "r"(epoch),
                          "r"(live16b ? stage_bytes : (stage_bytes >> 1))
                        : "memory"
                    );
                }

                // P32 current-panel warmup. Four 256T bands each cover eight
                // K64 rows; the hint is weak and deliberately has no wait.
                if (((uint32_t)threadIdx.x & 255u) < 128u
                    && (((uint32_t)threadIdx.x >> 8) << 3)
                       + (((uint32_t)threadIdx.x & 255u) >> 4)
                       < ((uint32_t)H >> 6)) {
                    asm volatile(
                        "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                        "cp.async.bulk.commit_group;"
                        :
                        : "l"((uint64_t)__cvta_generic_to_global(
                              X4
                              + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                  + (((uint32_t)threadIdx.x & 255u) >> 4))
                                  * ((uint32_t)NP >> 4) + (uint32_t)n16) << 7)
                              + (((uint32_t)threadIdx.x & 15u) << 3)))
                        : "memory"
                    );
                    if (live16b) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X4
                                  + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                      + (((uint32_t)threadIdx.x & 255u) >> 4))
                                      * ((uint32_t)NP >> 4) + (uint32_t)n16b) << 7)
                                  + (((uint32_t)threadIdx.x & 15u) << 3)))
                            : "memory");
                    }
                }
                if (((uint32_t)threadIdx.x & 255u) < 16u
                    && (((uint32_t)threadIdx.x >> 8) << 3)
                       + (((uint32_t)threadIdx.x & 255u) >> 1)
                       < ((uint32_t)H >> 6)) {
                    asm volatile(
                        "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                        "cp.async.bulk.commit_group;"
                        :
                        : "l"((uint64_t)__cvta_generic_to_global(
                              Sx
                              + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                  + (((uint32_t)threadIdx.x & 255u) >> 1))
                                  * ((uint32_t)NP >> 4) + (uint32_t)n16) << 4)
                              + (((uint32_t)threadIdx.x & 1u) << 3)))
                        : "memory"
                    );
                    if (live16b) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Sx
                                  + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                      + (((uint32_t)threadIdx.x & 255u) >> 1))
                                      * ((uint32_t)NP >> 4) + (uint32_t)n16b) << 4)
                                  + (((uint32_t)threadIdx.x & 1u) << 3)))
                            : "memory");
                    }
                }
                if (((uint32_t)threadIdx.x & 255u) < 128u) {
                    asm volatile(
                        "barrier.cta.sync 1, 512;" ::: "memory"
                    );
                }

                // One issuer per K64 row: one or two routed N16 payloads.
                if (((uint32_t)threadIdx.x & 255u) < 8u
                    && (((uint32_t)threadIdx.x >> 8) << 3)
                       + ((uint32_t)threadIdx.x & 255u)
                       < ((uint32_t)H >> 6)) {
                    asm volatile(
                        "{\n\t"
                        ".reg .pred pair;\n\t"
                        "setp.ne.u32 pair, %5, 0;\n\t"
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%2], 512, [%4];\n\t"
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%1], [%3], 64, [%4];\n\t"
                        "@!pair bra pair_done_%=;\n\t"
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0 + 512], [%6], 512, [%4];\n\t"
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%1 + 64], [%7], 64, [%4];\n\t"
                        "pair_done_%=:\n\t"
                        "}"
                        :
                        : "r"((uint32_t)__cvta_generic_to_shared(
                              smem + (((((uint32_t)threadIdx.x >> 8) << 3)
                                  + ((uint32_t)threadIdx.x & 255u)) << 10))),
                          "r"((uint32_t)__cvta_generic_to_shared(
                              smem + stage_x4_bytes
                              + (((((uint32_t)threadIdx.x >> 8) << 3)
                                  + ((uint32_t)threadIdx.x & 255u)) << 7))),
                          "l"((uint64_t)__cvta_generic_to_global(
                              X4
                              + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                  + ((uint32_t)threadIdx.x & 255u))
                                  * ((uint32_t)NP >> 4) + (uint32_t)n16) << 7))),
                          "l"((uint64_t)__cvta_generic_to_global(
                              Sx
                              + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                  + ((uint32_t)threadIdx.x & 255u))
                                  * ((uint32_t)NP >> 4) + (uint32_t)n16) << 4))),
                          "r"((uint32_t)__cvta_generic_to_shared(mbar)),
                          "r"((uint32_t)(live16b != 0u)),
                          "l"((uint64_t)__cvta_generic_to_global(
                              X4
                              + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                  + ((uint32_t)threadIdx.x & 255u))
                                  * ((uint32_t)NP >> 4) + (uint32_t)n16b) << 7))),
                          "l"((uint64_t)__cvta_generic_to_global(
                              Sx
                              + ((((uint64_t)(((uint32_t)threadIdx.x >> 8) << 3)
                                  + ((uint32_t)threadIdx.x & 255u))
                                  * ((uint32_t)NP >> 4) + (uint32_t)n16b) << 4)))
                        : "memory"
                    );
                }

                // Lead-one PF: panel 1 warms while panel 0 TMA is in flight.
                if (((H + 2047) >> 11) > 1) {
                    if (((uint32_t)threadIdx.x & 255u) < 128u
                        && 32u + (((uint32_t)threadIdx.x >> 8) << 3)
                           + (((uint32_t)threadIdx.x & 255u) >> 4)
                           < ((uint32_t)H >> 6)) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X4
                                  + ((uint64_t)(32u
                                      + (((uint32_t)threadIdx.x >> 8) << 3)
                                      + (((uint32_t)threadIdx.x & 255u) >> 4))
                                      * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                  + (((uint32_t)threadIdx.x & 15u) << 3)))
                            : "memory"
                        );
                        if (live16b) {
                            asm volatile(
                                "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                "cp.async.bulk.commit_group;"
                                :
                                : "l"((uint64_t)__cvta_generic_to_global(
                                      X4
                                      + ((uint64_t)(32u
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + (((uint32_t)threadIdx.x & 255u) >> 4))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 128u
                                      + (((uint32_t)threadIdx.x & 15u) << 3)))
                                : "memory");
                        }
                    }
                    if (((uint32_t)threadIdx.x & 255u) < 16u
                        && 32u + (((uint32_t)threadIdx.x >> 8) << 3)
                           + (((uint32_t)threadIdx.x & 255u) >> 1)
                           < ((uint32_t)H >> 6)) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Sx
                                  + ((uint64_t)(32u
                                      + (((uint32_t)threadIdx.x >> 8) << 3)
                                      + (((uint32_t)threadIdx.x & 255u) >> 1))
                                      * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                  + (((uint32_t)threadIdx.x & 1u) << 3)))
                            : "memory"
                        );
                        if (live16b) {
                            asm volatile(
                                "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                "cp.async.bulk.commit_group;"
                                :
                                : "l"((uint64_t)__cvta_generic_to_global(
                                      Sx
                                      + ((uint64_t)(32u
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + (((uint32_t)threadIdx.x & 255u) >> 1))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 16u
                                      + (((uint32_t)threadIdx.x & 1u) << 3)))
                                : "memory");
                        }
                    }
                    if (((uint32_t)threadIdx.x & 255u) < 128u)
                        asm volatile("barrier.cta.sync 1, 512;" ::: "memory");
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 seen;\n\t"
                    ".reg .b64 state;\n\t"
                    ".reg .pred ready, done;\n\t"
                    "epoch_wait_%=:\n\t"
                    "ld.acquire.cta.shared::cta.u32 seen, [%2];\n\t"
                    "setp.eq.u32 ready, seen, %3;\n\t"
                    "@!ready bra epoch_wait_%=;\n\t"
                    "ld.shared.u64 state, [%1];\n\t"
                    "tma_wait_%=:\n\t"
                    "mbarrier.try_wait.acquire.cta.shared::cta.b64 "
                    "done, [%0], state;\n\t"
                    "@!done bra tma_wait_%=;\n\t"
                    "}"
                    :
                    : "r"((uint32_t)__cvta_generic_to_shared(mbar)),
                      "r"((uint32_t)__cvta_generic_to_shared(mstate)),
                      "r"((uint32_t)__cvta_generic_to_shared(mepoch)),
                      "r"(epoch)
                    : "memory"
                );
#endif

                for (int hp = 0; hp < ((H + 2047) >> 11); hp++) {
#if !ACTREG
                    // Fill the alternate dynamic H2048-cap stage while this
                    // stage feeds MMA.
                    if (hp + 1 < ((H + 2047) >> 11)) {
                        epoch++;
                        if (!threadIdx.x) {
                            if (((hp + 1) & 1) ? ready_epoch1 : ready_epoch0) {
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred done;\n\t"
                                    "stage_wait_%=:\n\t"
                                    "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 "
                                    "done, [%0], %1;\n\t"
                                    "@!done bra stage_wait_%=;\n\t"
                                    "}"
                                    :
                                    : "r"((uint32_t)__cvta_generic_to_shared(
                                          ready + ((hp + 1) & 1))),
                                      "r"(((((hp + 1) & 1)
                                          ? ready_epoch1 : ready_epoch0) - 1u) & 1u)
                                    : "memory");
                            }
                            asm volatile(
                                "{\n\t"
                                ".reg .b64 state;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
                                "state, [%0], %4;\n\t"
                                "st.shared.u64 [%1], state;\n\t"
                                "st.release.cta.shared::cta.u32 [%2], %3;\n\t"
                                "}"
                                :
                                : "r"((uint32_t)__cvta_generic_to_shared(
                                      mbar + ((hp + 1) & 1))),
                                  "r"((uint32_t)__cvta_generic_to_shared(
                                      mstate + ((hp + 1) & 1))),
                                  "r"((uint32_t)__cvta_generic_to_shared(
                                      mepoch + ((hp + 1) & 1))),
                                  "r"(epoch),
                                  "r"(((((uint32_t)H >> 6)
                                      - ((uint32_t)(hp + 1) << 5)) < 32u
                                      ? (((uint32_t)H >> 6)
                                         - ((uint32_t)(hp + 1) << 5))
                                      : 32u) * (live16b ? 1152u : 576u))
                                : "memory"
                            );
                        }
                        if (((uint32_t)threadIdx.x & 255u) < 8u
                            && ((uint32_t)(hp + 1) << 5)
                               + (((uint32_t)threadIdx.x >> 8) << 3)
                               + ((uint32_t)threadIdx.x & 255u)
                               < ((uint32_t)H >> 6)) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 seen;\n\t"
                                ".reg .pred ready;\n\t"
                                "issue_wait_%=:\n\t"
                                "ld.acquire.cta.shared::cta.u32 seen, [%0];\n\t"
                                "setp.eq.u32 ready, seen, %1;\n\t"
                                "@!ready bra issue_wait_%=;\n\t"
                                "}"
                                :
                                : "r"((uint32_t)__cvta_generic_to_shared(
                                      mepoch + ((hp + 1) & 1))),
                                  "r"(epoch)
                                : "memory"
                            );
                            asm volatile(
                                "{\n\t"
                                ".reg .pred pair;\n\t"
                                "setp.ne.u32 pair, %5, 0;\n\t"
                                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                "[%0], [%2], 512, [%4];\n\t"
                                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                "[%1], [%3], 64, [%4];\n\t"
                                "@!pair bra pair_done_%=;\n\t"
                                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                "[%0 + 512], [%6], 512, [%4];\n\t"
                                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                "[%1 + 64], [%7], 64, [%4];\n\t"
                                "pair_done_%=:\n\t"
                                "}"
                                :
                                : "r"((uint32_t)__cvta_generic_to_shared(
                                      smem + ((uint32_t)((hp + 1) & 1) * stage_bytes)
                                      + (((((uint32_t)threadIdx.x >> 8) << 3)
                                          + ((uint32_t)threadIdx.x & 255u)) << 10))),
                                  "r"((uint32_t)__cvta_generic_to_shared(
                                      smem + ((uint32_t)((hp + 1) & 1) * stage_bytes)
                                      + stage_x4_bytes
                                      + (((((uint32_t)threadIdx.x >> 8) << 3)
                                          + ((uint32_t)threadIdx.x & 255u)) << 7))),
                                  "l"((uint64_t)__cvta_generic_to_global(
                                      X4
                                      + ((uint64_t)(((hp + 1) << 5)
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + ((uint32_t)threadIdx.x & 255u))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u)),
                                  "l"((uint64_t)__cvta_generic_to_global(
                                      Sx
                                      + ((uint64_t)(((hp + 1) << 5)
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + ((uint32_t)threadIdx.x & 255u))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u)),
                                  "r"((uint32_t)__cvta_generic_to_shared(
                                      mbar + ((hp + 1) & 1))),
                                  "r"((uint32_t)(live16b != 0u)),
                                  "l"((uint64_t)__cvta_generic_to_global(
                                      X4
                                      + ((uint64_t)(((hp + 1) << 5)
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + ((uint32_t)threadIdx.x & 255u))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 128u)),
                                  "l"((uint64_t)__cvta_generic_to_global(
                                      Sx
                                      + ((uint64_t)(((hp + 1) << 5)
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + ((uint32_t)threadIdx.x & 255u))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 16u))
                                : "memory"
                            );
                        }

                        // TMA consumes PF(hp+1); PF(hp+2) overlaps its flight
                        // and the current panel's 64 MMA issue slots.
                        if (hp + 2 < ((H + 2047) >> 11)) {
                            if (((uint32_t)threadIdx.x & 255u) < 128u
                                && ((uint32_t)(hp + 2) << 5)
                                   + (((uint32_t)threadIdx.x >> 8) << 3)
                                   + (((uint32_t)threadIdx.x & 255u) >> 4)
                                   < ((uint32_t)H >> 6)) {
                                asm volatile(
                                    "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                    "cp.async.bulk.commit_group;"
                                    :
                                    : "l"((uint64_t)__cvta_generic_to_global(
                                          X4
                                          + ((uint64_t)(((hp + 2) << 5)
                                              + (((uint32_t)threadIdx.x >> 8) << 3)
                                              + (((uint32_t)threadIdx.x & 255u) >> 4))
                                              * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                          + (((uint32_t)threadIdx.x & 15u) << 3)))
                                    : "memory"
                                );
                                if (live16b) {
                                    asm volatile(
                                        "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                        "cp.async.bulk.commit_group;"
                                        :
                                        : "l"((uint64_t)__cvta_generic_to_global(
                                              X4
                                              + ((uint64_t)(((hp + 2) << 5)
                                                  + (((uint32_t)threadIdx.x >> 8) << 3)
                                                  + (((uint32_t)threadIdx.x & 255u) >> 4))
                                                  * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 128u
                                              + (((uint32_t)threadIdx.x & 15u) << 3)))
                                        : "memory");
                                }
                            }
                            if (((uint32_t)threadIdx.x & 255u) < 16u
                                && ((uint32_t)(hp + 2) << 5)
                                   + (((uint32_t)threadIdx.x >> 8) << 3)
                                   + (((uint32_t)threadIdx.x & 255u) >> 1)
                                   < ((uint32_t)H >> 6)) {
                                asm volatile(
                                    "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                    "cp.async.bulk.commit_group;"
                                    :
                                    : "l"((uint64_t)__cvta_generic_to_global(
                                          Sx
                                          + ((uint64_t)(((hp + 2) << 5)
                                              + (((uint32_t)threadIdx.x >> 8) << 3)
                                              + (((uint32_t)threadIdx.x & 255u) >> 1))
                                              * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                          + (((uint32_t)threadIdx.x & 1u) << 3)))
                                    : "memory"
                                );
                                if (live16b) {
                                    asm volatile(
                                        "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                        "cp.async.bulk.commit_group;"
                                        :
                                        : "l"((uint64_t)__cvta_generic_to_global(
                                              Sx
                                              + ((uint64_t)(((hp + 2) << 5)
                                                  + (((uint32_t)threadIdx.x >> 8) << 3)
                                                  + (((uint32_t)threadIdx.x & 255u) >> 1))
                                                  * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 16u
                                              + (((uint32_t)threadIdx.x & 1u) << 3)))
                                        : "memory");
                                }
                            }
                            // The ready epoch below protects payload reuse;
                            // prefetch itself remains a weak residency hint.
                        }
                    }

#else
                    // Every routed lane rank is identical across all 32 warps.
                    // Hint each current K64/N16 payload once at CTA scope;
                    // matching warp loads below land directly in B registers.
                    if (((uint32_t)threadIdx.x & 255u) < 128u
                        && ((hp << 5)
                            + (((uint32_t)threadIdx.x >> 8) << 3)
                            + (((uint32_t)threadIdx.x & 255u) >> 4)
                            < ((uint32_t)H >> 6))) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X4
                                  + ((uint64_t)((hp << 5)
                                      + (((uint32_t)threadIdx.x >> 8) << 3)
                                      + (((uint32_t)threadIdx.x & 255u) >> 4))
                                      * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                  + (((uint32_t)threadIdx.x & 15u) << 3)))
                            : "memory");
                        if (live16b) {
                            asm volatile(
                                "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                "cp.async.bulk.commit_group;"
                                :
                                : "l"((uint64_t)__cvta_generic_to_global(
                                      X4
                                      + ((uint64_t)((hp << 5)
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + (((uint32_t)threadIdx.x & 255u) >> 4))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 128u
                                      + (((uint32_t)threadIdx.x & 15u) << 3)))
                                : "memory");
                        }
                    }
                    if (((uint32_t)threadIdx.x & 255u) < 16u
                        && ((hp << 5)
                            + (((uint32_t)threadIdx.x >> 8) << 3)
                            + (((uint32_t)threadIdx.x & 255u) >> 1)
                            < ((uint32_t)H >> 6))) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Sx
                                  + ((uint64_t)((hp << 5)
                                      + (((uint32_t)threadIdx.x >> 8) << 3)
                                      + (((uint32_t)threadIdx.x & 255u) >> 1))
                                      * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                  + (((uint32_t)threadIdx.x & 1u) << 3)))
                            : "memory");
                        if (live16b) {
                            asm volatile(
                                "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                "cp.async.bulk.commit_group;"
                                :
                                : "l"((uint64_t)__cvta_generic_to_global(
                                      Sx
                                      + ((uint64_t)((hp << 5)
                                          + (((uint32_t)threadIdx.x >> 8) << 3)
                                          + (((uint32_t)threadIdx.x & 255u) >> 1))
                                          * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 16u
                                      + (((uint32_t)threadIdx.x & 1u) << 3)))
                            : "memory");
                        }
                    }

                    // warp0..31: B0[live lanes] -> B1[live lanes]
                    //             -> B2[live lanes] -> B3[live lanes]
                    // Each warp materializes its own complete MMA B fragment.
                    b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                    if (xbm0) {
                        asm volatile(
                            "ld.global.ca.nc.u32 %0, [%2];\n\t"
                            "ld.global.ca.nc.u32 %1, [%2 + 4];"
                            : "=r"(b00), "=r"(b01)
                            : "l"(X4
                                + ((uint64_t)(hp << 5)
                                    * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                + (((uint32_t)threadIdx.x & 31u) << 2))
                            : "memory");
                    }
                    if (xbm1) {
                        asm volatile(
                            "ld.global.ca.nc.u32 %0, [%2];\n\t"
                            "ld.global.ca.nc.u32 %1, [%2 + 4];"
                            : "=r"(b10), "=r"(b11)
                            : "l"(X4
                                + ((uint64_t)(hp << 5)
                                    * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                + (((uint32_t)threadIdx.x & 31u) << 2) + 2u)
                            : "memory");
                    }
                    scaleB0 = 0u; scaleB1 = 0u;
                    if (!((uint32_t)threadIdx.x & 3u)) {
                        if (xbm0) {
                            asm volatile(
                                "ld.global.ca.nc.u32 %0, [%1];"
                                : "=r"(scaleB0)
                                : "l"(Sx
                                    + ((uint64_t)(hp << 5)
                                        * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                    + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1))
                                : "memory");
                        }
                        if (xbm1) {
                            asm volatile(
                                "ld.global.ca.nc.u32 %0, [%1];"
                                : "=r"(scaleB1)
                                : "l"(Sx
                                    + ((uint64_t)(hp << 5)
                                        * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                    + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1) + 1u)
                                : "memory");
                        }
                    }
#endif

                    for (int k = 0; k < 32 && ((hp << 5) + k) < (H >> 6); k++) {
#if ACTREG
                        c00 = 0u; c01 = 0u; c10 = 0u; c11 = 0u;
                        if (k + 1 < 32 && ((hp << 5) + k + 1) < (H >> 6) && xbm0) {
                            asm volatile(
                                "ld.global.ca.nc.u32 %0, [%2];\n\t"
                                "ld.global.ca.nc.u32 %1, [%2 + 4];"
                                : "=r"(c00), "=r"(c01)
                                : "l"(X4
                                    + ((uint64_t)((hp << 5) + k + 1)
                                        * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                    + (((uint32_t)threadIdx.x & 31u) << 2))
                                : "memory");
                        }
                        if (k + 1 < 32 && ((hp << 5) + k + 1) < (H >> 6) && xbm1) {
                            asm volatile(
                                "ld.global.ca.nc.u32 %0, [%2];\n\t"
                                "ld.global.ca.nc.u32 %1, [%2 + 4];"
                                : "=r"(c10), "=r"(c11)
                                : "l"(X4
                                    + ((uint64_t)((hp << 5) + k + 1)
                                        * ((uint32_t)NP >> 4) + (uint32_t)n16) * 128u
                                    + (((uint32_t)threadIdx.x & 31u) << 2) + 2u)
                                : "memory");
                        }
                        scaleC0 = 0u; scaleC1 = 0u;
                        if (k + 1 < 32 && ((hp << 5) + k + 1) < (H >> 6)
                            && !((uint32_t)threadIdx.x & 3u)) {
                            if (xbm0) {
                                asm volatile(
                                    "ld.global.ca.nc.u32 %0, [%1];"
                                    : "=r"(scaleC0)
                                    : "l"(Sx
                                        + ((uint64_t)((hp << 5) + k + 1)
                                            * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                        + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1))
                                    : "memory");
                            }
                            if (xbm1) {
                                asm volatile(
                                    "ld.global.ca.nc.u32 %0, [%1];"
                                    : "=r"(scaleC1)
                                    : "l"(Sx
                                        + ((uint64_t)((hp << 5) + k + 1)
                                            * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                        + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1) + 1u)
                                    : "memory");
                            }
                        }
#else
                        asm volatile(
                            "ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11)
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + ((uint32_t)(hp & 1) * stage_bytes)
                                  + ((uint32_t)k << 10)
                                  + (((uint32_t)threadIdx.x & 31u) << 4)))
                            : "memory"
                        );
                        scaleB0 = 0u; scaleB1 = 0u;
                        if (!((uint32_t)threadIdx.x & 3u)) {
                            asm volatile(
                                "ld.shared.v2.u32 {%0,%1}, [%2];"
                                : "=r"(scaleB0), "=r"(scaleB1)
                                : "r"((uint32_t)__cvta_generic_to_shared(
                                      smem + ((uint32_t)(hp & 1) * stage_bytes)
                                      + stage_x4_bytes + ((uint32_t)k << 7)
                                      + ((((uint32_t)threadIdx.x & 31u) >> 2) << 3)))
                                : "memory"
                            );
                        }
                        b00 &= xbm0; b01 &= xbm0; scaleB0 &= xbm0;
                        b10 &= xbm1; b11 &= xbm1; scaleB1 &= xbm1;
#endif

                        asm volatile(
                            "ld.global.cs.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n\t"
                            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                            : "l"(W13
                                + (uint64_t)blockIdx.x * ((uint32_t)H >> 6)
                                    * ((uint32_t)I << 4)
                                + (uint64_t)((hp << 5) + k) * ((uint32_t)I << 4)
                                + ((uint64_t)i << 14)
                                + ((uint64_t)plane << 12)
                                + ((uint64_t)threadIdx.x << 2))
                            : "memory");

                        // W1/W3 were sign-negated once while packing W13; their
                        // signs cancel in the SwiGLU elementwise product.

                        scaleA = 0u;
                        if ((threadIdx.x & 3) < 2) {
                            asm volatile(
                                "ld.global.cs.nc.b32 %0, [%1];\n\t"
                                : "=r"(scaleA)
                                : "l"(S13
                                    + (uint64_t)blockIdx.x * ((uint32_t)H >> 6)
                                        * ((uint32_t)I << 2)
                                    + (uint64_t)((hp << 5) + k)
                                        * ((uint32_t)I << 2)
                                    + ((uint64_t)i << 12)
                                    + ((uint64_t)plane << 10)
                                    + ((uint64_t)(threadIdx.x & 3) << 8)
                                    + ((uint32_t)threadIdx.x >> 2))
                                : "memory");
                        }

                        // One W/S fragment, up to four N8 consumers.
                        asm volatile(
                            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n\t"
                            : "+f"(s00), "+f"(s01), "+f"(s02), "+f"(s03)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b00), "r"(b01), "r"(scaleA),
                              "h"(selector0), "h"(selector0), "r"(scaleB0),
                              "h"(selector0), "h"(selector0)
                        );
                        asm volatile(
                            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n\t"
                            : "+f"(s10), "+f"(s11), "+f"(s12), "+f"(s13)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b10), "r"(b11), "r"(scaleA),
                              "h"(selector0), "h"(selector0), "r"(scaleB1),
                              "h"(selector0), "h"(selector0)
                        );
                        if (live16b) {
#if ACTREG
                            b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                            if (xbm2) {
                                asm volatile(
                                    "ld.global.ca.nc.u32 %0, [%2];\n\t"
                                    "ld.global.ca.nc.u32 %1, [%2 + 4];"
                                    : "=r"(b00), "=r"(b01)
                                    : "l"(X4
                                        + ((uint64_t)((hp << 5) + k)
                                            * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 128u
                                        + (((uint32_t)threadIdx.x & 31u) << 2))
                                    : "memory");
                            }
                            if (xbm3) {
                                asm volatile(
                                    "ld.global.ca.nc.u32 %0, [%2];\n\t"
                                    "ld.global.ca.nc.u32 %1, [%2 + 4];"
                                    : "=r"(b10), "=r"(b11)
                                    : "l"(X4
                                        + ((uint64_t)((hp << 5) + k)
                                            * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 128u
                                        + (((uint32_t)threadIdx.x & 31u) << 2) + 2u)
                                    : "memory");
                            }
                            scaleB0 = 0u; scaleB1 = 0u;
                            if (!((uint32_t)threadIdx.x & 3u)) {
                                if (xbm2) {
                                    asm volatile(
                                        "ld.global.ca.nc.u32 %0, [%1];"
                                        : "=r"(scaleB0)
                                        : "l"(Sx
                                            + ((uint64_t)((hp << 5) + k)
                                                * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 16u
                                            + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1))
                                        : "memory");
                                }
                                if (xbm3) {
                                    asm volatile(
                                        "ld.global.ca.nc.u32 %0, [%1];"
                                        : "=r"(scaleB1)
                                        : "l"(Sx
                                            + ((uint64_t)((hp << 5) + k)
                                                * ((uint32_t)NP >> 4) + (uint32_t)n16b) * 16u
                                            + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1) + 1u)
                                        : "memory");
                                }
                            }
#else
                            asm volatile(
                                "ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(b00), "=r"(b01), "=r"(b10), "=r"(b11)
                                : "r"((uint32_t)__cvta_generic_to_shared(
                                      smem + ((uint32_t)(hp & 1) * stage_bytes)
                                      + ((uint32_t)k << 10) + 512u
                                      + (((uint32_t)threadIdx.x & 31u) << 4)))
                                : "memory");
                            scaleB0 = 0u; scaleB1 = 0u;
                            if (!((uint32_t)threadIdx.x & 3u)) {
                                asm volatile(
                                    "ld.shared.v2.u32 {%0,%1}, [%2];"
                                    : "=r"(scaleB0), "=r"(scaleB1)
                                    : "r"((uint32_t)__cvta_generic_to_shared(
                                          smem + ((uint32_t)(hp & 1) * stage_bytes)
                                          + stage_x4_bytes + ((uint32_t)k << 7) + 64u
                                          + ((((uint32_t)threadIdx.x & 31u) >> 2) << 3)))
                                    : "memory");
                            }
                            b00 &= xbm2; b01 &= xbm2; scaleB0 &= xbm2;
                            b10 &= xbm3; b11 &= xbm3; scaleB1 &= xbm3;
#endif
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n\t"
                                : "+f"(t00), "+f"(t01), "+f"(t02), "+f"(t03)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b00), "r"(b01), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB0),
                                  "h"(selector0), "h"(selector0));
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n\t"
                                : "+f"(t10), "+f"(t11), "+f"(t12), "+f"(t13)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b10), "r"(b11), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB1),
                                  "h"(selector0), "h"(selector0));
                        }
#if ACTREG
                        if (k + 1 < 32 && ((hp << 5) + k + 1) < (H >> 6)) {
                            b00 = c00; b01 = c01; b10 = c10; b11 = c11;
                            scaleB0 = scaleC0; scaleB1 = scaleC1;
                        }
#endif
                    }

#if !ACTREG
                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                    if (!((uint32_t)threadIdx.x & 31u)) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  ready + (hp & 1)))
                            : "memory");
                    }
                    if (hp & 1) ready_epoch1++;
                    else ready_epoch0++;

                    if (hp + 1 < ((H + 2047) >> 11)) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 seen;\n\t"
                            ".reg .b64 state;\n\t"
                            ".reg .pred ready, done;\n\t"
                            "epoch_wait_%=:\n\t"
                            "ld.acquire.cta.shared::cta.u32 seen, [%2];\n\t"
                            "setp.eq.u32 ready, seen, %3;\n\t"
                            "@!ready bra epoch_wait_%=;\n\t"
                            "ld.shared.u64 state, [%1];\n\t"
                            "tma_wait_%=:\n\t"
                            "mbarrier.try_wait.acquire.cta.shared::cta.b64 "
                            "done, [%0], state;\n\t"
                            "@!done bra tma_wait_%=;\n\t"
                            "}"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  mbar + ((hp + 1) & 1))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  mstate + ((hp + 1) & 1))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  mepoch + ((hp + 1) & 1))),
                              "r"(epoch)
                            : "memory"
                        );
                    }
#endif
                }

                if (live16 & 0xffu) {
                    tw = 0u;
                    if (!((uint32_t)threadIdx.x & 28u))
                        tw = *reinterpret_cast<const uint32_t*>(
                            topk_W
                            + (uint64_t)blockIdx.x * (uint32_t)NP
                            + (n16 << 4) + ((threadIdx.x & 3) << 1));
                    tw = __shfl_sync(0xffffffffu, tw, threadIdx.x & 3);
                    topk0 = __uint_as_float((tw & 0xffffu) << 16);
                    topk1 = __uint_as_float(tw & 0xffff0000u);

                    x1 = s00 * post1; x3 = s02 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d00)
                                 : "f"(x1 * 1.4426950408889634f));
                    d00 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d00));
                    o0 = (x1 * x3 * d00) * topk0;
                    x1 = s01 * post1; x3 = s03 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d01)
                                 : "f"(x1 * 1.4426950408889634f));
                    d01 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d01));
                    o1 = (x1 * x3 * d01) * topk1;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                 : "=r"(out) : "f"(o1), "f"(o0));
                    reinterpret_cast<uint32_t*>(
                        smem + ACT_OUT_OFFSET)
                        [threadIdx.x] = out;

                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                    if (!((uint32_t)threadIdx.x & 3u)) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;\n\t"
                            "cp.async.bulk.wait_group 0;\n\t"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X
                                  + (uint64_t)((i << 10) + (plane << 8)
                                    + (threadIdx.x >> 2)) * (uint32_t)NP
                                  + (n16 << 4))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + ACT_OUT_OFFSET
                                  + ((uint32_t)(threadIdx.x >> 2) << 4)))
                            : "memory"
                        );
                    }
                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                }

                if (live16 & 0xff00u) {
                    tw = 0u;
                    if (!((uint32_t)threadIdx.x & 28u))
                        tw = *reinterpret_cast<const uint32_t*>(
                            topk_W
                            + (uint64_t)blockIdx.x * (uint32_t)NP
                            + (n16 << 4) + 8u + ((threadIdx.x & 3) << 1));
                    tw = __shfl_sync(0xffffffffu, tw, threadIdx.x & 3);
                    topk0 = __uint_as_float((tw & 0xffffu) << 16);
                    topk1 = __uint_as_float(tw & 0xffff0000u);

                    x1 = s10 * post1; x3 = s12 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d10)
                                 : "f"(x1 * 1.4426950408889634f));
                    d10 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d10));
                    o0 = (x1 * x3 * d10) * topk0;
                    x1 = s11 * post1; x3 = s13 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d11)
                                 : "f"(x1 * 1.4426950408889634f));
                    d11 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d11));
                    o1 = (x1 * x3 * d11) * topk1;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                 : "=r"(out) : "f"(o1), "f"(o0));
                    reinterpret_cast<uint32_t*>(
                        smem + ACT_OUT_OFFSET)
                        [threadIdx.x] = out;

                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                    if (!((uint32_t)threadIdx.x & 3u)) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;\n\t"
                            "cp.async.bulk.wait_group 0;\n\t"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X
                                  + (uint64_t)((i << 10) + (plane << 8)
                                    + (threadIdx.x >> 2)) * (uint32_t)NP
                                  + (n16 << 4) + 8u)),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + ACT_OUT_OFFSET
                                  + ((uint32_t)(threadIdx.x >> 2) << 4)))
                            : "memory"
                        );
                    }
                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                }

                if (live16b & 0xffu) {
                    tw = 0u;
                    if (!((uint32_t)threadIdx.x & 28u))
                        tw = *reinterpret_cast<const uint32_t*>(
                            topk_W
                            + (uint64_t)blockIdx.x * (uint32_t)NP
                            + (n16b << 4) + ((threadIdx.x & 3) << 1));
                    tw = __shfl_sync(0xffffffffu, tw, threadIdx.x & 3);
                    topk0 = __uint_as_float((tw & 0xffffu) << 16);
                    topk1 = __uint_as_float(tw & 0xffff0000u);

                    x1 = t00 * post1; x3 = t02 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d00)
                                 : "f"(x1 * 1.4426950408889634f));
                    d00 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d00));
                    o0 = (x1 * x3 * d00) * topk0;
                    x1 = t01 * post1; x3 = t03 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d01)
                                 : "f"(x1 * 1.4426950408889634f));
                    d01 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d01));
                    o1 = (x1 * x3 * d01) * topk1;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                 : "=r"(out) : "f"(o1), "f"(o0));
                    reinterpret_cast<uint32_t*>(
                        smem + ACT_OUT_OFFSET)
                        [threadIdx.x] = out;

                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                    if (!((uint32_t)threadIdx.x & 3u)) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;\n\t"
                            "cp.async.bulk.wait_group 0;\n\t"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X
                                  + (uint64_t)((i << 10) + (plane << 8)
                                    + (threadIdx.x >> 2)) * (uint32_t)NP
                                  + (n16b << 4))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + ACT_OUT_OFFSET
                                  + ((uint32_t)(threadIdx.x >> 2) << 4)))
                            : "memory");
                    }
                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                }

                if (live16b & 0xff00u) {
                    tw = 0u;
                    if (!((uint32_t)threadIdx.x & 28u))
                        tw = *reinterpret_cast<const uint32_t*>(
                            topk_W
                            + (uint64_t)blockIdx.x * (uint32_t)NP
                            + (n16b << 4) + 8u + ((threadIdx.x & 3) << 1));
                    tw = __shfl_sync(0xffffffffu, tw, threadIdx.x & 3);
                    topk0 = __uint_as_float((tw & 0xffffu) << 16);
                    topk1 = __uint_as_float(tw & 0xffff0000u);

                    x1 = t10 * post1; x3 = t12 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d10)
                                 : "f"(x1 * 1.4426950408889634f));
                    d10 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d10));
                    o0 = (x1 * x3 * d10) * topk0;
                    x1 = t11 * post1; x3 = t13 * post3;
                    asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                 : "=f"(d11)
                                 : "f"(x1 * 1.4426950408889634f));
                    d11 += 1.0f;
                    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d11));
                    o1 = (x1 * x3 * d11) * topk1;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                 : "=r"(out) : "f"(o1), "f"(o0));
                    reinterpret_cast<uint32_t*>(
                        smem + ACT_OUT_OFFSET)
                        [threadIdx.x] = out;

                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                    if (!((uint32_t)threadIdx.x & 3u)) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;\n\t"
                            "cp.async.bulk.wait_group 0;\n\t"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X
                                  + (uint64_t)((i << 10) + (plane << 8)
                                    + (threadIdx.x >> 2)) * (uint32_t)NP
                                  + (n16b << 4) + 8u)),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + ACT_OUT_OFFSET
                                  + ((uint32_t)(threadIdx.x >> 2) << 4)))
                            : "memory");
                    }
                    asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                }
            }
        }
        n16 = n16b;
    }
}
#undef ACT_OUT_OFFSET

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void init_W13(uint32_t* p, uint64_t n) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t plane = (uint32_t)(x >> 12) & 3u;
        uint32_t q = (x & 1u) ? 2u : plane + 1u;
        p[x] = q * 0x11111111u ^ 0x88888888u;
    }
}

__global__ void init_u32(uint32_t* p, uint64_t n, uint32_t value) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; x < n; x += stride) p[x] = value;
}

__global__ void init_bf16(uint16_t* p, uint64_t n, uint16_t value) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; x < n; x += stride) p[x] = value;
}

__global__ void init_X4(uint32_t* p, uint64_t n, uint32_t N16) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t token = (uint32_t)((x >> 7) % N16) * 16u
                       + (((uint32_t)x >> 2) & 31u) / 4u
                       + (((uint32_t)x & 3u) >> 1) * 8u;
        uint32_t q = 1u + ((token + (token >> 3)) & 3u);
        p[x] = q * 0x11111111u;
    }
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 9) {
        std::fprintf(stderr,
            "usage: %s E N I H [live_experts route_seed] iters output.csv\n",
            argv[0]);
        return 2;
    }

    int E = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int I = std::atoi(argv[3]);
    int H = std::atoi(argv[4]);
    int live_expert_target = argc == 9 ? std::atoi(argv[5])
                                      : std::min(E, N * TOPK);
    uint32_t route_seed = argc == 9
        ? (uint32_t)std::strtoul(argv[6], nullptr, 0) : 0x6a09e667u;
    int iters = std::atoi(argv[argc - 2]);
    char* csv_path = argv[argc - 1];
    int NP = (N + 15) & ~15;
    int Xb_stride = 1 + ((N + 31) >> 5);
    int stage_k64 = (H >> 6) < 32 ? (H >> 6) : 32;
    int smem_bytes = ACTREG ? CTA * 4 : stage_k64 * 2304 + CTA * 4;

    if (E < TOPK || N <= 0 || (I & 1023) || (H & 63) || iters <= 0
        || live_expert_target < TOPK || live_expert_target > E
        || (uint64_t)live_expert_target > (uint64_t)N * TOPK) {
        std::fprintf(stderr,
            "requires E>=8, N>0, I%%1024=0, H%%64=0, iters>0, "
            "8<=live_experts<=min(E,N*TOPK)\n");
        return 2;
    }

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    uint64_t H64 = (uint64_t)H >> 6;
    uint64_t W_u32 = (uint64_t)E * H64 * ((uint64_t)I << 4);
    uint64_t S_u32 = (uint64_t)E * H64 * ((uint64_t)I << 2);
    uint64_t X4_u32 = (uint64_t)NP * ((uint64_t)H >> 3);
    uint64_t Sx_u32 = (uint64_t)NP * ((uint64_t)H >> 6);
    std::vector<uint32_t> hXb((uint64_t)E * (uint32_t)Xb_stride, 0u);
    std::vector<uint32_t> expert_assignments(E, 0u);
    std::vector<uint32_t> expert_pool(E);
    std::vector<__nv_bfloat16> hTopk((uint64_t)E * (uint32_t)NP,
                                    __float2bfloat16(0.0f));
    std::vector<__nv_bfloat16> hX((uint64_t)I * (uint32_t)NP);

    for (int e = 0; e < E; e++) expert_pool[e] = (uint32_t)e;
    uint32_t route_state = route_seed;
    for (uint32_t j = (uint32_t)E; j > 1u; --j) {
        route_state = mix32(route_state + j);
        std::swap(expert_pool[j - 1u], expert_pool[route_state % j]);
    }

    uint64_t assignments = 0;
    route_state = mix32(route_state ^ route_seed ^ (uint32_t)N);
    for (int token = 0; token < N; token++) {
        for (int k = 0; k < TOPK; k++) {
            uint32_t e = expert_pool[(route_state + (uint32_t)token * TOPK
                                    + (uint32_t)k)
                                   % (uint32_t)live_expert_target];
            hXb[(uint64_t)e * (uint32_t)Xb_stride
               + 1u + ((uint32_t)token >> 5)]
                |= 1u << ((uint32_t)token & 31u);
            hTopk[(uint64_t)e * (uint32_t)NP + (uint32_t)token]
                = __float2bfloat16(0.125f);
            expert_assignments[e]++;
            assignments++;
        }
    }
    uint32_t live_experts = 0;
    uint32_t min_tokens_per_live_expert = (uint32_t)assignments;
    uint32_t max_tokens_per_live_expert = 0;
    uint64_t active_route_tiles = 0;
    uint64_t active_route_pairs = 0;
    uint64_t active_n8_tiles = 0;
    for (int e = 0; e < E; e++) {
        hXb[(uint64_t)e * (uint32_t)Xb_stride] = expert_assignments[e];
        live_experts += expert_assignments[e] != 0u;
        if (expert_assignments[e]) {
            min_tokens_per_live_expert = std::min(
                min_tokens_per_live_expert, expert_assignments[e]);
            max_tokens_per_live_expert = std::max(
                max_tokens_per_live_expert, expert_assignments[e]);
        }
        uint32_t expert_route_tiles = 0;
        for (int n16 = 0; n16 < (NP >> 4); n16++) {
            uint32_t bits = (hXb[(uint64_t)e * (uint32_t)Xb_stride
                                + 1u + ((uint32_t)n16 >> 1)]
                            >> ((n16 & 1) << 4)) & 0xffffu;
            active_route_tiles += bits != 0u;
            expert_route_tiles += bits != 0u;
            active_n8_tiles += (bits & 0xffu) != 0u;
            active_n8_tiles += (bits & 0xff00u) != 0u;
        }
        active_route_pairs += (expert_route_tiles + 1u) >> 1;
    }
    if (live_experts != (uint32_t)live_expert_target) {
        std::fprintf(stderr, "routing invariant: live=%u target=%d\n",
                     live_experts, live_expert_target);
        return 2;
    }

    uint32_t *dW = nullptr, *dS = nullptr, *dX4 = nullptr;
    uint32_t *dSx = nullptr, *dXb = nullptr;
    __nv_bfloat16 *dWGS = nullptr, *dTopk = nullptr, *dX = nullptr;
    CUDA_CHECK(cudaMalloc(&dW, W_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dS, S_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dWGS, (uint64_t)E * 2u * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dX4, X4_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dSx, Sx_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dXb, hXb.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dTopk, hTopk.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dX, hX.size() * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(dXb, hXb.data(), hXb.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTopk, hTopk.data(), hTopk.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    init_W13<<<4096, 256>>>(dW, W_u32);
    init_u32<<<4096, 256>>>(dS, S_u32, 0x38383838u);
    init_bf16<<<256, 256>>>(reinterpret_cast<uint16_t*>(dWGS),
                            (uint64_t)E * 2u, 0x3f80u);
    init_X4<<<4096, 256>>>(dX4, X4_u32, (uint32_t)NP >> 4);
    init_u32<<<4096, 256>>>(dSx, Sx_u32, 0x38383838u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFuncSetAttribute(
        ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    __nv_bfloat16 XGSINV = __float2bfloat16(0.5f);
    CUDA_CHECK(cudaMemset(dX, 0, hX.size() * sizeof(__nv_bfloat16)));
    ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe<<<E, CTA, smem_bytes>>>(
        dW, dS, dWGS, dX4, dSx, XGSINV, dXb, dTopk, dX,
        Xb_stride, NP, I, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hX.data(), dX, hX.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int ii = 0; ii < I; ii++) {
        int plane = (ii & 1023) >> 8;
        float w1 = 0.5f * float(plane + 1);
        for (int token = 0; token < N; token++) {
            float xv = 0.25f * float(1 + ((token + (token >> 3)) & 3));
            float d1 = float(H) * w1 * xv;
            float d3 = float(H) * xv;
            float expected = (d1 / (1.0f + std::exp(-d1))) * d3;
            float got = __bfloat162float(hX[(uint64_t)ii * (uint32_t)NP + token]);
            double ae = std::fabs((double)got - (double)expected);
            double re = ae / std::fmax(1.0, std::fabs((double)expected));
            max_abs = std::fmax(max_abs, ae);
            max_rel = std::fmax(max_rel, re);
        }
    }
    uint64_t checksum = 0xcbf29ce484222325ull;
    for (const __nv_bfloat16 v : hX) {
        uint16_t raw;
        std::memcpy(&raw, &v, sizeof(raw));
        checksum = (checksum ^ raw) * 0x100000001b3ull;
    }

    CUDA_CHECK(cudaMemset(dX, 0, hX.size() * sizeof(__nv_bfloat16)));
    for (int j = 0; j < 3; j++) {
        ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe<<<E, CTA, smem_bytes>>>(
            dW, dS, dWGS, dX4, dSx, XGSINV, dXb, dTopk, dX,
            Xb_stride, NP, I, H);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(dX, 0, hX.size() * sizeof(__nv_bfloat16)));

    std::vector<float> times(iters);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int j = 0; j < iters; j++) {
        CUDA_CHECK(cudaMemset(dX, 0, hX.size() * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaEventRecord(start));
        ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe<<<E, CTA, smem_bytes>>>(
            dW, dS, dWGS, dX4, dSx, XGSINV, dXb, dTopk, dX,
            Xb_stride, NP, I, H);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[j], start, stop));
    }
    std::sort(times.begin(), times.end());
    double kernel_ms = times[times.size() / 2];
    double flops = 4.0 * (double)N * (double)TOPK * (double)I * (double)H;
    double tflops = flops / (kernel_ms * 1.0e9);
    uint64_t mma_inst = active_route_tiles * (uint64_t)(I >> 10) * 4u
                      * (uint64_t)(H >> 6) * 32u * 2u;
    double mma_flops = (double)mma_inst * (16.0 * 8.0 * 64.0 * 2.0);
    double issued_mma_tflops = mma_flops / (kernel_ms * 1.0e9);
    double mma_ginst_s = (double)mma_inst / (kernel_ms * 1.0e6);
    double mma_useful = mma_flops ? 100.0 * flops / mma_flops : 0.0;
    uint64_t activation_replay = (uint64_t)(I >> 10) * 4u;
    uint64_t X4_bytes = ACTREG
                      ? assignments * activation_replay * H64 * 1024u
                      : active_route_tiles * activation_replay * H64 * 512u;
    uint64_t Sx_bytes = ACTREG
                      ? assignments * activation_replay * H64 * 128u
                      : active_route_tiles * activation_replay * H64 * 64u;
    uint64_t pf_bytes = active_route_tiles * activation_replay * H64 * 576u;
    uint64_t W_bytes = active_route_pairs * activation_replay * H64 * 16384u;
    uint64_t S_bytes = active_route_pairs * activation_replay * H64 * 2048u;
    uint64_t reduce_bytes = active_n8_tiles * (uint64_t)(I >> 10) * 4u * 4096u;
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, ff1_w4a4_swiglu_topk_reduce_mech_actreg_pipe, CTA, smem_bytes));
    int pass = max_rel <= 0.02 && blocks_per_sm == 1;

    FILE* f = std::fopen(csv_path, "w");
    if (!f) {
        std::perror(csv_path);
        return 1;
    }
    std::fprintf(f,
        "gpu,sms,mode,routing,route_seed,repeats,E,N,NP,I,H,TOPK,live_expert_target,live_experts,min_tokens_per_live_expert,max_tokens_per_live_expert,assignments,active_n8_tiles,active_n16_tiles,active_n16_pairs,weight_replay,activation_replay,blocks_per_sm,smem_bytes,x4_global_bytes,sx_global_bytes,pf_requested_bytes,W_bytes,S_bytes,reduce_bytes,mma_inst,kernel_ms,useful_tflops,issued_mma_tflops,mma_ginst_s,mma_useful_pct,max_abs_err,max_rel_err,checksum,status\n");
    std::fprintf(f,
        "%s,%d,no_workspace_n16x2_pf32_actreg_ca_scalar_pipe1,balanced_exact_live_scattered,"
        "0x%08x,%d,%d,%d,%d,%d,%d,8,%d,%u,%u,%u,%llu,%llu,%llu,%llu,"
        "%.6f,%llu,%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%.6f,%.6f,%.6f,%.6f,%.4f,%.9g,%.9g,0x%016llx,%s\n",
        prop.name, prop.multiProcessorCount, route_seed, iters,
        E, N, NP, I, H, live_expert_target, live_experts,
        min_tokens_per_live_expert, max_tokens_per_live_expert,
        (unsigned long long)assignments,
        (unsigned long long)active_n8_tiles,
        (unsigned long long)active_route_tiles,
        (unsigned long long)active_route_pairs,
        live_experts ? (double)active_route_pairs / live_experts : 0.0,
        (unsigned long long)activation_replay, blocks_per_sm, smem_bytes,
        (unsigned long long)X4_bytes, (unsigned long long)Sx_bytes,
        (unsigned long long)pf_bytes,
        (unsigned long long)W_bytes, (unsigned long long)S_bytes,
        (unsigned long long)reduce_bytes, (unsigned long long)mma_inst,
        kernel_ms, tflops, issued_mma_tflops, mma_ginst_s, mma_useful,
        max_abs, max_rel, (unsigned long long)checksum,
        pass ? "PASS" : "FAIL");
    std::fclose(f);

    std::printf("+------+-----+------+----------+------------+------------+------------+---------+\n");
    std::printf("| E    | N   | H    | ms       | useful TF  | W replay   | X replay   | status  |\n");
    std::printf("+------+-----+------+----------+------------+------------+------------+---------+\n");
    std::printf("| %-4d | %-3d | %-4d | %-8.4f | %-10.2f | %-10.2f | %-10llu | %-7s |\n",
                E, N, H, kernel_ms, tflops,
                live_experts ? (double)active_route_pairs / live_experts : 0.0,
                (unsigned long long)activation_replay,
                pass ? "PASS" : "FAIL");
    std::printf("+------+-----+------+----------+------------+------------+------------+---------+\n");
    std::printf("CSV: %s\n", csv_path);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dWGS));
    CUDA_CHECK(cudaFree(dX4));
    CUDA_CHECK(cudaFree(dSx));
    CUDA_CHECK(cudaFree(dXb));
    CUDA_CHECK(cudaFree(dTopk));
    CUDA_CHECK(cudaFree(dX));
    return pass ? 0 : 3;
}
