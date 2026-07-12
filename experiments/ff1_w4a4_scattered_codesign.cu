// xCaliber CTA1024 W4A4 FF1 -> SwiGLU -> topk reduce experiment.
//
// Live schedule mirrored from kernel.cu:
//
//   kt K64 pipeline
//     X4/SX panel[kt] loads once into its monolithic stage
//     i I1024 split: W13/S13 load once, reuse panel[kt]
//       n8 consumer: four I256 MMA planes across all 32 warps
//       BF16 W1/W3 partials -> expert-major global Y workspace
//     final kt: SwiGLU * topk -> BF16 global Y final tail
//
// Global activation rows remain token-major and scattered after routing.
// Shared X4 is fragment-native and each scalar ld.shared maps bank=lane.
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120a -lineinfo -Xptxas=-v \
//     experiments/ff1_w4a4_scattered_codesign.cu -o /tmp/ff1_w4a4_scattered

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                \
    cudaError_t e__ = (call);                                                \
    if (e__ != cudaSuccess) {                                                \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                     cudaGetErrorString(e__));                               \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

#define CTA 1024
#define TOPK_LOCK 8u

enum Mode : int {
    NO_PF_TMA = 0,
    PF32X4_TMA = 1,
    NO_PF_CP_ASYNC_CA = 2,
    PF32X4_CP_ASYNC_CA = 3,
    MODE_COUNT = 4,
};

static const char* mode_name(int mode) {
    switch (mode) {
    case NO_PF_TMA: return "no_pf_tma_smem";
    case PF32X4_TMA: return "pf32x4_n256_tma_smem";
    case NO_PF_CP_ASYNC_CA: return "no_pf_cp_async_ca_smem";
    case PF32X4_CP_ASYNC_CA: return "pf32x4_n256_cp_async_ca_smem";
    default: return "unknown";
    }
}

static const char* transport_name(int mode) {
    return mode < NO_PF_CP_ASYNC_CA ? "tma" : "cp_async_ca";
}

static const char* prefetch_name(int mode) {
    return (mode & 1) ? "4x32B_n256_K64_stripes_no_wait" : "none";
}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

template <int MODE>
__global__ __launch_bounds__(CTA, 1)
void ff1_w4a4_scattered(
    const uint32_t* __restrict__ W13,
    const uint32_t* __restrict__ S13,
    const __nv_bfloat16* __restrict__ W13GS,
    const uint32_t* __restrict__ X4,
    const uint32_t* __restrict__ SX,
    __nv_bfloat16 XGSINV,
    const uint32_t* __restrict__ Xb,
    const __nv_bfloat16* __restrict__ topk_W,
    __nv_bfloat16* __restrict__ Y,
    uint32_t N,
    uint32_t I,
    uint32_t H
) {

    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ __align__(16) uint64_t mbar[2][4];
    __shared__ uint64_t mstate[2][4];
    __shared__ __align__(16) uint64_t ready[2];

    uint32_t rmem[26];
    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(rmem[20])
                 : "l"((uint64_t)__cvta_generic_to_global(
                       Xb + uint64_t(blockIdx.x) *
                       (1u + ((N + 31u) >> 5))))
                 : "memory");
    if (!rmem[20]) return;

    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(rmem[16])
                 : "l"((uint64_t)__cvta_generic_to_global(
                       W13GS + uint64_t(blockIdx.x) * 2u))
                 : "memory");
    rmem[17] = __bfloat16_as_ushort(XGSINV);
    rmem[17] |= rmem[17] << 16;
    asm volatile(
        "{\n\t"
        ".reg .b32 zero;\n\t"
        "mov.b32 zero, 0;\n\t"
        "fma.rn.bf16x2 %0, %0, %1, zero;\n\t"
        "}"
        : "+r"(rmem[16])
        : "r"(rmem[17]));

    if (!threadIdx.x) {
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], 32;\n\t"
            "mbarrier.init.shared::cta.b64 [%1], 32;\n\t"
            "fence.mbarrier_init.release.cluster;"
            :
            : "r"((uint32_t)__cvta_generic_to_shared(ready)),
              "r"((uint32_t)__cvta_generic_to_shared(ready + 1))
            : "memory");
    }
    if constexpr (MODE < NO_PF_CP_ASYNC_CA) {
        if (!(threadIdx.x & 255u)) {
            asm volatile(
                "mbarrier.init.shared::cta.b64 [%0], 1;\n\t"
                "mbarrier.init.shared::cta.b64 [%1], 1;\n\t"
                "fence.mbarrier_init.release.cluster;\n\t"
                "fence.proxy.async.shared::cta;"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(
                      mbar[0] + (threadIdx.x >> 8))),
                  "r"((uint32_t)__cvta_generic_to_shared(
                      mbar[1] + (threadIdx.x >> 8)))
                : "memory");
        }
    }
    __syncthreads();

    // kt is the producer panel. At kt>0, consume panel kt-1 while the
    // asynchronous producer fills panel kt in the opposite monolithic stage.
    for (uint32_t kt = 0; kt <= (H >> 6); ++kt) {
        if (kt) {
            if constexpr (MODE < NO_PF_CP_ASYNC_CA) {
                if (!(threadIdx.x & 255u)) {
                    asm volatile(
                        "{\n\t"
                        ".reg .pred done;\n\t"
                        "wait_%=:\n\t"
                        "mbarrier.test_wait.acquire.cta.shared::cta.b64 "
                        "done, [%0], %1;\n\t"
                        "@!done bra wait_%=;\n\t"
                        "}"
                        :
                        : "r"((uint32_t)__cvta_generic_to_shared(
                              mbar[(kt - 1u) & 1u] + (threadIdx.x >> 8))),
                          "l"(mstate[(kt - 1u) & 1u][threadIdx.x >> 8])
                        : "memory");
                }
            }
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
            if (!(threadIdx.x & 31u)) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
                    :
                    : "r"((uint32_t)__cvta_generic_to_shared(
                          ready + ((kt - 1u) & 1u)))
                    : "memory");
            }
            asm volatile(
                "{\n\t"
                ".reg .pred done;\n\t"
                "ready_%=:\n\t"
                "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 "
                "done, [%0], %1;\n\t"
                "@!done bra ready_%=;\n\t"
                "}"
                :
                : "r"((uint32_t)__cvta_generic_to_shared(
                      ready + ((kt - 1u) & 1u))),
                  "r"(((kt - 1u) >> 1) & 1u)
                : "memory");
        }

        if (kt < (H >> 6)) {
            if constexpr (MODE == PF32X4_TMA ||
                          MODE == PF32X4_CP_ASYNC_CA) {
                for (uint32_t n256 = 0; n256 < ((N + 255u) >> 8); ++n256) {
                    rmem[20] = 0u;
                    if ((n256 << 8) + (threadIdx.x & 255u) < N) {
                        asm volatile("ldu.global.u32 %0, [%1];"
                                     : "=r"(rmem[20])
                                     : "l"((uint64_t)__cvta_generic_to_global(
                                           Xb + uint64_t(blockIdx.x)
                                           * (1u + ((N + 31u) >> 5)) + 1u
                                           + (n256 << 3)
                                           + ((threadIdx.x & 255u) >> 5)))
                                     : "memory");
                        rmem[20] = (rmem[20] >> (threadIdx.x & 31u)) & 1u;
                    }
                    if (!(kt & 3u) && rmem[20]
                        && kt + (threadIdx.x >> 8) < (H >> 6)) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  X4 + (uint64_t((n256 << 8)
                                      + (threadIdx.x & 255u)) * (H >> 6)
                                      + kt + (threadIdx.x >> 8)) * 8u))
                            : "memory");
                    }
                    if (!(kt & 7u) && !(threadIdx.x >> 8) && rmem[20]
                        && kt + 7u < (H >> 6)) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  SX + uint64_t((n256 << 8)
                                      + (threadIdx.x & 255u)) * (H >> 6) + kt))
                            : "memory");
                    }
                }
            }

            if constexpr (MODE < NO_PF_CP_ASYNC_CA) {
                rmem[21] = 0u;
                if (!(threadIdx.x & 224u)) {
                    if ((threadIdx.x & 31u) < 4u
                        && (threadIdx.x >> 8) * 4u + (threadIdx.x & 31u)
                        < ((N + 31u) >> 5)) {
                        asm volatile("ld.global.nc.u32 %0, [%1];"
                                     : "=r"(rmem[21])
                                     : "l"((uint64_t)__cvta_generic_to_global(
                                           Xb + uint64_t(blockIdx.x)
                                           * (1u + ((N + 31u) >> 5)) + 1u
                                           + (threadIdx.x >> 8) * 4u
                                           + (threadIdx.x & 31u)))
                                     : "memory");
                    }
                    rmem[21] = __reduce_add_sync(0xffffffffu, __popc(rmem[21]));
                    if (!(threadIdx.x & 255u)) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
                            "%0, [%1], %2;"
                            : "=l"(mstate[kt & 1u][threadIdx.x >> 8])
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  mbar[kt & 1u] + (threadIdx.x >> 8))),
                              "r"(rmem[21] << 5)
                            : "memory");
                    }
                }
                asm volatile("bar.sync %0, 256;"
                             :
                             : "r"(1u + (threadIdx.x >> 8))
                             : "memory");
            }

            if ((threadIdx.x >> 1) < N) {
                asm volatile("ldu.global.u32 %0, [%1];"
                             : "=r"(rmem[20])
                             : "l"((uint64_t)__cvta_generic_to_global(
                                   Xb + uint64_t(blockIdx.x)
                                   * (1u + ((N + 31u) >> 5)) + 1u
                                   + ((threadIdx.x >> 1) >> 5)))
                             : "memory");
                if ((rmem[20] >> ((threadIdx.x >> 1) & 31u)) & 1u) {
                    if constexpr (MODE < NO_PF_CP_ASYNC_CA) {
                        asm volatile(
                            "cp.async.bulk.shared::cta.global."
                            "mbarrier::complete_tx::bytes "
                            "[%0], [%1], 16, [%2];"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + blockDim.x * 16u + (kt & 1u) * N * 36u
                                  + ((threadIdx.x >> 1) >> 3) * 256u
                                  + (threadIdx.x & 1u) * 128u
                                  + ((threadIdx.x >> 1) & 7u) * 16u)),
                              "l"((uint64_t)__cvta_generic_to_global(
                                  X4 + (uint64_t(threadIdx.x >> 1) * (H >> 6)
                                      + kt) * 8u + (threadIdx.x & 1u) * 4u)),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  mbar[kt & 1u] + (threadIdx.x >> 8)))
                            : "memory");
                    } else {
                        asm volatile(
                            "cp.async.ca.shared.global [%0], [%1], 16;"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + blockDim.x * 16u + (kt & 1u) * N * 36u
                                  + ((threadIdx.x >> 1) >> 3) * 256u
                                  + (threadIdx.x & 1u) * 128u
                                  + ((threadIdx.x >> 1) & 7u) * 16u)),
                              "l"((uint64_t)__cvta_generic_to_global(
                                  X4 + (uint64_t(threadIdx.x >> 1) * (H >> 6)
                                      + kt) * 8u + (threadIdx.x & 1u) * 4u))
                            : "memory");
                    }
                }
            }

            if ((threadIdx.x & 255u) < 128u
                && (threadIdx.x >> 8) * 128u + (threadIdx.x & 255u) < N) {
                asm volatile("ldu.global.u32 %0, [%1];"
                             : "=r"(rmem[20])
                             : "l"((uint64_t)__cvta_generic_to_global(
                                   Xb + uint64_t(blockIdx.x)
                                   * (1u + ((N + 31u) >> 5)) + 1u
                                   + (((threadIdx.x >> 8) * 128u
                                       + (threadIdx.x & 255u)) >> 5)))
                             : "memory");
                if ((rmem[20] >> (((threadIdx.x >> 8) * 128u
                    + (threadIdx.x & 255u)) & 31u)) & 1u) {
                    asm volatile(
                        "cp.async.ca.shared.global [%0], [%1], 4;"
                        :
                        : "r"((uint32_t)__cvta_generic_to_shared(
                              smem + blockDim.x * 16u + (kt & 1u) * N * 36u
                              + N * 32u
                              + (((threadIdx.x >> 8) * 128u
                                  + (threadIdx.x & 255u)) >> 3) * 32u
                              + (((threadIdx.x >> 8) * 128u
                                  + (threadIdx.x & 255u)) & 7u) * 4u)),
                          "l"((uint64_t)__cvta_generic_to_global(
                              SX + uint64_t((threadIdx.x >> 8) * 128u
                                  + (threadIdx.x & 255u)) * (H >> 6) + kt))
                        : "memory");
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }

        if (!kt) continue;

        for (uint32_t i = 0; i < (I >> 10); ++i) {
            // S is dead before W arrives: one scratch register stages all four
            // plane-major packets, then becomes W packet 0.
            rmem[0] = 0u;
            if ((threadIdx.x & 3u) < 2u)
                asm volatile("ld.global.nc.b32 %0, [%1];"
                             : "=r"(rmem[0])
                             : "l"((uint64_t)__cvta_generic_to_global(
                                   S13 + uint64_t(blockIdx.x) * (H >> 6) * (I << 2)
                                   + uint64_t(kt - 1u) * (I << 2) + (uint64_t(i) << 12)
                                   + (threadIdx.x >> 2)
                                   + ((threadIdx.x & 3u) << 8))) : "memory");
            asm volatile("st.shared.u32 [%0], %1;" :
                         : "r"((uint32_t)__cvta_generic_to_shared(
                               smem + threadIdx.x * 4u)),
                           "r"(rmem[0]) : "memory");

            rmem[0] = 0u;
            if ((threadIdx.x & 3u) < 2u)
                asm volatile("ld.global.nc.b32 %0, [%1];"
                             : "=r"(rmem[0])
                             : "l"((uint64_t)__cvta_generic_to_global(
                                   S13 + uint64_t(blockIdx.x) * (H >> 6) * (I << 2)
                                   + uint64_t(kt - 1u) * (I << 2) + (uint64_t(i) << 12)
                                   + blockDim.x + (threadIdx.x >> 2)
                                   + ((threadIdx.x & 3u) << 8))) : "memory");
            asm volatile("st.shared.u32 [%0], %1;" :
                         : "r"((uint32_t)__cvta_generic_to_shared(
                               smem + blockDim.x * 4u
                               + threadIdx.x * 4u)),
                           "r"(rmem[0]) : "memory");

            rmem[0] = 0u;
            if ((threadIdx.x & 3u) < 2u)
                asm volatile("ld.global.nc.b32 %0, [%1];"
                             : "=r"(rmem[0])
                             : "l"((uint64_t)__cvta_generic_to_global(
                                   S13 + uint64_t(blockIdx.x) * (H >> 6) * (I << 2)
                                   + uint64_t(kt - 1u) * (I << 2) + (uint64_t(i) << 12)
                                   + blockDim.x * 2u + (threadIdx.x >> 2)
                                   + ((threadIdx.x & 3u) << 8))) : "memory");
            asm volatile("st.shared.u32 [%0], %1;" :
                         : "r"((uint32_t)__cvta_generic_to_shared(
                               smem + blockDim.x * 8u
                               + threadIdx.x * 4u)),
                           "r"(rmem[0]) : "memory");

            rmem[0] = 0u;
            if ((threadIdx.x & 3u) < 2u)
                asm volatile("ld.global.nc.b32 %0, [%1];"
                             : "=r"(rmem[0])
                             : "l"((uint64_t)__cvta_generic_to_global(
                                   S13 + uint64_t(blockIdx.x) * (H >> 6) * (I << 2)
                                   + uint64_t(kt - 1u) * (I << 2) + (uint64_t(i) << 12)
                                   + blockDim.x * 3u + (threadIdx.x >> 2)
                                   + ((threadIdx.x & 3u) << 8))) : "memory");
            asm volatile("st.shared.u32 [%0], %1;" :
                         : "r"((uint32_t)__cvta_generic_to_shared(
                               smem + blockDim.x * 12u
                               + threadIdx.x * 4u)),
                           "r"(rmem[0]) : "memory");

            // One K64 / I1024 bundle. Only 16 W packets stay live across n8.
            asm volatile(
                "{\n\t"
                ".reg .u32 q, cta;\n\t"
                ".reg .u64 addr;\n\t"
                "mov.u32 cta, %%ctaid.x;\n\t"
                "shr.u32 q, %17, 6;\n\t"
                "mul.lo.u32 q, q, %18;\n\t"
                "shl.b32 q, q, 6;\n\t"
                "mad.wide.u32 addr, cta, q, %16;\n\t"
                "shl.b32 q, %18, 6;\n\t"
                "mad.wide.u32 addr, %19, q, addr;\n\t"
                "mad.wide.u32 addr, %20, 65536, addr;\n\t"
                "mov.u32 q, %%tid.x;\n\t"
                "mad.wide.u32 addr, q, 16, addr;\n\t"
                "ld.global.nc.L2::64B.v4.u32 {%0,%1,%2,%3}, [addr];\n\t"
                "ld.global.nc.v4.u32 {%4,%5,%6,%7}, [addr + 16384];\n\t"
                "ld.global.nc.v4.u32 {%8,%9,%10,%11}, [addr + 32768];\n\t"
                "ld.global.nc.v4.u32 {%12,%13,%14,%15}, [addr + 49152];\n\t"
                "}"
                : "=r"(rmem[0]), "=r"(rmem[1]), "=r"(rmem[2]), "=r"(rmem[3]),
                  "=r"(rmem[4]), "=r"(rmem[5]), "=r"(rmem[6]), "=r"(rmem[7]),
                  "=r"(rmem[8]), "=r"(rmem[9]), "=r"(rmem[10]), "=r"(rmem[11]),
                  "=r"(rmem[12]), "=r"(rmem[13]), "=r"(rmem[14]), "=r"(rmem[15])
                : "l"((uint64_t)__cvta_generic_to_global(W13)),
                  "r"(H), "r"(I), "r"(kt - 1u), "r"(i)
                : "memory");
            rmem[0] ^= 0x88888888u; rmem[1] ^= 0x88888888u;
            rmem[2] ^= 0x88888888u; rmem[3] ^= 0x88888888u;
            rmem[4] ^= 0x88888888u; rmem[5] ^= 0x88888888u;
            rmem[6] ^= 0x88888888u; rmem[7] ^= 0x88888888u;
            rmem[8] ^= 0x88888888u; rmem[9] ^= 0x88888888u;
            rmem[10] ^= 0x88888888u; rmem[11] ^= 0x88888888u;
            rmem[12] ^= 0x88888888u; rmem[13] ^= 0x88888888u;
            rmem[14] ^= 0x88888888u; rmem[15] ^= 0x88888888u;

            for (uint32_t n8 = 0; (n8 << 3) < N; ++n8) {
                        asm volatile("ldu.global.u32 %0, [%1];"
                                     : "=r"(rmem[25])
                                     : "l"((uint64_t)__cvta_generic_to_global(
                                           Xb + uint64_t(blockIdx.x)
                                           * (1u + ((N + 31u) >> 5)) + 1u
                                           + (n8 >> 2)))
                                     : "memory");
                        if (!((rmem[25] >> ((n8 & 3u) << 3)) & 0xffu)) continue;
                        rmem[20] = rmem[21] = rmem[22] = 0u;
                        asm volatile("ld.shared.u32 %0, [%1];"
                                     : "=r"(rmem[20])
                                     : "r"((uint32_t)__cvta_generic_to_shared(
                                           smem + blockDim.x * 16u
                                           + ((kt - 1u) & 1u) * N * 36u
                                           + n8 * 256u
                                           + (threadIdx.x & 31u) * 4u))
                                     : "memory");
                        asm volatile("ld.shared.u32 %0, [%1];"
                                     : "=r"(rmem[21])
                                     : "r"((uint32_t)__cvta_generic_to_shared(
                                           smem + blockDim.x * 16u
                                           + ((kt - 1u) & 1u) * N * 36u
                                           + n8 * 256u + 128u
                                           + (threadIdx.x & 31u) * 4u))
                                     : "memory");
                        if (!(threadIdx.x & 3u)) {
                            asm volatile("ld.shared.u32 %0, [%1];"
                                         : "=r"(rmem[22])
                                         : "r"((uint32_t)__cvta_generic_to_shared(
                                               smem + blockDim.x * 16u
                                               + ((kt - 1u) & 1u) * N * 36u
                                               + N * 32u
                                               + n8 * 32u
                                               + ((threadIdx.x & 31u) >> 2) * 4u))
                                         : "memory");
                        }
                        if (!((((rmem[25]
                              >> ((n8 & 3u) << 3)) & 0xffu)
                              >> ((threadIdx.x & 31u) >> 2)) & 1u)) {
                            rmem[20] = rmem[21] = rmem[22] = 0u;
                        }

                        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

                        if (kt > 1u) {
                            asm volatile(
                                "ld.global.nc.u32 %0, [%2];\n\t"
                                "ld.global.nc.u32 %1, [%3];"
                                : "=r"(rmem[23]), "=r"(rmem[24])
                                : "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1)
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1))),
                                  "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + I
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1)))
                                : "memory");
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            d2 = __uint_as_float((rmem[24] & 0xffffu) << 16);
                            d3 = __uint_as_float(rmem[24] & 0xffff0000u);
                        }
                        asm volatile("ld.shared.u32 %0, [%1];"
                                     : "=r"(rmem[25])
                                     : "r"((uint32_t)__cvta_generic_to_shared(
                                           smem + threadIdx.x * 4u)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 sel;\n\t"
                            "mov.b16 sel, 0;\n\t"
                            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4."
                            "block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3}, %10, {sel,sel}, %11, {sel,sel};\n\t"
                            "}"
                            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                            : "r"(rmem[0]), "r"(rmem[1]),
                              "r"(rmem[2]), "r"(rmem[3]),
                              "r"(rmem[20]), "r"(rmem[21]),
                              "r"(rmem[25]), "r"(rmem[22]));
                        if (kt < (H >> 6)) {
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1)
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[23]) : "memory");
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + I
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[24]) : "memory");
                        } else {
                            asm volatile("ld.global.nc.u32 %0, [%1];"
                                         : "=r"(rmem[25])
                                         : "l"((uint64_t)__cvta_generic_to_global(
                                             topk_W + uint64_t(blockIdx.x) * N
                                             + (n8 << 3)
                                             + ((threadIdx.x & 3u) << 1)))
                                         : "memory");
                            d0 = fmaf(d0, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d1 = fmaf(d1, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d2 = fmaf(d2, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            d3 = fmaf(d3, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            float ex0, ex1;
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(d0 * 1.4426950408889634f));
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(d1 * 1.4426950408889634f));
                            ex0 += 1.0f;
                            ex1 += 1.0f;
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(ex0));
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(ex1));
                            ex0 *= __uint_as_float((rmem[25] & 0xffffu) << 16);
                            ex1 *= __uint_as_float(rmem[25] & 0xffff0000u);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[25]) : "f"(ex1), "f"(ex0));
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 sw, zero;\n\t"
                                "mov.b32 zero, 0;\n\t"
                                "mul.rn.bf16x2 sw, %1, %2;\n\t"
                                "fma.rn.bf16x2 %0, sw, %0, zero;\n\t"
                                "}"
                                : "+r"(rmem[25])
                                : "r"(rmem[23]), "r"(rmem[24]));
                            asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(gridDim.x) * (I << 1)
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[25]) : "memory");
                        }

                        d0 = d1 = d2 = d3 = 0.0f;
                        if (kt > 1u) {
                            asm volatile(
                                "ld.global.nc.u32 %0, [%2];\n\t"
                                "ld.global.nc.u32 %1, [%3];"
                                : "=r"(rmem[23]), "=r"(rmem[24])
                                : "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + 256u
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1))),
                                  "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + I + 256u
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1)))
                                : "memory");
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            d2 = __uint_as_float((rmem[24] & 0xffffu) << 16);
                            d3 = __uint_as_float(rmem[24] & 0xffff0000u);
                        }
                        asm volatile("ld.shared.u32 %0, [%1];"
                                     : "=r"(rmem[25])
                                     : "r"((uint32_t)__cvta_generic_to_shared(
                                           smem + blockDim.x * 4u
                                           + threadIdx.x * 4u)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 sel;\n\t"
                            "mov.b16 sel, 0;\n\t"
                            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4."
                            "block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3}, %10, {sel,sel}, %11, {sel,sel};\n\t"
                            "}"
                            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                            : "r"(rmem[4]), "r"(rmem[5]),
                              "r"(rmem[6]), "r"(rmem[7]),
                              "r"(rmem[20]), "r"(rmem[21]),
                              "r"(rmem[25]), "r"(rmem[22]));
                        if (kt < (H >> 6)) {
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + 256u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[23]) : "memory");
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + I + 256u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[24]) : "memory");
                        } else {
                            asm volatile("ld.global.nc.u32 %0, [%1];"
                                         : "=r"(rmem[25])
                                         : "l"((uint64_t)__cvta_generic_to_global(
                                             topk_W + uint64_t(blockIdx.x) * N
                                             + (n8 << 3)
                                             + ((threadIdx.x & 3u) << 1)))
                                         : "memory");
                            d0 = fmaf(d0, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d1 = fmaf(d1, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d2 = fmaf(d2, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            d3 = fmaf(d3, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            float ex0, ex1;
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(d0 * 1.4426950408889634f));
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(d1 * 1.4426950408889634f));
                            ex0 += 1.0f;
                            ex1 += 1.0f;
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(ex0));
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(ex1));
                            ex0 *= __uint_as_float((rmem[25] & 0xffffu) << 16);
                            ex1 *= __uint_as_float(rmem[25] & 0xffff0000u);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[25]) : "f"(ex1), "f"(ex0));
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 sw, zero;\n\t"
                                "mov.b32 zero, 0;\n\t"
                                "mul.rn.bf16x2 sw, %1, %2;\n\t"
                                "fma.rn.bf16x2 %0, sw, %0, zero;\n\t"
                                "}"
                                : "+r"(rmem[25])
                                : "r"(rmem[23]), "r"(rmem[24]));
                            asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(gridDim.x) * (I << 1) + 256u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[25]) : "memory");
                        }

                        d0 = d1 = d2 = d3 = 0.0f;
                        if (kt > 1u) {
                            asm volatile(
                                "ld.global.nc.u32 %0, [%2];\n\t"
                                "ld.global.nc.u32 %1, [%3];"
                                : "=r"(rmem[23]), "=r"(rmem[24])
                                : "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + 512u
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1))),
                                  "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + I + 512u
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1)))
                                : "memory");
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            d2 = __uint_as_float((rmem[24] & 0xffffu) << 16);
                            d3 = __uint_as_float(rmem[24] & 0xffff0000u);
                        }
                        asm volatile("ld.shared.u32 %0, [%1];"
                                     : "=r"(rmem[25])
                                     : "r"((uint32_t)__cvta_generic_to_shared(
                                           smem + blockDim.x * 8u
                                           + threadIdx.x * 4u)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 sel;\n\t"
                            "mov.b16 sel, 0;\n\t"
                            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4."
                            "block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3}, %10, {sel,sel}, %11, {sel,sel};\n\t"
                            "}"
                            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                            : "r"(rmem[8]), "r"(rmem[9]),
                              "r"(rmem[10]), "r"(rmem[11]),
                              "r"(rmem[20]), "r"(rmem[21]),
                              "r"(rmem[25]), "r"(rmem[22]));
                        if (kt < (H >> 6)) {
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + 512u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[23]) : "memory");
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + I + 512u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[24]) : "memory");
                        } else {
                            asm volatile("ld.global.nc.u32 %0, [%1];"
                                         : "=r"(rmem[25])
                                         : "l"((uint64_t)__cvta_generic_to_global(
                                             topk_W + uint64_t(blockIdx.x) * N
                                             + (n8 << 3)
                                             + ((threadIdx.x & 3u) << 1)))
                                         : "memory");
                            d0 = fmaf(d0, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d1 = fmaf(d1, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d2 = fmaf(d2, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            d3 = fmaf(d3, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            float ex0, ex1;
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(d0 * 1.4426950408889634f));
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(d1 * 1.4426950408889634f));
                            ex0 += 1.0f;
                            ex1 += 1.0f;
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(ex0));
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(ex1));
                            ex0 *= __uint_as_float((rmem[25] & 0xffffu) << 16);
                            ex1 *= __uint_as_float(rmem[25] & 0xffff0000u);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[25]) : "f"(ex1), "f"(ex0));
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 sw, zero;\n\t"
                                "mov.b32 zero, 0;\n\t"
                                "mul.rn.bf16x2 sw, %1, %2;\n\t"
                                "fma.rn.bf16x2 %0, sw, %0, zero;\n\t"
                                "}"
                                : "+r"(rmem[25])
                                : "r"(rmem[23]), "r"(rmem[24]));
                            asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(gridDim.x) * (I << 1) + 512u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[25]) : "memory");
                        }

                        d0 = d1 = d2 = d3 = 0.0f;
                        if (kt > 1u) {
                            asm volatile(
                                "ld.global.nc.u32 %0, [%2];\n\t"
                                "ld.global.nc.u32 %1, [%3];"
                                : "=r"(rmem[23]), "=r"(rmem[24])
                                : "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + 768u
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1))),
                                  "l"((uint64_t)__cvta_generic_to_global(Y
                                    + (uint64_t(blockIdx.x) * (I << 1) + I + 768u
                                      + (i << 10) + (threadIdx.x >> 2)) * N
                                    + (n8 << 3)
                                    + ((threadIdx.x & 3u) << 1)))
                                : "memory");
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            d2 = __uint_as_float((rmem[24] & 0xffffu) << 16);
                            d3 = __uint_as_float(rmem[24] & 0xffff0000u);
                        }
                        asm volatile("ld.shared.u32 %0, [%1];"
                                     : "=r"(rmem[25])
                                     : "r"((uint32_t)__cvta_generic_to_shared(
                                           smem + blockDim.x * 12u
                                           + threadIdx.x * 4u)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 sel;\n\t"
                            "mov.b16 sel, 0;\n\t"
                            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4."
                            "block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                            "{%0,%1,%2,%3}, %10, {sel,sel}, %11, {sel,sel};\n\t"
                            "}"
                            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                            : "r"(rmem[12]), "r"(rmem[13]),
                              "r"(rmem[14]), "r"(rmem[15]),
                              "r"(rmem[20]), "r"(rmem[21]),
                              "r"(rmem[25]), "r"(rmem[22]));
                        if (kt < (H >> 6)) {
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + 768u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[23]) : "memory");
                            asm volatile("st.global.wb.u32 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(blockIdx.x) * (I << 1) + I + 768u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[24]) : "memory");
                        } else {
                            asm volatile("ld.global.nc.u32 %0, [%1];"
                                         : "=r"(rmem[25])
                                         : "l"((uint64_t)__cvta_generic_to_global(
                                             topk_W + uint64_t(blockIdx.x) * N
                                             + (n8 << 3)
                                             + ((threadIdx.x & 3u) << 1)))
                                         : "memory");
                            d0 = fmaf(d0, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d1 = fmaf(d1, __uint_as_float(
                                (rmem[16] & 0xffffu) << 16), 0.0f);
                            d2 = fmaf(d2, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            d3 = fmaf(d3, __uint_as_float(
                                rmem[16] & 0xffff0000u), 0.0f);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[23]) : "f"(d1), "f"(d0));
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[24]) : "f"(d3), "f"(d2));
                            d0 = __uint_as_float((rmem[23] & 0xffffu) << 16);
                            d1 = __uint_as_float(rmem[23] & 0xffff0000u);
                            float ex0, ex1;
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(d0 * 1.4426950408889634f));
                            asm volatile("ex2.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(d1 * 1.4426950408889634f));
                            ex0 += 1.0f;
                            ex1 += 1.0f;
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex0) : "f"(ex0));
                            asm volatile("rcp.approx.ftz.f32 %0, %1;"
                                         : "=f"(ex1) : "f"(ex1));
                            ex0 *= __uint_as_float((rmem[25] & 0xffffu) << 16);
                            ex1 *= __uint_as_float(rmem[25] & 0xffff0000u);
                            asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                         : "=r"(rmem[25]) : "f"(ex1), "f"(ex0));
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 sw, zero;\n\t"
                                "mov.b32 zero, 0;\n\t"
                                "mul.rn.bf16x2 sw, %1, %2;\n\t"
                                "fma.rn.bf16x2 %0, sw, %0, zero;\n\t"
                                "}"
                                : "+r"(rmem[25])
                                : "r"(rmem[23]), "r"(rmem[24]));
                            asm volatile("red.global.add.noftz.bf16x2 [%0], %1;"
                                         :
                                         : "l"((uint64_t)__cvta_generic_to_global(Y
                                           + (uint64_t(gridDim.x) * (I << 1) + 768u
                                             + (i << 10) + (threadIdx.x >> 2)) * N
                                           + (n8 << 3)
                                           + ((threadIdx.x & 3u) << 1))),
                                           "r"(rmem[25]) : "memory");
                        }
                    }
        }
    }

}

__global__ void init_W13(uint32_t* p, uint64_t n) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t plane = uint32_t(x >> 12) & 3u;
        uint32_t q = (x & 1u) ? 2u : plane + 1u;
        p[x] = q * 0x11111111u;
    }
}

__global__ void init_u32(uint32_t* p, uint64_t n, uint32_t value) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) p[x] = value;
}

__global__ void init_X4(uint32_t* p, uint64_t n, uint32_t K64) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t token = uint32_t(x / (uint64_t(K64) * 8u));
        uint32_t q = 1u + ((token + (token >> 3)) & 3u);
        p[x] = q * 0x11111111u;
    }
}

__global__ void init_bf16(uint16_t* p, uint64_t n, uint16_t value) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) p[x] = value;
}

struct Config {
    uint32_t E = 384;
    uint32_t N = 256;
    uint32_t I = 2048;
    uint32_t H = 7168;
    uint32_t active_experts = 384;
    uint32_t route_seed = 0x6a09e667u;
    uint32_t repeats = 20;
    std::string methods = "all";
    const char* csv_path = "ff1_w4a4_scattered.csv";
};

static uint32_t u32(const char* s) {
    char* end = nullptr;
    const unsigned long x = std::strtoul(s, &end, 0);
    if (!end || *end) {
        std::fprintf(stderr, "bad integer: %s\n", s);
        std::exit(2);
    }
    return uint32_t(x);
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int a = 1; a < argc; ++a) {
        auto need = [&](const char* flag) {
            if (++a == argc) {
                std::fprintf(stderr, "%s needs a value\n", flag);
                std::exit(2);
            }
            return argv[a];
        };
        if (!std::strcmp(argv[a], "--E")) c.E = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--N")) c.N = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--I")) c.I = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--H")) c.H = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--active-experts")) c.active_experts = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--route-seed")) c.route_seed = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--repeats")) c.repeats = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--methods")) c.methods = need(argv[a]);
        else if (!std::strcmp(argv[a], "--csv")) c.csv_path = need(argv[a]);
        else {
            std::fprintf(stderr, "unknown flag: %s\n", argv[a]);
            std::exit(2);
        }
    }
    return c;
}

static bool selected(const std::string& methods, const char* name) {
    if (methods == "all") return true;
    const std::string list = "," + methods + ",";
    return list.find("," + std::string(name) + ",") != std::string::npos;
}

template <int MODE>
static void launch(
    uint32_t E,
    const uint32_t* W13,
    const uint32_t* S13,
    const __nv_bfloat16* W13GS,
    const uint32_t* X4,
    const uint32_t* SX,
    __nv_bfloat16 XGSINV,
    const uint32_t* Xb,
    const __nv_bfloat16* topk_W,
    __nv_bfloat16* Y,
    uint32_t N,
    uint32_t I,
    uint32_t H
) {
    ff1_w4a4_scattered<MODE><<<E, CTA, N * 72u + CTA * 16u>>>(
        W13, S13, W13GS, X4, SX, XGSINV, Xb, topk_W, Y, N, I, H);
}

static void launch_mode(
    int mode,
    uint32_t E,
    const uint32_t* W13,
    const uint32_t* S13,
    const __nv_bfloat16* W13GS,
    const uint32_t* X4,
    const uint32_t* SX,
    __nv_bfloat16 XGSINV,
    const uint32_t* Xb,
    const __nv_bfloat16* topk_W,
    __nv_bfloat16* Y,
    uint32_t N,
    uint32_t I,
    uint32_t H
) {
    switch (mode) {
    case NO_PF_TMA: launch<NO_PF_TMA>(E,W13,S13,W13GS,X4,SX,XGSINV,Xb,topk_W,Y,N,I,H); break;
    case PF32X4_TMA: launch<PF32X4_TMA>(E,W13,S13,W13GS,X4,SX,XGSINV,Xb,topk_W,Y,N,I,H); break;
    case NO_PF_CP_ASYNC_CA: launch<NO_PF_CP_ASYNC_CA>(E,W13,S13,W13GS,X4,SX,XGSINV,Xb,topk_W,Y,N,I,H); break;
    case PF32X4_CP_ASYNC_CA: launch<PF32X4_CP_ASYNC_CA>(E,W13,S13,W13GS,X4,SX,XGSINV,Xb,topk_W,Y,N,I,H); break;
    }
}

static uint64_t checksum(const std::vector<__nv_bfloat16>& y) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const __nv_bfloat16 v : y) {
        uint16_t x;
        std::memcpy(&x, &v, sizeof(x));
        h = (h ^ x) * 0x100000001b3ull;
    }
    return h;
}

int main(int argc, char** argv) {
    const Config c = parse_args(argc, argv);
    if (!c.E || !c.N || c.N > 512u || (c.N & 7u) || !c.I || (c.I & 1023u) ||
        !c.H || (c.H & 63u) || !c.repeats ||
        c.active_experts < TOPK_LOCK || c.active_experts > c.E ||
        uint64_t(c.active_experts) > uint64_t(c.N) * TOPK_LOCK) {
        std::fprintf(stderr,
            "require E>0; 8<=N<=512 divisible by 8; I divisible by 1024; "
            "H divisible by 64; repeats>0; "
            "8<=active-experts<=min(E,N*TOPK)\n");
        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const uint32_t K64 = c.H >> 6;
    uint32_t smem_bytes = c.N * 72u + CTA * 16u;
    const uint32_t Xb_stride = 1u + ((c.N + 31u) >> 5);
    const uint64_t W_u32 = uint64_t(c.E) * K64 * (uint64_t(c.I) << 4);
    const uint64_t S_u32 = uint64_t(c.E) * K64 * (uint64_t(c.I) << 2);
    const uint64_t X4_u32 = uint64_t(c.N) * K64 * 8u;
    const uint64_t SX_u32 = uint64_t(c.N) * K64;
    const uint64_t partial_u16 = uint64_t(c.E) * (uint64_t(c.I) << 1) * c.N;
    const uint64_t final_u16 = uint64_t(c.I) * c.N;

    std::vector<uint32_t> h_Xb(uint64_t(c.E) * Xb_stride, 0u);
    std::vector<uint32_t> expert_assignments(c.E, 0u);
    std::vector<uint32_t> expert_pool(c.E);
    std::vector<__nv_bfloat16> h_topk(uint64_t(c.E) * c.N,
                                      __float2bfloat16(0.0f));
    for (uint32_t e = 0; e < c.E; ++e) expert_pool[e] = e;
    uint32_t route_state = c.route_seed;
    for (uint32_t j = c.E; j > 1u; --j) {
        route_state = mix32(route_state + j);
        std::swap(expert_pool[j - 1u], expert_pool[route_state % j]);
    }

    uint64_t assignments = 0;
    route_state = mix32(route_state ^ c.route_seed ^ c.N);
    for (uint32_t token = 0; token < c.N; ++token) {
        for (uint32_t slot = 0; slot < TOPK_LOCK; ++slot) {
            uint32_t e = expert_pool[(route_state + token * TOPK_LOCK + slot)
                                   % c.active_experts];
            h_Xb[uint64_t(e) * Xb_stride + 1u + (token >> 5)]
                |= 1u << (token & 31u);
            h_topk[uint64_t(e) * c.N + token] = __float2bfloat16(0.125f);
            ++expert_assignments[e];
            ++assignments;
        }
    }

    uint32_t live_experts = 0;
    uint32_t min_tokens_per_live_expert = c.N * TOPK_LOCK;
    uint32_t max_tokens_per_live_expert = 0;
    uint64_t active_n8_tiles = 0;
    for (uint32_t e = 0; e < c.E; ++e) {
        h_Xb[uint64_t(e) * Xb_stride] = expert_assignments[e];
        live_experts += expert_assignments[e] != 0u;
        if (expert_assignments[e]) {
            min_tokens_per_live_expert = std::min(
                min_tokens_per_live_expert, expert_assignments[e]);
            max_tokens_per_live_expert = std::max(
                max_tokens_per_live_expert, expert_assignments[e]);
        }
        for (uint32_t n8 = 0; n8 < (c.N >> 3); ++n8) {
            active_n8_tiles += ((h_Xb[uint64_t(e) * Xb_stride + 1u + (n8 >> 2)]
                              >> ((n8 & 3u) << 3)) & 0xffu) != 0u;
        }
    }
    if (live_experts != c.active_experts) {
        std::fprintf(stderr, "routing invariant: live=%u target=%u\n",
                     live_experts, c.active_experts);
        return 2;
    }

    uint32_t *W13 = nullptr, *S13 = nullptr, *X4 = nullptr, *SX = nullptr;
    uint32_t* Xb = nullptr;
    __nv_bfloat16 *W13GS = nullptr, *topk_W = nullptr, *Y = nullptr;
    CUDA_CHECK(cudaMalloc(&W13, W_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&S13, S_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&W13GS, uint64_t(c.E) * 2u * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&X4, X4_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&SX, SX_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&Xb, h_Xb.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&topk_W, h_topk.size() * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Y, (partial_u16 + final_u16) * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(Xb, h_Xb.data(), h_Xb.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(topk_W, h_topk.data(),
                          h_topk.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));

    init_W13<<<4096, 256>>>(W13, W_u32);
    init_u32<<<4096, 256>>>(S13, S_u32, 0x38383838u);
    init_bf16<<<256, 256>>>(reinterpret_cast<uint16_t*>(W13GS),
                            uint64_t(c.E) * 2u, 0x3f80u);
    init_X4<<<4096, 256>>>(X4, X4_u32, K64);
    init_u32<<<4096, 256>>>(SX, SX_u32, 0x38383838u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFuncSetAttribute(ff1_w4a4_scattered<NO_PF_TMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(ff1_w4a4_scattered<PF32X4_TMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(ff1_w4a4_scattered<NO_PF_CP_ASYNC_CA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(ff1_w4a4_scattered<PF32X4_CP_ASYNC_CA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    int blocks_per_sm[MODE_COUNT]{};
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm[0], ff1_w4a4_scattered<NO_PF_TMA>, CTA, smem_bytes));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm[1], ff1_w4a4_scattered<PF32X4_TMA>, CTA, smem_bytes));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm[2], ff1_w4a4_scattered<NO_PF_CP_ASYNC_CA>, CTA, smem_bytes));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm[3], ff1_w4a4_scattered<PF32X4_CP_ASYNC_CA>, CTA, smem_bytes));

    FILE* csv = std::fopen(c.csv_path, "w");
    if (!csv) {
        std::perror(c.csv_path);
        return 1;
    }
    std::fprintf(csv,
        "gpu,sms,mode,transport,prefetch,routing,route_seed,repeats,E,N,I,H,TOPK,live_expert_target,live_experts,min_tokens_per_live_expert,max_tokens_per_live_expert,assignments,active_n8_tiles,blocks_per_sm,smem_bytes,n256_count,x4_global_bytes,sx_global_bytes,pf_requested_bytes,W_bytes,S_bytes,partial_load_bytes,partial_store_bytes,final_reduce_bytes,Y_workspace_bytes,mma_inst,kernel_ms,useful_tflops,issued_mma_tflops,mma_useful_pct,max_abs_err,max_rel_err,checksum,status\n");

    const __nv_bfloat16 XGSINV = __float2bfloat16(0.5f);
    std::vector<__nv_bfloat16> h_final(final_u16);
    const uint64_t i_count = c.I >> 10;
    const uint64_t x4_bytes = assignments * K64 * 32u;
    const uint64_t sx_bytes = assignments * K64 * 4u;
    const uint64_t W_bytes = uint64_t(live_experts) * K64 * i_count * 65536u;
    const uint64_t S_bytes = uint64_t(live_experts) * K64 * i_count * 8192u;
    const uint64_t partial_load_bytes =
        active_n8_tiles * (K64 - 1u) * i_count * 32768u;
    const uint64_t partial_store_bytes = partial_load_bytes;
    const uint64_t final_reduce_bytes = active_n8_tiles * uint64_t(c.I) * 16u;
    const uint64_t workspace_bytes = (partial_u16 + final_u16) * sizeof(__nv_bfloat16);
    const uint64_t mma_inst = active_n8_tiles * i_count * K64 * 4u * 32u;
    const double useful_flops = 4.0 * double(assignments) * c.I * c.H;
    const double issued_flops = double(mma_inst) * (16.0 * 8.0 * 64.0 * 2.0);

    std::printf("gpu,%s\nsms,%d\nrouting,balanced_exact_live_scattered\n",
                prop.name, prop.multiProcessorCount);
    std::printf("E,%u\nN,%u\nI,%u\nH,%u\nTOPK,8\nlive_expert_target,%u\n",
                c.E, c.N, c.I, c.H, c.active_experts);

    for (int mode = 0; mode < MODE_COUNT; ++mode) {
        if (!selected(c.methods, mode_name(mode))) continue;

        CUDA_CHECK(cudaMemset(Y + partial_u16, 0,
                              final_u16 * sizeof(__nv_bfloat16)));
        launch_mode(mode, c.E, W13, S13, W13GS, X4, SX, XGSINV,
                    Xb, topk_W, Y, c.N, c.I, c.H);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_final.data(), Y + partial_u16,
                              final_u16 * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));

        double max_abs = 0.0, max_rel = 0.0;
        for (uint32_t ii = 0; ii < c.I; ++ii) {
            const float w1 = 0.5f * float(((ii & 1023u) >> 8) + 1u);
            for (uint32_t token = 0; token < c.N; ++token) {
                const float x4 = 0.5f *
                    float(1u + ((token + (token >> 3)) & 3u));
                float a = 0.0f, b = 0.0f;
                for (uint32_t kt = 0; kt < K64; ++kt) {
                    a = __bfloat162float(__float2bfloat16(
                        a + 64.0f * w1 * x4));
                    b = __bfloat162float(__float2bfloat16(
                        b + 64.0f * x4));
                }
                a = __bfloat162float(__float2bfloat16(a * 0.5f));
                b = __bfloat162float(__float2bfloat16(b * 0.5f));
                __nv_bfloat16 contribution = __float2bfloat16(
                    __bfloat162float(__float2bfloat16(a * b))
                    * __bfloat162float(__float2bfloat16(
                        0.125f / (1.0f + std::exp(-a)))));
                __nv_bfloat16 reduced = __float2bfloat16(0.0f);
                for (uint32_t slot = 0; slot < TOPK_LOCK; ++slot) {
                    reduced = __float2bfloat16(
                        __bfloat162float(reduced)
                        + __bfloat162float(contribution));
                }
                const float expected = __bfloat162float(reduced);
                const float got = __bfloat162float(h_final[uint64_t(ii) * c.N + token]);
                const double ae = std::fabs(double(got) - double(expected));
                const double re = ae / std::fmax(1.0, std::fabs(double(expected)));
                max_abs = std::fmax(max_abs, ae);
                max_rel = std::fmax(max_rel, re);
            }
        }
        const uint64_t sum = checksum(h_final);

        for (int warm = 0; warm < 2; ++warm) {
            CUDA_CHECK(cudaMemset(Y + partial_u16, 0,
                                  final_u16 * sizeof(__nv_bfloat16)));
            launch_mode(mode, c.E, W13, S13, W13GS, X4, SX, XGSINV,
                        Xb, topk_W, Y, c.N, c.I, c.H);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<float> times(c.repeats);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (uint32_t repeat = 0; repeat < c.repeats; ++repeat) {
            CUDA_CHECK(cudaMemset(Y + partial_u16, 0,
                                  final_u16 * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaEventRecord(start));
            launch_mode(mode, c.E, W13, S13, W13GS, X4, SX, XGSINV,
                        Xb, topk_W, Y, c.N, c.I, c.H);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[repeat], start, stop));
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        std::sort(times.begin(), times.end());
        const double ms = times[times.size() / 2];

        const double useful_tflops = useful_flops / (ms * 1.0e9);
        const double issued_tflops = issued_flops / (ms * 1.0e9);
        const uint64_t pf_bytes = (mode & 1)
            ? assignments * (K64 * 32u + (K64 >> 3) * 32u) : 0u;
        const bool pass = max_rel <= 0.02 && blocks_per_sm[mode] == 1;

        std::fprintf(csv,
            "%s,%d,%s,%s,%s,balanced_exact_live_scattered,0x%08x,"
            "%u,%u,%u,%u,%u,8,%u,%u,%u,%u,"
            "%llu,%llu,%d,%u,%u,"
            "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
            "%.6f,%.6f,%.6f,%.4f,%.9g,%.9g,0x%016llx,%s\n",
            prop.name, prop.multiProcessorCount, mode_name(mode),
            transport_name(mode), prefetch_name(mode), c.route_seed,
            c.repeats, c.E, c.N, c.I, c.H, c.active_experts, live_experts,
            min_tokens_per_live_expert, max_tokens_per_live_expert,
            (unsigned long long)assignments,
            (unsigned long long)active_n8_tiles, blocks_per_sm[mode], smem_bytes,
            (c.N + 255u) >> 8, (unsigned long long)x4_bytes,
            (unsigned long long)sx_bytes, (unsigned long long)pf_bytes,
            (unsigned long long)W_bytes, (unsigned long long)S_bytes,
            (unsigned long long)partial_load_bytes,
            (unsigned long long)partial_store_bytes,
            (unsigned long long)final_reduce_bytes,
            (unsigned long long)workspace_bytes, (unsigned long long)mma_inst,
            ms, useful_tflops, issued_tflops,
            issued_flops ? 100.0 * useful_flops / issued_flops : 0.0,
            max_abs, max_rel, (unsigned long long)sum, pass ? "PASS" : "FAIL");
        std::fflush(csv);

        std::printf("%-38s %9.4f ms  useful %8.2f TF  issued %8.2f TF  %s\n",
                    mode_name(mode), ms, useful_tflops, issued_tflops,
                    pass ? "PASS" : "FAIL");
    }
    std::fclose(csv);
    std::printf("CSV: %s\n", c.csv_path);

    CUDA_CHECK(cudaFree(W13));
    CUDA_CHECK(cudaFree(S13));
    CUDA_CHECK(cudaFree(W13GS));
    CUDA_CHECK(cudaFree(X4));
    CUDA_CHECK(cudaFree(SX));
    CUDA_CHECK(cudaFree(Xb));
    CUDA_CHECK(cudaFree(topk_W));
    CUDA_CHECK(cudaFree(Y));
    return 0;
}
