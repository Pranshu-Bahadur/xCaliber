#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define STREAM_CTA 1024
#define STREAM_PARTIAL_BYTES 8192u
#define STREAM_OUT_BYTES 1024u
#define STREAM_SMEM_BYTES (STREAM_PARTIAL_BYTES + STREAM_OUT_BYTES)

#define STREAM_T ((uint32_t)threadIdx.x)
#define STREAM_Q (STREAM_T & 255u)
#define STREAM_LP (STREAM_T & 3u)
#define STREAM_G ((STREAM_T & 31u) >> 2)
#define STREAM_H64 ((uint32_t)H >> 6)
#define STREAM_P (((uint32_t)H + 255u) >> 8)

__device__ __constant__ uint32_t XbMaskLUT[2] = {0u, 0xffffffffu};

/*
    Dense K64 stream board

    CTA1024 = one expert CTA / SM

      b = tid>>8          four H stripes
      q = tid&255         eight I8 warps / band
      g = lane>>2         I row in one paired W1/W3 fragment
      p = lane&3          two N columns

    One weight step spans four K64 stripes = H256:

      band0 K64(4*wp+0)  band1 K64(4*wp+1)
      band2 K64(4*wp+2)  band3 K64(4*wp+3)

    W13[e,H64,I64,warp8,lane,v4]
      one band / wp = 256T * 16B = 4KB
      one CTA  / wp =   4 * 4KB   = 16KB
      A rows 0..7 = preflipped W1, rows 8..15 = preflipped W3

    S13[e,H64,I64,pad256]
      p0/p1 supply the paired W1/W3 UE4M3 vectors

    X4[token,H64,p,{K0,K32}] = one contiguous v2 per MMA lane
    SX[token,H64]             = one UE4M3x4 word

    loop = n256 -> i64 -> n8 -> kt
      W13 streams with ld.global.cs.nc
      X4/SX first i64 pass warms L2; later i64 planes reuse it
      8KB shared joins four bands once after the kt loop

    activation PF
      one p2 lane / token / band hints one 32B K64 row
      no wait; never shares a bulk group with cp.reduce

    W13 is already sign-bit flipped. No runtime xor belongs here.
*/

#define STREAM_MMA(PSTEP, A0, A1, A2, A3) do {                            \
    uint32_t b0_, b1_, sb_;                                                \
    uint32_t cfg_ = STREAM_H64 | ((uint32_t)n256 << 8)                        \
                  | ((uint32_t)n8 << 10) | (live_expert << 15)             \
                  | ((uint32_t)(PSTEP) << 23);                             \
    asm volatile(                                                           \
        "{\n\t"                                                            \
        ".reg .pred p;\n\t"                                                \
        ".reg .b32 t, off;\n\t"                                            \
        ".reg .b64 a;\n\t"                                                \
        "mov.b32 %0, 0;\n\t"                                               \
        "mov.b32 %1, 0;\n\t"                                               \
        "mov.b32 %2, 0;\n\t"                                               \
        "mov.u32 t, %%tid.x;\n\t"                                         \
        "shr.u32 off, t, 8;\n\t"                                           \
        "bfe.u32 %1, %5, 23, 5;\n\t"                                     \
        "mad.lo.u32 off, %1, 4, off;\n\t"                                  \
        "bfe.u32 %1, %5, 0, 8;\n\t"                                      \
        "setp.ge.u32 p, off, %1;\n\t"                                      \
        "@p bra.uni STREAM_B_done_%=;\n\t"                                    \
        "bfe.u32 %1, t, 2, 3;\n\t"                                       \
        "bfe.u32 %0, %5, 15, 8;\n\t"                                     \
        "bfe.u32 %0, %0, %1, 1;\n\t"                                     \
        "setp.eq.u32 p, %0, 0;\n\t"                                       \
        "@p bra STREAM_B_done_%=;\n\t"                                        \
        "bfe.u32 %0, %5, 10, 5;\n\t"                                     \
        "shl.b32 %0, %0, 3;\n\t"                                         \
        "bfe.u32 %1, %5, 8, 2;\n\t"                                      \
        "mad.lo.u32 %0, %1, 256, %0;\n\t"                                \
        "bfe.u32 %1, t, 2, 3;\n\t"                                       \
        "add.u32 %0, %0, %1;\n\t"                                         \
        "bfe.u32 %1, %5, 0, 8;\n\t"                                      \
        "mad.lo.u32 %1, %0, %1, off;\n\t"                                \
        "bfe.u32 off, t, 0, 2;\n\t"                                      \
        "setp.ne.u32 p, off, 0;\n\t"                                      \
        "@p bra STREAM_SX_done_%=;\n\t"                                       \
        "mad.wide.u32 a, %1, 4, %4;\n\t"                                  \
        "ld.global.ca.nc.u32 %2, [a];\n\t"                                  \
        "STREAM_SX_done_%=:\n\t"                                              \
        "shl.b32 %1, %1, 5;\n\t"                                         \
        "mad.wide.u32 a, %1, 1, %3;\n\t"                                  \
        "mad.wide.u32 a, off, 8, a;\n\t"                                  \
        "ld.global.ca.nc.v2.u32 {%0,%1}, [a];\n\t"                       \
        "STREAM_B_done_%=:\n\t"                                               \
        "}"                                                                 \
        : "=&r"(b0_), "=&r"(b1_), "=&r"(sb_)                             \
        : "l"((uint64_t)__cvta_generic_to_global(X4)),                     \
          "l"((uint64_t)__cvta_generic_to_global(SX)), "r"(cfg_)            \
        : "memory");                                                        \
    cfg_ = STREAM_H64 | ((uint32_t)IP << 8) | ((uint32_t)i << 21)          \
         | ((uint32_t)(PSTEP) << 27);                                      \
    asm volatile(                                                           \
        "{\n\t"                                                            \
        ".reg .pred p;\n\t"                                                \
        ".reg .b32 t, k, off;\n\t"                                         \
        ".reg .b64 a;\n\t"                                                \
        "mov.u32 t, %%tid.x;\n\t"                                         \
        "bfe.u32 off, t, 0, 2;\n\t"                                      \
        "setp.ge.u32 p, off, 2;\n\t"                                      \
        "@p bra STREAM_S_zero_%=;\n\t"                                        \
        "shr.u32 k, t, 8;\n\t"                                             \
        "bfe.u32 off, %0, 27, 5;\n\t"                                    \
        "mad.lo.u32 k, off, 4, k;\n\t"                                     \
        "bfe.u32 off, %0, 0, 8;\n\t"                                    \
        "setp.ge.u32 p, k, off;\n\t"                                       \
        "@p bra.uni STREAM_S_zero_%=;\n\t"                                    \
        "bfe.u32 t, %0, 21, 6;\n\t"                                     \
        "bfe.u32 %0, %0, 8, 13;\n\t"                                    \
        "shl.b32 %0, %0, 2;\n\t"                                         \
        "shl.b32 t, t, 16;\n\t"                                           \
        "lop3.b32 %0, %0, t, 0, 0xfc;\n\t"                               \
        "mov.u32 t, %%ctaid.x;\n\t"                                      \
        "mad.lo.u32 off, t, off, k;\n\t"                                \
        "bfe.u32 t, %0, 0, 14;\n\t"                                     \
        "mul.lo.u32 off, off, t;\n\t"                                    \
        "bfe.u32 t, %0, 16, 6;\n\t"                                     \
        "mad.lo.u32 off, t, 256, off;\n\t"                               \
        "mov.u32 t, %%tid.x;\n\t"                                         \
        "bfe.u32 k, t, 0, 2;\n\t"                                        \
        "mad.lo.u32 off, k, 64, off;\n\t"                               \
        "bfe.u32 k, t, 2, 6;\n\t"                                        \
        "add.u32 off, off, k;\n\t"                                      \
        "mad.wide.u32 a, off, 4, %1;\n\t"                                \
        "ld.global.cs.nc.u32 %0, [a];\n\t"                                  \
        "bra.uni STREAM_S_done_%=;\n\t"                                      \
        "STREAM_S_zero_%=:\n\t"                                              \
        "mov.b32 %0, 0;\n\t"                                               \
        "STREAM_S_done_%=:\n\t"                                               \
        "}"                                                                 \
        : "+r"(cfg_)                                                        \
        : "l"((uint64_t)__cvta_generic_to_global(S13))                     \
        : "memory");                                                        \
    asm volatile(                                                           \
        "{\n\t"                                                            \
        ".reg .b16 z, h0, h1, h2, h3;\n\t"                                \
        ".reg .f32 c1, c3;\n\t"                                            \
        "mov.b16 z, 0;\n\t"                                                \
        "mov.b32 {h0,h1}, %0;\n\t"                                       \
        "mov.b32 {h2,h3}, %1;\n\t"                                       \
        "cvt.f32.bf16 %0, h0;\n\t"                                        \
        "cvt.f32.bf16 c1, h1;\n\t"                                        \
        "cvt.f32.bf16 %1, h2;\n\t"                                        \
        "cvt.f32.bf16 c3, h3;\n\t"                                        \
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4."               \
        "block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "              \
        "{%0,c1,%1,c3}, {%2,%3,%4,%5}, {%6,%7}, "                         \
        "{%0,c1,%1,c3}, %8, {z,z}, %9, {z,z};\n\t"                    \
        "cvt.rn.bf16x2.f32 %0, c1, %0;\n\t"                              \
        "cvt.rn.bf16x2.f32 %1, c3, %1;\n\t"                              \
        "}"                                                                 \
        : "+r"(d01), "+r"(d23)                                            \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(b0_), "r"(b1_),     \
          "r"(cfg_), "r"(sb_));                                            \
} while (0)

__global__ __launch_bounds__(STREAM_CTA, 1)
void kernel_stream(
    const uint32_t* __restrict__ W13,
    const uint32_t* __restrict__ S13,
    const __nv_bfloat16* __restrict__ W13GS,
    const uint32_t* __restrict__ X4,
    const uint32_t* __restrict__ SX,
    __nv_bfloat16 XGSINV,
    const uint32_t* __restrict__ Xb,
    const __nv_bfloat16* __restrict__ topk_W,
    __nv_bfloat16* __restrict__ Y,
    int Xb_stride,
    int NP,
    int IP,
    int I,
    int H
) {
    extern __shared__ __align__(16) unsigned char smem[];

    uint32_t live_expert;

    asm volatile(
        "ldu.global.u32 %0, [%1];"
        : "=r"(live_expert)
        : "l"((uint64_t)__cvta_generic_to_global(
            Xb + (uint64_t)blockIdx.x * (uint32_t)Xb_stride))
        : "memory");
    if (!live_expert) return;

    for (int n256 = 0; n256 < ((NP + 255) >> 8); n256++) {
        uint32_t route_word = 0u;
        if ((STREAM_T & 31u) < 8u
            && (n256 << 3) + (STREAM_T & 31u) < Xb_stride - 1) {
            asm volatile(
                "ldu.global.u32 %0, [%1];"
                : "=r"(route_word)
                : "l"((uint64_t)__cvta_generic_to_global(
                    Xb + (uint64_t)blockIdx.x * (uint32_t)Xb_stride
                    + 1u + (n256 << 3) + (STREAM_T & 31u)))
                : "memory");
        }

        for (int i = 0; i < (IP >> 6); i++) {
            for (int n8 = 0; n8 < 32; n8++) {
                live_expert = __shfl_sync(0xffffffffu, route_word, n8 >> 2);
                live_expert = (live_expert >> ((n8 & 3) << 3)) & 255u;

                uint32_t d01 = 0u, d23 = 0u;
                for (uint32_t kt = 0; kt < STREAM_P; kt++) {
                if (!live_expert) break;
                if (!i && (STREAM_Q >> 5) == 0u && STREAM_LP == 2u
                    && (live_expert & (1u << STREAM_G))
                    && (kt << 2) + (STREAM_T >> 8) < STREAM_H64) {
                    asm volatile(
                        "cp.async.bulk.prefetch.L2.global [%0], 32;"
                        :
                        : "l"((uint64_t)__cvta_generic_to_global(
                            X4
                            + ((uint64_t)((n256 << 8) + (n8 << 3) + STREAM_G)
                               * STREAM_H64 + (kt << 2) + (STREAM_T >> 8)) * 8u))
                        : "memory");
                    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
                }

                uint32_t a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
                if ((kt << 2) + (STREAM_T >> 8) < STREAM_H64) {
                    asm volatile(
                        "ld.global.cs.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                        : "l"((uint64_t)__cvta_generic_to_global(
                            W13
                            + (uint64_t)blockIdx.x * STREAM_H64 * (IP << 4)
                            + (uint64_t)((kt << 2) + (STREAM_T >> 8)) * (IP << 4)
                            + ((uint32_t)i << 10) + (STREAM_Q << 2)))
                        : "memory");
                }
                STREAM_MMA(kt, a0, a1, a2, a3);

                }

                if (!live_expert) continue;

                asm volatile(
                    "st.shared.v2.u32 [%0], {%1,%2};"
                    :
                    : "r"((uint32_t)__cvta_generic_to_shared(
                          smem + (STREAM_T << 3))),
                      "r"(d01), "r"(d23)
                    : "memory");
                asm volatile("barrier.cta.sync 0;" ::: "memory");

                if ((STREAM_T >> 8) == 0u) {
                    uint32_t gate, up;
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 x, y;\n\t"
                        "ld.shared.v2.u32 {%0,%1}, [%2];\n\t"
                        "ld.shared.v2.u32 {x,y}, [%2 + 2048];\n\t"
                        "add.rn.bf16x2 %0, %0, x;\n\t"
                        "add.rn.bf16x2 %1, %1, y;\n\t"
                        "ld.shared.v2.u32 {x,y}, [%2 + 4096];\n\t"
                        "add.rn.bf16x2 %0, %0, x;\n\t"
                        "add.rn.bf16x2 %1, %1, y;\n\t"
                        "ld.shared.v2.u32 {x,y}, [%2 + 6144];\n\t"
                        "add.rn.bf16x2 %0, %0, x;\n\t"
                        "add.rn.bf16x2 %1, %1, y;\n\t"
                        "}"
                        : "=&r"(gate), "=&r"(up)
                        : "r"((uint32_t)__cvta_generic_to_shared(
                            smem + (STREAM_T << 3)))
                        : "memory");

                    asm volatile(
                        "{\n\t"
                        ".reg .b16 xgs;\n\t"
                        ".reg .b32 post, scale;\n\t"
                        "mov.b16 xgs, %2;\n\t"
                        "mov.b32 scale, {xgs,xgs};\n\t"
                        "ldu.global.u32 post, [%3];\n\t"
                        "mul.rn.bf16x2 post, post, scale;\n\t"
                        "lop3.b32 scale, post, 0x0000ffff, 0, 0xc0;\n\t"
                        "bfi.b32 scale, scale, scale, 16, 16;\n\t"
                        "mul.rn.bf16x2 %0, %0, scale;\n\t"
                        "shr.u32 scale, post, 16;\n\t"
                        "bfi.b32 scale, scale, scale, 16, 16;\n\t"
                        "mul.rn.bf16x2 %1, %1, scale;\n\t"
                        "}"
                        : "+r"(gate), "+r"(up)
                        : "h"(__bfloat16_as_ushort(XGSINV)),
                          "l"((uint64_t)__cvta_generic_to_global(
                            W13GS + (uint64_t)blockIdx.x * 2u))
                        : "memory");

                        asm volatile(
                        "{\n\t"
                        ".reg .b32 half;\n\t"
                        "mul.rn.bf16x2 %0, %0, %1;\n\t"
                        "mov.b32 half, 0x3f003f00;\n\t"
                        "mul.rn.bf16x2 %1, %0, half;\n\t"
                        "tanh.approx.bf16x2 %1, %1;\n\t"
                        "fma.rn.bf16x2 %1, %1, half, half;\n\t"
                        "mul.rn.bf16x2 %0, %0, %1;\n\t"
                        "}"
                        : "+r"(gate), "+r"(up));

                        up = 0u;
                        if (!(STREAM_T & 28u)) {
                            asm volatile(
                            "ld.global.ca.nc.u32 %0, [%1];"
                            : "=r"(up)
                            : "l"((uint64_t)__cvta_generic_to_global(
                                topk_W + (uint64_t)blockIdx.x * (uint32_t)NP
                                + (n256 << 8) + (n8 << 3) + (STREAM_LP << 1)))
                            : "memory");
                        }
                        up = __shfl_sync(0xffffffffu, up, STREAM_LP);

                        asm volatile(
                        "mul.rn.bf16x2 %0, %0, %2;\n\t"
                        "st.shared.u32 [%1], %0;"
                        : "+r"(gate)
                        : "r"((uint32_t)__cvta_generic_to_shared(
                            smem + STREAM_PARTIAL_BYTES + (STREAM_T << 2))),
                          "r"(up)
                        : "memory");

                        asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
                        if (!STREAM_LP
                            && ((i << 6) + ((STREAM_T >> 5) << 3) + STREAM_G) < I) {
                            asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group."
                            "add.noftz.bf16 [%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;\n\t"
                            "cp.async.bulk.wait_group 0;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                Y
                                + (uint64_t)((i << 6) + ((STREAM_T >> 5) << 3) + STREAM_G)
                                    * (uint32_t)NP
                                + (n256 << 8) + (n8 << 3))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                smem + STREAM_PARTIAL_BYTES
                                + ((STREAM_T >> 2) << 4)))
                            : "memory");
                        }
                }
                asm volatile("barrier.cta.sync 0;" ::: "memory");
            }
            }
        }
    }

#undef STREAM_MMA
#undef STREAM_T
#undef STREAM_Q
#undef STREAM_LP
#undef STREAM_G
#undef STREAM_H64
#undef STREAM_P
