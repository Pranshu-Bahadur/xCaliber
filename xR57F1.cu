// xR57F1: SM120a W4A4 FF1 -> SwiGLU -> topk_W -> BF16 bulk-reduce.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define CTA 1024

// One routed bit -> one all-ones/all-zero MMA fragment mask.
__device__ __constant__ uint32_t XbMaskLUT[2] = {0u, 0xffffffffu};

/*
    Kernel contract

    W13[e,kt,i1024,plane4,t1024,v4]
      one plane = 16KB = CTA x one lane-native m16n8k64 A fragment
      A rows 0..7 = W1 I8, A rows 8..15 = W3 matching I8
      input contract: e2m1 sign bits are already flipped for exp(-A)
      no runtime sign-bit transform

    S13[e,kt,i1024,plane4,half2,q256,pad512]
      plane span = 1024 u32 to preserve the live I<<2 checkpoint stride
      half0 -> p0 / W1 row g, half1 -> p1 / W3 row g+8

    X4[kt,n16,lane,{n0lo,n0hi,n1lo,n1hi}] u32
      one K64 / N16 = 32 lanes x 16B = 512B
      L2 hint once at CTA scope, then routed lanes load B registers directly

    Sx[kt,n16,g,{n0,n1}] u32
      one K64 / N16 = 8 token rows x two UE4M3x4 words = 64B

    Four 256T prefetch bands each own eight K64 rows per H2048 stage.
    Every activation sector is hinted once; consumers do not wait on PF.

    routed N16 pairing
      scan Xb for the next two live N16 tiles, adjacency not required
      load one W13/S13 packet -> 2 N8 MMAs per live N16
      W/S replay = sum_e ceil(live_n16[e]/2), no global partial workspace

    sync
      output / four lanes     warp publish -> one 16B cp.reduce issuer
      dynamic SMEM            CTA*4B fragment-native BF16 output only

    post13 = BF16x2(W13GS[0/1] * XGSINV)

    topk_W[e,NP] BF16
      routed expert-major sidecar emitted with Xb

    X[I,NP] BF16
      zero before launch; this proof reduces topk-weighted SwiGLU directly
*/
__global__ __launch_bounds__(CTA, 1)
void xR57F1(
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

    uint32_t live16, live16b;
    uint32_t xbm0, xbm1, xbm2, xbm3;
    uint32_t a0, a1, a2, a3;
    uint32_t b00, b01, b10, b11;
    uint32_t scaleA, scaleB0, scaleB1;
    uint32_t tw, out, post13, xgsinv2;
    uint16_t selector0 = 0;
    float d00, d01, d10, d11;
    float s00, s01, s02, s03;
    float s10, s11, s12, s13;
    float t00, t01, t02, t03;
    float t10, t11, t12, t13;
    float post1, post3, x1, x3, o0, o1, topk0, topk1;
    int n16b;
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


                for (int hp = 0; hp < ((H + 2047) >> 11); hp++) {
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

                    for (int k = 0; k < 32 && ((hp << 5) + k) < (H >> 6); k++) {
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
                                    + ((uint64_t)((hp << 5) + k)
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
                                    + ((uint64_t)((hp << 5) + k)
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
                                        + ((uint64_t)((hp << 5) + k)
                                            * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                        + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1))
                                    : "memory");
                            }
                            if (xbm1) {
                                asm volatile(
                                    "ld.global.ca.nc.u32 %0, [%1];"
                                    : "=r"(scaleB1)
                                    : "l"(Sx
                                        + ((uint64_t)((hp << 5) + k)
                                            * ((uint32_t)NP >> 4) + (uint32_t)n16) * 16u
                                        + ((((uint32_t)threadIdx.x & 31u) >> 2) << 1) + 1u)
                                    : "memory");
                            }
                        }

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

                        // W1/W3 arrive sign-negated; their signs cancel in the
                        // SwiGLU elementwise product.

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
                    }

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
                        smem)
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
                                  smem
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
                        smem)
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
                                  smem
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
                        smem)
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
                                  smem
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
                        smem)
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
                                  smem
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
