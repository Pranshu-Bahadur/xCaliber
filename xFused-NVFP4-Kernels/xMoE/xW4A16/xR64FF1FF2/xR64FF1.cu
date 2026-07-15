// xR57F1_modelopt_graph: conventional NVFP4 FF1 -> paired NVFP4 Y.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "preamble.cuh"

#define CTA 1024
#define F1Y4_SMEM_BYTES 16512

/*
    Kernel contract

    ModelOpt checkpoint tensors, stacked by the loader over expert e:
      W13 U8[projection2,e,I,H/2]       two E2M1 values per byte
      S13 E4M3[projection2,e,I,H/16]    one scale per K16
      W13GS F32[projection2,e]

    register panel, one warp / K64:
      a0 = W1[row g][lp*8 +  0.. 7]
      a1 = W3[row g][lp*8 +  0.. 7]
      a2 = W1[row g][lp*8 + 32..39]
      a3 = W3[row g][lp*8 + 32..39]
      t0..t3 form one contiguous 16B row half; t4..t7 form the next row
      lp0 in each W1/W3 block prefetches its aligned 32B row fragment to L2
      coherent .cs scalar loads preserve the PTX m16n8k64 fragment without
      a checkpoint repack, shuffle, or intermediate global panel
      xor 0x88888888 in a0..a3 supplies the existing negated-W1/W3 identity

    S1/S3 row-major K16 scales:
      lp0 supplies W1 row g, lp1 supplies W3 row g

    X4[e,n16,H64,q8,lp4,{n8_0x2,n8_1x2}]
      one lane-native v4 materializes both adjacent N8 B fragments

    Sx[e,n32,H64,q8,{n16a_0,n16a_1,n16b_0,n16b_1}]
      one lane-native v4 materializes four adjacent N8 scale fragments

    routed N16 pairing
      expert header count -> adjacent packed N16 pairs
      no sparse scan and no activation prefetch
      materialize one conventional W1/W3 register panel -> up to four N8 MMAs

    Y4[e,n16,I64,q8,lp4,{n8_0x2,n8_1x2}]
      one FF2 lane reads both N8 B fragments with one v4
      v4 = {B0[2lp+0],B0[2lp+1],B1[2lp+0],B1[2lp+1]}

    SY[e,n32,I64,q8,{n16a_0,n16a_1,n16b_0,n16b_1}]
      one FF2 scale lane reads all four MMA scales with one v4
      allocation = final SY + transient BF16 I16 absmax tail
        final     = E x NP x I64 u32
        transient = E x NP x I64 x 2 u32; dead after return

    YGSINV[e]
      one expert BF16x2 inverse-scale sidecar

    dynamic SMEM
      16KB                    four BF16 I256/N8 output tiles
      32*u32                  expert-max warp reduction tail

    XGSINV2 = BF16x2({XGSINV,XGSINV}) produced by the ACT preamble
    post1/post3 = F32(W1GS/W3GS) * BF16(XGSINV2.lo)

    topk_W[e,NP] BF16 in packed expert-local row order

    expert_token_idx[e,0] = routed packed row count

    Y4/SY/YGSINV retain expert identity for the routed FF2 replay.
*/
__global__ __launch_bounds__(CTA, 1)
void xR57F1_modelopt_graph(
    const uint32_t* __restrict__ W13,
    const uint32_t* __restrict__ S13,
    const float* __restrict__ W13GS,
    const uint32_t* __restrict__ X4,
    const uint32_t* __restrict__ Sx,
    const uint32_t* __restrict__ XGSINV2,
    const __nv_bfloat16* __restrict__ topk_W,
    const uint16_t* __restrict__ expert_token_idx,
    uint32_t* __restrict__ Y4,
    uint32_t* __restrict__ SY,
    uint32_t* __restrict__ YGSINV,
    int NP,
    int I,
    int H
) {

    extern __shared__ __align__(16) unsigned char smem[];

    uint32_t count, live16, live16b;
    uint32_t xbm0, xbm1, xbm2, xbm3;
    uint32_t a0, a1, a2, a3;
    uint32_t b00, b01, b10, b11;
    uint32_t scaleA, scaleB0, scaleB1, scaleB2, scaleB3;
    uint32_t tw, out, xgsinv2;
    uint16_t selector0 = 0;
    float d00, d01, d10, d11;
    float s00, s01, s02, s03;
    float s10, s11, s12, s13;
    float t00, t01, t02, t03;
    float t10, t11, t12, t13;
    float post1, post3, x1, x3, o0, o1, topk0, topk1;
    uint32_t yrun = 0u;
    int n16b;
    asm volatile("ldu.global.u16 %0, [%1];"
                 : "=h"(selector0)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       expert_token_idx
                       + (uint64_t)blockIdx.x * ((uint32_t)NP + 1u)))
                 : "memory");
    count = selector0;
    selector0 = 0;
    if (!count) return;

    asm volatile("ldu.global.f32 %0, [%1];"
                 : "=f"(post1)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       W13GS + (uint32_t)blockIdx.x))
                 : "memory");
    asm volatile("ldu.global.f32 %0, [%1];"
                 : "=f"(post3)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       W13GS + (uint32_t)gridDim.x
                           + (uint32_t)blockIdx.x))
                 : "memory");
    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(xgsinv2)
                 : "l"((uint64_t)__cvta_generic_to_global(XGSINV2))
                 : "memory");
    selector0 = (uint16_t)xgsinv2;
    asm volatile("cvt.f32.bf16 %0, %1;"
                 : "=f"(x1)
                 : "h"(selector0));
    selector0 = 0;
    post1 *= x1;
    post3 *= x1;
    xgsinv2 = 0u;

    for (int n16 = 0; (uint32_t)(n16 << 4) < count; n16++) {
        n16b = n16 + 1;
        live16 = count - (uint32_t)(n16 << 4);
        live16 = live16 >= 16u ? 0xffffu : (1u << live16) - 1u;
        live16b = count > (uint32_t)(n16b << 4)
            ? count - (uint32_t)(n16b << 4) : 0u;
        live16b = live16b >= 16u ? 0xffffu
            : (live16b ? (1u << live16b) - 1u : 0u);
        xbm0 = 0u - ((uint32_t)(n16 << 4)
            + (((uint32_t)threadIdx.x & 31u) >> 2) < count);
        xbm1 = 0u - ((uint32_t)(n16 << 4) + 8u
            + (((uint32_t)threadIdx.x & 31u) >> 2) < count);
        xbm2 = 0u - ((uint32_t)(n16b << 4)
            + (((uint32_t)threadIdx.x & 31u) >> 2) < count);
        xbm3 = 0u - ((uint32_t)(n16b << 4) + 8u
            + (((uint32_t)threadIdx.x & 31u) >> 2) < count);

        for (int i = 0; i < ((I + 1023) >> 10); i++) {
            // Four I256 planes are independent. Serializing them keeps only
            // one plane's paired four-N8 consumer set live.
            for (int plane = 0;
                 plane < 4 && ((i << 10) + (plane << 8)) < I;
                 plane++) {
                s00 = 0.0f; s01 = 0.0f; s02 = 0.0f; s03 = 0.0f;
                s10 = 0.0f; s11 = 0.0f; s12 = 0.0f; s13 = 0.0f;
                t00 = 0.0f; t01 = 0.0f; t02 = 0.0f; t03 = 0.0f;
                t10 = 0.0f; t11 = 0.0f; t12 = 0.0f; t13 = 0.0f;


                for (int hp = 0; hp < ((H + 2047) >> 11); hp++) {
                    for (int k = 0; k < 32 && ((hp << 5) + k) < (H >> 6); k++) {
                        // One v4 per lane materializes both packed N8 fragments.
                        b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                        if (xbm0 | xbm1) {
                            asm volatile(
                                "ld.global.ca.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(b00), "=r"(b01),
                                  "=r"(b10), "=r"(b11)
                                : "l"(X4
                                    + (uint64_t)blockIdx.x * (uint32_t)NP
                                        * ((uint32_t)H >> 3)
                                    + (uint64_t)n16 * ((uint32_t)H << 1)
                                    + ((uint64_t)((hp << 5) + k) << 7)
                                    + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        * 16u
                                    + (((uint32_t)threadIdx.x & 3u) << 2))
                                : "memory");
                        }
                        b00 &= xbm0; b01 &= xbm0;
                        b10 &= xbm1; b11 &= xbm1;

                        scaleB0 = 0u; scaleB1 = 0u;
                        scaleB2 = 0u; scaleB3 = 0u;
                        if (!((uint32_t)threadIdx.x & 3u)
                            && (xbm0 | xbm1 | xbm2 | xbm3)) {
                            asm volatile(
                                "ld.global.ca.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(scaleB0), "=r"(scaleB1),
                                  "=r"(scaleB2), "=r"(scaleB3)
                                : "l"(Sx
                                    + (uint64_t)blockIdx.x * (uint32_t)NP
                                        * ((uint32_t)H >> 6)
                                    + (uint64_t)(n16 >> 1)
                                        * (((uint32_t)H >> 6) << 5)
                                    + (uint32_t)((hp << 5) + k) * 32u
                                    + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        * 4u)
                                : "memory");
                        }
                        scaleB0 &= xbm0; scaleB1 &= xbm1;
                        scaleB2 &= xbm2; scaleB3 &= xbm3;

                        // ModelOpt [out, packed-in] -> PTX A fragment in rmem.
                        a0 = 0u; a1 = 0u; a2 = 0u; a3 = 0u;
                        if ((i << 10) + (plane << 8)
                            + ((uint32_t)threadIdx.x >> 2) < (uint32_t)I) {
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                ".reg .b32 q;\n\t"
                                "mov.u32 q, %%laneid;\n\t"
                                "and.b32 q, q, 3;\n\t"
                                "setp.eq.u32 p, q, 0;\n\t"
                                "@p cp.async.bulk.prefetch.L2.global [%2], 32;\n\t"
                                "ld.global.cs.u32 %0, [%2];\n\t"
                                "ld.global.cs.u32 %1, [%2 + 16];\n\t"
                                "}"
                                : "=r"(a0), "=r"(a2)
                                : "l"(W13
                                    + ((uint64_t)blockIdx.x * (uint32_t)I
                                        + (uint32_t)(i << 10)
                                        + (uint32_t)(plane << 8)
                                        + ((uint32_t)threadIdx.x >> 2))
                                        * ((uint32_t)H >> 3)
                                    + (uint32_t)((hp << 5) + k) * 8u
                                    + ((uint32_t)threadIdx.x & 3u))
                                : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p;\n\t"
                                ".reg .b32 q;\n\t"
                                "mov.u32 q, %%laneid;\n\t"
                                "and.b32 q, q, 3;\n\t"
                                "setp.eq.u32 p, q, 0;\n\t"
                                "@p cp.async.bulk.prefetch.L2.global [%2], 32;\n\t"
                                "ld.global.cs.u32 %0, [%2];\n\t"
                                "ld.global.cs.u32 %1, [%2 + 16];\n\t"
                                "}"
                                : "=r"(a1), "=r"(a3)
                                : "l"(W13
                                    + (uint64_t)gridDim.x * (uint32_t)I
                                        * ((uint32_t)H >> 3)
                                    + ((uint64_t)blockIdx.x * (uint32_t)I
                                        + (uint32_t)(i << 10)
                                        + (uint32_t)(plane << 8)
                                        + ((uint32_t)threadIdx.x >> 2))
                                        * ((uint32_t)H >> 3)
                                    + (uint32_t)((hp << 5) + k) * 8u
                                    + ((uint32_t)threadIdx.x & 3u))
                                : "memory");
                            asm volatile(
                                "xor.b32 %0, %0, 0x88888888;\n\t"
                                "xor.b32 %1, %1, 0x88888888;\n\t"
                                "xor.b32 %2, %2, 0x88888888;\n\t"
                                "xor.b32 %3, %3, 0x88888888;"
                                : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3));
                        }

                        scaleA = 0u;
                        if ((threadIdx.x & 3) < 2
                            && (i << 10) + (plane << 8)
                                + ((uint32_t)threadIdx.x >> 2) < (uint32_t)I) {
                            asm volatile(
                                "ld.global.cs.b32 %0, [%1];\n\t"
                                : "=r"(scaleA)
                                : "l"(S13
                                    + (uint64_t)(threadIdx.x & 3)
                                        * (uint32_t)gridDim.x * (uint32_t)I
                                        * ((uint32_t)H >> 6)
                                    + ((uint64_t)blockIdx.x * (uint32_t)I
                                        + (uint32_t)(i << 10)
                                        + (uint32_t)(plane << 8)
                                        + ((uint32_t)threadIdx.x >> 2))
                                        * ((uint32_t)H >> 6)
                                    + (uint32_t)((hp << 5) + k))
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
                            if (xbm2 | xbm3) {
                                asm volatile(
                                    "ld.global.ca.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(b00), "=r"(b01),
                                      "=r"(b10), "=r"(b11)
                                    : "l"(X4
                                        + (uint64_t)blockIdx.x * (uint32_t)NP
                                            * ((uint32_t)H >> 3)
                                        + (uint64_t)n16b * ((uint32_t)H << 1)
                                        + ((uint64_t)((hp << 5) + k) << 7)
                                        + (((uint32_t)threadIdx.x & 31u) >> 2)
                                            * 16u
                                        + (((uint32_t)threadIdx.x & 3u) << 2))
                                    : "memory");
                            }
                            b00 &= xbm2; b01 &= xbm2;
                            b10 &= xbm3; b11 &= xbm3;
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n\t"
                                : "+f"(t00), "+f"(t01), "+f"(t02), "+f"(t03)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b00), "r"(b01), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB2),
                                  "h"(selector0), "h"(selector0));
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};\n\t"
                                : "+f"(t10), "+f"(t11), "+f"(t12), "+f"(t13)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b10), "r"(b11), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB3),
                                  "h"(selector0), "h"(selector0));
                        }
                    }

                }
                asm volatile("fence.acquire.cta;" ::: "memory");

                reinterpret_cast<uint32_t*>(smem)[threadIdx.x] = 0u;
                reinterpret_cast<uint32_t*>(smem)[1024u + threadIdx.x] = 0u;
                reinterpret_cast<uint32_t*>(smem)[2048u + threadIdx.x] = 0u;
                reinterpret_cast<uint32_t*>(smem)[3072u + threadIdx.x] = 0u;

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
                        [1024u + threadIdx.x] = out;
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
                        [2048u + threadIdx.x] = out;
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
                        [3072u + threadIdx.x] = out;
                }

                __syncthreads();

                // One warp owns one original token x I64.  Popcount compacts
                // only live rows into ascending packed-token order.
                for (uint32_t yp = 0; yp < 1u + (live16b != 0u); yp++) {
                    for (uint32_t half = 0; half < 2; half++) {
                        uint32_t yrow = (((uint32_t)threadIdx.x >> 8) << 6)
                                      + (((uint32_t)threadIdx.x & 31u) << 1);
                        uint32_t yshift
                            = ((((uint32_t)threadIdx.x & 255u) >> 5) & 1u) << 4;
                        uint32_t y0 = reinterpret_cast<uint32_t*>(smem)[
                            ((yp << 1) + half) * 1024u
                            + yrow * 4u
                            + (((uint32_t)threadIdx.x & 255u) >> 6)];
                        uint32_t y1 = reinterpret_cast<uint32_t*>(smem)[
                            ((yp << 1) + half) * 1024u
                            + (yrow + 1u) * 4u
                            + (((uint32_t)threadIdx.x & 255u) >> 6)];
                        y0 = ((y0 >> yshift) & 0xffffu)
                           | (((y1 >> yshift) & 0xffffu) << 16);

                        uint32_t ymx = max(y0 & 0x7fffu,
                                          (y0 >> 16) & 0x7fffu);
                        ymx = max(ymx, __shfl_xor_sync(
                            0xffffffffu, ymx, 1, 8));
                        ymx = max(ymx, __shfl_xor_sync(
                            0xffffffffu, ymx, 2, 8));
                        ymx = max(ymx, __shfl_xor_sync(
                            0xffffffffu, ymx, 4, 8));
                        yrun = max(yrun, ymx);

                        uint32_t yscale = 0u;
                        if (!((uint32_t)threadIdx.x & 7u) && ymx) {
                            uint32_t ybf = __bfloat16_as_ushort(
                                __float2bfloat16(
                                    6.0f / __uint_as_float(ymx << 16)));
                            yscale = ybf | (ybf << 16);
                        }
                        yscale = __shfl_sync(
                            0xffffffffu, yscale, 0, 8);
                        uint32_t yq = ymx
                            ? moe_e2m1x2_bf16x2(y0, yscale) : 0u;
                        uint32_t yq0 = __shfl_sync(
                            0xffffffffu, yq,
                            ((uint32_t)threadIdx.x & 31u) & ~3u);
                        uint32_t yq1 = __shfl_sync(
                            0xffffffffu, yq,
                            (((uint32_t)threadIdx.x & 31u) & ~3u) + 1u);
                        uint32_t yq2 = __shfl_sync(
                            0xffffffffu, yq,
                            (((uint32_t)threadIdx.x & 31u) & ~3u) + 2u);
                        uint32_t yq3 = __shfl_sync(
                            0xffffffffu, yq,
                            (((uint32_t)threadIdx.x & 31u) & ~3u) + 3u);
                        tw = ((yp ? (uint32_t)n16b : (uint32_t)n16) << 4)
                           + (half << 3)
                           + (((uint32_t)threadIdx.x & 255u) >> 5);
                        if (!((uint32_t)threadIdx.x & 3u)
                            && ((yp ? live16b : live16)
                                & (1u << ((half << 3)
                                    + (((uint32_t)threadIdx.x & 255u) >> 5))))
                            && ((uint32_t)i << 4) + ((uint32_t)plane << 2)
                                + ((uint32_t)threadIdx.x >> 8)
                                < ((uint32_t)I >> 6)) {
                            Y4[(uint64_t)blockIdx.x * (uint32_t)NP
                                    * ((uint32_t)I >> 3)
                                + (uint64_t)(tw >> 4) * ((uint32_t)I << 1)
                                + ((uint64_t)(((uint32_t)i << 4)
                                  + ((uint32_t)plane << 2)
                                  + ((uint32_t)threadIdx.x >> 8)) << 7)
                                + ((uint32_t)(tw & 7u) << 4)
                                + (((uint32_t)threadIdx.x & 31u) >> 3) * 4u
                                + ((tw & 8u) >> 2)
                                + (((uint32_t)threadIdx.x >> 2) & 1u)]
                                = yq0 | (yq1 << 8)
                                | (yq2 << 16) | (yq3 << 24);
                        }
                        uint32_t ym0 = __shfl_sync(0xffffffffu, ymx, 0);
                        uint32_t ym1 = __shfl_sync(0xffffffffu, ymx, 8);
                        uint32_t ym2 = __shfl_sync(0xffffffffu, ymx, 16);
                        uint32_t ym3 = __shfl_sync(0xffffffffu, ymx, 24);
                        if (!((uint32_t)threadIdx.x & 31u)
                            && ((yp ? live16b : live16)
                                & (1u << ((half << 3)
                                    + (((uint32_t)threadIdx.x & 255u) >> 5))))
                            && ((uint32_t)i << 4) + ((uint32_t)plane << 2)
                                + ((uint32_t)threadIdx.x >> 8)
                                < ((uint32_t)I >> 6)) {
                            SY[(uint64_t)gridDim.x * (uint32_t)NP
                                    * ((uint32_t)I >> 6)
                                + (uint64_t)blockIdx.x * (uint32_t)NP
                                    * ((uint32_t)I >> 6) * 2u
                                + (((uint64_t)(tw >> 5)
                                    * (((uint32_t)I >> 6) << 5)
                                  + (((uint32_t)i << 4)
                                    + ((uint32_t)plane << 2)
                                    + ((uint32_t)threadIdx.x >> 8)) * 32u
                                  + ((tw & 7u) << 2)
                                  + ((tw >> 3) & 3u)) << 1)]
                                = ym0 | (ym1 << 16);
                            SY[(uint64_t)gridDim.x * (uint32_t)NP
                                    * ((uint32_t)I >> 6)
                                + (uint64_t)blockIdx.x * (uint32_t)NP
                                    * ((uint32_t)I >> 6) * 2u
                                + (((uint64_t)(tw >> 5)
                                    * (((uint32_t)I >> 6) << 5)
                                  + (((uint32_t)i << 4)
                                    + ((uint32_t)plane << 2)
                                    + ((uint32_t)threadIdx.x >> 8)) * 32u
                                  + ((tw & 7u) << 2)
                                  + ((tw >> 3) & 3u)) << 1) + 1u]
                                = ym2 | (ym3 << 16);
                        }
                    }
                }
                __syncthreads();
            }
        }

        xgsinv2 += __popc(live16) + __popc(live16b);
        n16 = n16b;
    }

    for (int d = 16; d; d >>= 1)
        yrun = max(yrun, __shfl_down_sync(0xffffffffu, yrun, d));
    if (!((uint32_t)threadIdx.x & 31u))
        reinterpret_cast<uint32_t*>(smem + 16384u)[threadIdx.x >> 5] = yrun;
    __syncthreads();
    if ((uint32_t)threadIdx.x < 32u) {
        yrun = reinterpret_cast<uint32_t*>(smem + 16384u)[threadIdx.x];
        for (int d = 16; d; d >>= 1)
            yrun = max(yrun, __shfl_down_sync(0xffffffffu, yrun, d));
        if (!threadIdx.x) {
            uint32_t ybf = yrun ? __bfloat16_as_ushort(__float2bfloat16(
                __uint_as_float(yrun << 16) * (1.0f / 2688.0f))) : 0u;
            YGSINV[blockIdx.x] = ybf | (ybf << 16);
            reinterpret_cast<uint32_t*>(smem + 16384u)[0] = yrun;
        }
    }
    __syncthreads();
    yrun = reinterpret_cast<uint32_t*>(smem + 16384u)[0];
    float ygs = yrun
        ? 2688.0f / __uint_as_float(yrun << 16) : 0.0f;
    for (uint32_t syw = (uint32_t)threadIdx.x;
         syw < ((xgsinv2 + 31u) & ~31u) * ((uint32_t)I >> 6);
         syw += CTA) {
        uint32_t ym0 = SY[(uint64_t)gridDim.x * (uint32_t)NP
                            * ((uint32_t)I >> 6)
                          + (uint64_t)blockIdx.x * (uint32_t)NP
                            * ((uint32_t)I >> 6) * 2u
                          + ((uint64_t)syw << 1)];
        uint32_t ym1 = SY[(uint64_t)gridDim.x * (uint32_t)NP
                            * ((uint32_t)I >> 6)
                          + (uint64_t)blockIdx.x * (uint32_t)NP
                            * ((uint32_t)I >> 6) * 2u
                          + ((uint64_t)syw << 1) + 1u];
        uint32_t ys = moe_ue4m3_ceil(
            __uint_as_float((ym0 & 0xffffu) << 16) * ygs * (1.0f / 6.0f));
        ys |= moe_ue4m3_ceil(
            __uint_as_float(ym0 & 0xffff0000u) * ygs * (1.0f / 6.0f)) << 8;
        ys |= moe_ue4m3_ceil(
            __uint_as_float((ym1 & 0xffffu) << 16) * ygs * (1.0f / 6.0f)) << 16;
        ys |= moe_ue4m3_ceil(
            __uint_as_float(ym1 & 0xffff0000u) * ygs * (1.0f / 6.0f)) << 24;
        SY[(uint64_t)blockIdx.x * (uint32_t)NP
             * ((uint32_t)I >> 6) + syw] = ys;
    }
}
