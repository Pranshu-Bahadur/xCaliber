// xR57F2_modelopt_direct_reduce_graph: conventional NVFP4 -> direct cp.reduce.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define F2_CTA 1024

/*
    Kernel contract

    runtime I = model intermediate reduction, runtime H = model hidden output

    ModelOpt checkpoint tensors, stacked by the loader over expert e:
      W2   U8[e,H,I/2]       two E2M1 values per byte
      S2   E4M3[e,H,I/16]    one scale per K16
      W2GS F32[e]

    register panel, one warp / K64:
      a0 = W2[row g    ][lp*8 +  0.. 7]
      a1 = W2[row g + 8][lp*8 +  0.. 7]
      a2 = W2[row g    ][lp*8 + 32..39]
      a3 = W2[row g + 8][lp*8 + 32..39]
      t0..t3 form one contiguous 16B row half; t4..t7 form the next row
      lp0 in each row block prefetches its aligned 32B fragment to L2
      coherent .cs scalar loads preserve the native MMA fragment directly

    S2 row-major K16 scales:
      lp0 supplies row g, lp1 supplies row g+8

    Y4[e,n16,kt,q8,lp4,{n8_0x2,n8_1x2}]
      one lane-native v4 materializes both N8 B fragments

    expert_token_idx[e,0]        = live count
    expert_token_idx[e,1+packed] = original token
      reduction tail maps packed rows directly to Y[token,H]

    SY[e,n32,kt,q8,{n16a_0,n16a_1,n16b_0,n16b_1}]
      one lane-native v4 materializes all four MMA scale fragments

    routed N16 pairing
      header count -> adjacent packed N16 pairs, no sparse Xb scan
      load one W2/S2 packet -> up to four N8 MMAs
      accumulate up to 32 K64 slabs -> reduce one K2048 panel into global Y

    sync
      output / four lanes     no-sync shared -> two 16B cp.reduce issuers
      dynamic SMEM            4 x F2_CTA*8B async reduction issue slots

    YGSINV[e] = BF16x2({YGSINV,YGSINV}) produced by FF1
    post2      = F32(W2GS[e]) * BF16(YGSINV[e].lo)

    Y[NP,H] BF16
      zero before launch; global Y is the linear FF2 K accumulator
*/
__global__ __launch_bounds__(F2_CTA, 1)
void xR57F2_modelopt_direct_reduce_graph(
    const uint32_t* __restrict__ W2,
    const uint32_t* __restrict__ S2,
    const float* __restrict__ W2GS,
    const uint32_t* __restrict__ Y4,
    const uint32_t* __restrict__ SY,
    const uint32_t* __restrict__ YGSINV,
    const uint16_t* __restrict__ expert_token_idx,
    __nv_bfloat16* __restrict__ Y,
    int NP,
    int I,
    int H
) {
    extern __shared__ __align__(16) unsigned char smem[];

    uint32_t count, live16, live16b;
    uint32_t a0, a1, a2, a3;
    uint32_t b00, b01, b10, b11;
    uint32_t scaleA, scaleB0, scaleB1, scaleB2, scaleB3;
    uint32_t out0, out1, ygsinv;
    uint16_t selector0;
    float s00, s01, s02, s03;
    float s10, s11, s12, s13;
    float t00, t01, t02, t03;
    float t10, t11, t12, t13;
    float post, o0, o1;
    int n16b;

    asm volatile("ldu.global.u16 %0, [%1];"
                 : "=h"(selector0)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       expert_token_idx
                       + (uint64_t)blockIdx.x * ((uint32_t)NP + 1u)))
                 : "memory");
    count = selector0;
    if (!count) return;

    asm volatile("ldu.global.f32 %0, [%1];"
                 : "=f"(post)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       W2GS + (uint32_t)blockIdx.x))
                 : "memory");
    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(ygsinv)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       YGSINV + (uint32_t)blockIdx.x))
                 : "memory");
    selector0 = (uint16_t)ygsinv;
    asm volatile("cvt.f32.bf16 %0, %1;"
                 : "=f"(o0)
                 : "h"(selector0));
    selector0 = 0;
    post *= o0;

    for (int n16 = 0; (uint32_t)(n16 << 4) < count; n16++) {
        n16b = n16 + 1;
        live16 = count - (uint32_t)(n16 << 4);
        live16 = live16 >= 16u ? 0xffffu : (1u << live16) - 1u;
        live16b = count > (uint32_t)(n16b << 4)
            ? count - (uint32_t)(n16b << 4) : 0u;
        live16b = live16b >= 16u ? 0xffffu
            : (live16b ? (1u << live16b) - 1u : 0u);

        for (int h512 = 0; h512 < ((H + 511) >> 9); h512++) {
                for (int ip = 0; ip < ((I + 2047) >> 11); ip++) {
                    s00 = 0.0f; s01 = 0.0f; s02 = 0.0f; s03 = 0.0f;
                    s10 = 0.0f; s11 = 0.0f; s12 = 0.0f; s13 = 0.0f;
                    t00 = 0.0f; t01 = 0.0f; t02 = 0.0f; t03 = 0.0f;
                    t10 = 0.0f; t11 = 0.0f; t12 = 0.0f; t13 = 0.0f;
                    for (int k = 0;
                         k < 32 && ((ip << 5) + k) < (I >> 6);
                         k++) {
                        a0 = 0u; a1 = 0u; a2 = 0u; a3 = 0u;
                        if ((h512 << 9)
                            + (((uint32_t)threadIdx.x >> 5) << 4)
                            + (((uint32_t)threadIdx.x & 31u) >> 2)
                            < (uint32_t)H) {
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
                                : "l"(W2
                                    + ((uint64_t)blockIdx.x * (uint32_t)H
                                        + (uint32_t)(h512 << 9)
                                        + (((uint32_t)threadIdx.x >> 5) << 4)
                                        + (((uint32_t)threadIdx.x & 31u) >> 2))
                                        * ((uint32_t)I >> 3)
                                    + (uint32_t)((ip << 5) + k) * 8u
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
                                : "l"(W2
                                    + ((uint64_t)blockIdx.x * (uint32_t)H
                                        + (uint32_t)(h512 << 9)
                                        + (((uint32_t)threadIdx.x >> 5) << 4)
                                        + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        + 8u)
                                        * ((uint32_t)I >> 3)
                                    + (uint32_t)((ip << 5) + k) * 8u
                                    + ((uint32_t)threadIdx.x & 3u))
                                : "memory");
                        }

                        scaleA = 0u;
                        if ((threadIdx.x & 3) < 2
                            && (h512 << 9)
                                + (((uint32_t)threadIdx.x >> 5) << 4)
                                + (((uint32_t)threadIdx.x & 31u) >> 2)
                                < (uint32_t)H) {
                            asm volatile(
                                "ld.global.cs.b32 %0, [%1];"
                                : "=r"(scaleA)
                                : "l"(S2
                                    + ((uint64_t)blockIdx.x * (uint32_t)H
                                        + (uint32_t)(h512 << 9)
                                        + (((uint32_t)threadIdx.x >> 5) << 4)
                                        + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        + ((uint32_t)(threadIdx.x & 3) << 3))
                                        * ((uint32_t)I >> 6)
                                    + (uint32_t)((ip << 5) + k))
                                : "memory");
                        }

                        // lane lp: v4 = {B0[2lp:2lp+1],B1[2lp:2lp+1]}.
                        b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                        if ((uint32_t)(n16 << 4)
                            + (((uint32_t)threadIdx.x & 31u) >> 2) < count) {
                            asm volatile(
                                "ld.acquire.gpu.global.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(b00), "=r"(b01),
                                  "=r"(b10), "=r"(b11)
                                : "l"(Y4
                                    + (uint64_t)blockIdx.x * (uint32_t)NP
                                        * ((uint32_t)I >> 3)
                                    + (uint64_t)n16 * ((uint32_t)I << 1)
                                    + ((uint64_t)((ip << 5) + k) << 7)
                                    + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        * 16u
                                    + (((uint32_t)threadIdx.x & 3u) << 2))
                                : "memory");
                        }
                        if ((uint32_t)(n16 << 4) + 8u
                            + (((uint32_t)threadIdx.x & 31u) >> 2) >= count)
                            b10 = b11 = 0u;
                        scaleB0 = 0u; scaleB1 = 0u;
                        scaleB2 = 0u; scaleB3 = 0u;
                        if (!((uint32_t)threadIdx.x & 3u)
                            && (uint32_t)(n16 << 4)
                                + (((uint32_t)threadIdx.x & 31u) >> 2)
                                < count) {
                            asm volatile(
                                "ld.acquire.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(scaleB0), "=r"(scaleB1),
                                  "=r"(scaleB2), "=r"(scaleB3)
                                : "l"(SY
                                    + (uint64_t)blockIdx.x * (uint32_t)NP
                                        * ((uint32_t)I >> 6)
                                    + (uint64_t)(n16 >> 1)
                                        * (((uint32_t)I >> 6) << 5)
                                    + (uint32_t)((ip << 5) + k) * 32u
                                    + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        * 4u)
                                : "memory");
                        }
                        if ((uint32_t)(n16 << 4) + 8u
                            + (((uint32_t)threadIdx.x & 31u) >> 2) >= count)
                            scaleB1 = 0u;
                        if ((uint32_t)(n16b << 4)
                            + (((uint32_t)threadIdx.x & 31u) >> 2) >= count)
                            scaleB2 = 0u;
                        if ((uint32_t)(n16b << 4) + 8u
                            + (((uint32_t)threadIdx.x & 31u) >> 2) >= count)
                            scaleB3 = 0u;

                        if (live16 & 0xffu) {
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};"
                                : "+f"(s00), "+f"(s01), "+f"(s02), "+f"(s03)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b00), "r"(b01), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB0),
                                  "h"(selector0), "h"(selector0));
                        }

                        if (live16 & 0xff00u) {
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};"
                                : "+f"(s10), "+f"(s11), "+f"(s12), "+f"(s13)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b10), "r"(b11), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB1),
                                  "h"(selector0), "h"(selector0));
                        }

                        // Same paired transaction for adjacent packed n16b.
                        b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                        if ((uint32_t)(n16b << 4)
                            + (((uint32_t)threadIdx.x & 31u) >> 2) < count) {
                            asm volatile(
                                "ld.acquire.gpu.global.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(b00), "=r"(b01),
                                  "=r"(b10), "=r"(b11)
                                : "l"(Y4
                                    + (uint64_t)blockIdx.x * (uint32_t)NP
                                        * ((uint32_t)I >> 3)
                                    + (uint64_t)n16b * ((uint32_t)I << 1)
                                    + ((uint64_t)((ip << 5) + k) << 7)
                                    + (((uint32_t)threadIdx.x & 31u) >> 2)
                                        * 16u
                                    + (((uint32_t)threadIdx.x & 3u) << 2))
                                : "memory");
                        }
                        if ((uint32_t)(n16b << 4) + 8u
                            + (((uint32_t)threadIdx.x & 31u) >> 2) >= count)
                            b10 = b11 = 0u;

                        if (live16b & 0xffu) {
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};"
                                : "+f"(t00), "+f"(t01), "+f"(t02), "+f"(t03)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b00), "r"(b01), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB2),
                                  "h"(selector0), "h"(selector0));
                        }

                        if (live16b & 0xff00u) {
                            asm volatile(
                                "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                                "{%0,%1,%2,%3}, %10, {%11,%12}, %13, {%14,%15};"
                                : "+f"(t10), "+f"(t11), "+f"(t12), "+f"(t13)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b10), "r"(b11), "r"(scaleA),
                                  "h"(selector0), "h"(selector0), "r"(scaleB3),
                                  "h"(selector0), "h"(selector0));
                        }

                    }

                    if (live16 & 0xffu) {
                        o0 = s00 * post; o1 = s01 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out0) : "f"(o1), "f"(o0));
                        o0 = s02 * post; o1 = s03 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out1) : "f"(o1), "f"(o0));
                        asm volatile(
                            "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 "
                            "[%0], {%1,%2};"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4))),
                              "r"(out0), "r"(out1)
                            : "memory");
                    }

                    if (live16 & 0xff00u) {
                        o0 = s10 * post; o1 = s11 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out0) : "f"(o1), "f"(o0));
                        o0 = s12 * post; o1 = s13 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out1) : "f"(o1), "f"(o0));
                        asm volatile(
                            "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 "
                            "[%0], {%1,%2};"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + F2_CTA * 8u
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4))),
                              "r"(out0), "r"(out1)
                            : "memory");
                    }

                    if (live16b & 0xffu) {
                        o0 = t00 * post; o1 = t01 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out0) : "f"(o1), "f"(o0));
                        o0 = t02 * post; o1 = t03 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out1) : "f"(o1), "f"(o0));
                        asm volatile(
                            "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 "
                            "[%0], {%1,%2};"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + F2_CTA * 16u
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4))),
                              "r"(out0), "r"(out1)
                            : "memory");
                    }

                    if (live16b & 0xff00u) {
                        o0 = t10 * post; o1 = t11 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out0) : "f"(o1), "f"(o0));
                        o0 = t12 * post; o1 = t13 * post;
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(out1) : "f"(o1), "f"(o0));
                        asm volatile(
                            "stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 "
                            "[%0], {%1,%2};"
                            :
                            : "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + F2_CTA * 24u
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4))),
                              "r"(out0), "r"(out1)
                            : "memory");
                    }

                    asm volatile(
                        "fence.proxy.async.shared::cta;" ::: "memory");

                    if (((uint32_t)threadIdx.x & 31u) < 16u
                        && (uint32_t)(n16 << 4)
                            + ((uint32_t)threadIdx.x & 7u) < count
                        && (uint32_t)(h512 << 9)
                            + (((uint32_t)threadIdx.x >> 5) << 4)
                            + ((uint32_t)threadIdx.x & 8u) < (uint32_t)H) {
                        out0 = expert_token_idx[
                            (uint64_t)blockIdx.x * ((uint32_t)NP + 1u) + 1u
                            + (uint32_t)(n16 << 4)
                            + ((uint32_t)threadIdx.x & 7u)];
                        asm volatile(
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Y + (uint64_t)out0 * (uint32_t)H
                                  + (uint32_t)(h512 << 9)
                                  + (((uint32_t)threadIdx.x >> 5) << 4)
                                  + ((uint32_t)threadIdx.x & 8u))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4)))
                            : "memory");
                    }

                    if (((uint32_t)threadIdx.x & 31u) < 16u
                        && (uint32_t)(n16 << 4) + 8u
                            + ((uint32_t)threadIdx.x & 7u) < count
                        && (uint32_t)(h512 << 9)
                            + (((uint32_t)threadIdx.x >> 5) << 4)
                            + ((uint32_t)threadIdx.x & 8u) < (uint32_t)H) {
                        out0 = expert_token_idx[
                            (uint64_t)blockIdx.x * ((uint32_t)NP + 1u) + 1u
                            + (uint32_t)(n16 << 4) + 8u
                            + ((uint32_t)threadIdx.x & 7u)];
                        asm volatile(
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Y + (uint64_t)out0 * (uint32_t)H
                                  + (uint32_t)(h512 << 9)
                                  + (((uint32_t)threadIdx.x >> 5) << 4)
                                  + ((uint32_t)threadIdx.x & 8u))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + F2_CTA * 8u
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4)))
                            : "memory");
                    }

                    if (((uint32_t)threadIdx.x & 31u) < 16u
                        && (uint32_t)(n16b << 4)
                            + ((uint32_t)threadIdx.x & 7u) < count
                        && (uint32_t)(h512 << 9)
                            + (((uint32_t)threadIdx.x >> 5) << 4)
                            + ((uint32_t)threadIdx.x & 8u) < (uint32_t)H) {
                        out0 = expert_token_idx[
                            (uint64_t)blockIdx.x * ((uint32_t)NP + 1u) + 1u
                            + (uint32_t)(n16b << 4)
                            + ((uint32_t)threadIdx.x & 7u)];
                        asm volatile(
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Y + (uint64_t)out0 * (uint32_t)H
                                  + (uint32_t)(h512 << 9)
                                  + (((uint32_t)threadIdx.x >> 5) << 4)
                                  + ((uint32_t)threadIdx.x & 8u))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + F2_CTA * 16u
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4)))
                            : "memory");
                    }

                    if (((uint32_t)threadIdx.x & 31u) < 16u
                        && (uint32_t)(n16b << 4) + 8u
                            + ((uint32_t)threadIdx.x & 7u) < count
                        && (uint32_t)(h512 << 9)
                            + (((uint32_t)threadIdx.x >> 5) << 4)
                            + ((uint32_t)threadIdx.x & 8u) < (uint32_t)H) {
                        out0 = expert_token_idx[
                            (uint64_t)blockIdx.x * ((uint32_t)NP + 1u) + 1u
                            + (uint32_t)(n16b << 4) + 8u
                            + ((uint32_t)threadIdx.x & 7u)];
                        asm volatile(
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 16;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Y + (uint64_t)out0 * (uint32_t)H
                                  + (uint32_t)(h512 << 9)
                                  + (((uint32_t)threadIdx.x >> 5) << 4)
                                  + ((uint32_t)threadIdx.x & 8u))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem + F2_CTA * 24u
                                  + ((uint32_t)threadIdx.x >> 5) * 256u
                                  + (((uint32_t)threadIdx.x & 7u) << 4)
                                  + (((uint32_t)threadIdx.x & 8u) << 4)))
                            : "memory");
                    }

                    asm volatile(
                        "cp.async.bulk.wait_group.read 0;" ::: "memory");
                    }
                }
        n16 = n16b;
    }
}
