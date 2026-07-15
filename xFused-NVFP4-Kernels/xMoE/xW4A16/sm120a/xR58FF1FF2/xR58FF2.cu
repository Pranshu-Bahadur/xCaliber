// xR57F2_direct_reduce_graph: H512 CTA1024 paired path with direct cp.reduce.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define F2_CTA 1024

/*
    Kernel contract

    runtime I = model intermediate reduction, runtime H = model hidden output

    W2[e,ktI,h512,t1024,v4]
      one plane = 16KB = F2_CTA x one lane-native m16n8k64 A fragment
      one plane = 16 warps x H16 = H256

    S2[e,ktI,h512,half2,q256]
      plane span = 1024 u32; half0/1 scale MMA rows 0..7/8..15

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
    post2      = W2GS[e] * YGSINV[e]

    Y[NP,H] BF16
      zero before launch; global Y is the linear FF2 K accumulator
*/
__global__ __launch_bounds__(F2_CTA, 1)
void xR57F2_direct_reduce_graph(
    const uint32_t* __restrict__ W2,
    const uint32_t* __restrict__ S2,
    const __nv_bfloat16* __restrict__ W2GS,
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
    uint32_t xbm0, xbm1, xbm2, xbm3;
    uint32_t a0, a1, a2, a3;
    uint32_t b00, b01, b10, b11;
    uint32_t scaleA, scaleB0, scaleB1, scaleB2, scaleB3;
    uint32_t out0, out1, post2, ygsinv;
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

    asm volatile("ldu.global.u16 %0, [%1];"
                 : "=h"(selector0)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       W2GS + (uint32_t)blockIdx.x))
                 : "memory");
    post2 = selector0;
    post2 |= post2 << 16;
    selector0 = 0;
    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(ygsinv)
                 : "l"((uint64_t)__cvta_generic_to_global(
                       YGSINV + (uint32_t)blockIdx.x))
                 : "memory");
    asm volatile(
        "{\n\t"
        ".reg .b32 zero;\n\t"
        "mov.b32 zero, 0;\n\t"
        "fma.rn.bf16x2 %0, %0, %1, zero;\n\t"
        "}"
        : "+r"(post2)
        : "r"(ygsinv));
    post = __uint_as_float((post2 & 0xffffu) << 16);

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
                                "ld.global.cs.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                : "l"(W2
                                    + (uint64_t)blockIdx.x
                                        * ((uint32_t)I >> 6)
                                        * ((uint32_t)H << 3)
                                    + (uint64_t)((ip << 5) + k)
                                        * ((uint32_t)H << 3)
                                    + ((uint64_t)h512 << 12)
                                    + ((uint64_t)threadIdx.x << 2))
                                : "memory");
                        }

                        scaleA = 0u;
                        if ((threadIdx.x & 3) < 2
                            && (h512 << 9)
                                + (((uint32_t)threadIdx.x >> 5) << 4)
                                + (((uint32_t)threadIdx.x & 31u) >> 2)
                                < (uint32_t)H) {
                            asm volatile(
                                "ld.global.cs.nc.b32 %0, [%1];"
                                : "=r"(scaleA)
                                : "l"(S2
                                    + (uint64_t)blockIdx.x
                                        * ((uint32_t)I >> 6)
                                        * ((uint32_t)H << 1)
                                    + (uint64_t)((ip << 5) + k)
                                        * ((uint32_t)H << 1)
                                    + ((uint64_t)h512 << 10)
                                    + ((uint64_t)(threadIdx.x & 3) << 8)
                                    + ((uint32_t)threadIdx.x >> 2))
                                : "memory");
                        }

                        // lane lp: v4 = {B0[2lp:2lp+1],B1[2lp:2lp+1]}.
                        b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                        if (xbm0 | xbm1) {
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
                        b00 &= xbm0; b01 &= xbm0;
                        b10 &= xbm1; b11 &= xbm1;
                        scaleB0 = 0u; scaleB1 = 0u;
                        scaleB2 = 0u; scaleB3 = 0u;
                        if (!((uint32_t)threadIdx.x & 3u)
                            && (xbm0 | xbm1 | xbm2 | xbm3)) {
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
                        scaleB0 &= xbm0; scaleB1 &= xbm1;
                        scaleB2 &= xbm2; scaleB3 &= xbm3;

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
                        if (xbm2 | xbm3) {
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
                        b00 &= xbm2; b01 &= xbm2;
                        b10 &= xbm3; b11 &= xbm3;

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
