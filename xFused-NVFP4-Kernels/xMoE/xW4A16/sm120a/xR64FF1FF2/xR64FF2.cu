// xR64 FF2 CTA1024: conventional W2 -> xR64 rift register engine -> cp.reduce.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define F2_CTA 1024
#define F2_SMEM_BYTES (F2_CTA * 32u)

/*
    Kernel contract

    runtime I = model intermediate reduction, runtime H = model hidden output

    ModelOpt checkpoint tensors, stacked by the loader over expert e:
      W2   U8[e,H,I/2]       two E2M1 values per byte
      S2   E4M3[e,H,I/16]    one scale per K16
      W2GS F32[e]

    W2[e,H,I/2] -> MMA A, one warp / H16 / K64:
      CTA1024 = 32 warps x H16 = H512

      source A, grouped v4:
        source lane[4:0] = {H2,H1,H0,H3,K5}
        source reg [1:0] = {K4,K3}
        two adjacent lanes load one complete 32B H row

      destination B, PTX m16n8k64 A:
        destination lane[4:0] = {H2,H1,H0,K4,K3}
        destination reg [1:0] = {K5,H3}

      B^-1 A:
        swap source reg bit 0 / lane bit 0 with shfl.bfly xor 1
        swap source reg bit 1 / lane bit 1 with shfl.bfly xor 2
        bind temporary register order {0,2,1,3} -> {a0,a1,a2,a3}

      result:
        a0,a2 = W2[row g    ][words lp,lp+4]
        a1,a3 = W2[row g + 8][words lp,lp+4]

    S2 row-major K16 scales:
      selector thread-id-a=0
      lp0 supplies row g, lp1 supplies row g+8
      each u32 supplies K16 scales 0:3; source A == destination B

    Y4[e,n16,I64,q8,lp4,{n8_0x2,n8_1x2}]
      one lane-native v4 materializes both adjacent N8 B fragments

    expert_token_idx[e,0]        = live count
    expert_token_idx[e,1+packed] = original token
      reduction tail maps packed rows directly to Y[token,H]

    SY[e,n32,I64,q8,{n16a_0,n16a_1,n16b_0,n16b_1}]
      one lane-native v4 materializes all four MMA scale fragments

    routed N16 pairing
      header count -> adjacent packed N16 pairs, no sparse Xb scan
      load one W2/S2 packet -> up to four N8 MMAs
      accumulate every K64 slab in FP32 -> one final shared panel -> global Y

    isolate
      W2 checkpoint storage and S2 addressing remain exact xR64
      only the W2 register/thread packet bits cross B^-1 A
      Y4/SY consumption remains native xR64 rift

    output panel
      warp slab               [warp16][packed token32][H16 BF16]
      shared row              32B, one packed token x H16
      producer phase          p0:1 then p2:3; each st.shared bank is unique
      drain                   one 32B cp.reduce per live packed token
      dynamic SMEM            32 warps x 32 tokens x 32B = 32KB

    YGSINV[e] = BF16x2({YGSINV,YGSINV}) produced by FF1
    post2      = F32(W2GS[e]) * BF16(YGSINV[e].lo)

    Y[NP,H] BF16
      zero before launch; global Y is the linear FF2 K accumulator
*/
__global__ __launch_bounds__(F2_CTA, 1)
void xR64FF2(
    const uint8_t* __restrict__ W2,
    const uint8_t* __restrict__ S2,
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
    uint32_t xbm0, xbm1, xbm2, xbm3;
    uint32_t a0, a1, a2, a3;
    uint32_t b00, b01, b10, b11;
    uint32_t scaleA, scaleB0, scaleB1, scaleB2, scaleB3;
    uint32_t out0, ygsinv;
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
                 : "=f"(o1)
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
    post = o1 * o0;

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
                s00 = 0.0f; s01 = 0.0f; s02 = 0.0f; s03 = 0.0f;
                s10 = 0.0f; s11 = 0.0f; s12 = 0.0f; s13 = 0.0f;
                t00 = 0.0f; t01 = 0.0f; t02 = 0.0f; t03 = 0.0f;
                t10 = 0.0f; t11 = 0.0f; t12 = 0.0f; t13 = 0.0f;
                for (int ip = 0; ip < ((I + 2047) >> 11); ip++) {
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
                                ".reg .pred p0, p1;\n\t"
                                ".reg .b32 lane, x, y;\n\t"
                                "ld.global.cs.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n\t"
                                "bar.warp.sync 0xffffffff;\n\t"
                                "mov.u32 lane, %%laneid;\n\t"
                                "and.b32 x, lane, 1;\n\t"
                                "setp.ne.u32 p0, x, 0;\n\t"
                                "and.b32 y, lane, 2;\n\t"
                                "setp.ne.u32 p1, y, 0;\n\t"
                                "shfl.sync.bfly.b32 x, %0, 1, 0x1c03, 0xffffffff;\n\t"
                                "shfl.sync.bfly.b32 y, %1, 1, 0x1c03, 0xffffffff;\n\t"
                                "selp.b32 %0, y, %0, p0;\n\t"
                                "selp.b32 %1, %1, x, p0;\n\t"
                                "shfl.sync.bfly.b32 x, %2, 1, 0x1c03, 0xffffffff;\n\t"
                                "shfl.sync.bfly.b32 y, %3, 1, 0x1c03, 0xffffffff;\n\t"
                                "selp.b32 %2, y, %2, p0;\n\t"
                                "selp.b32 %3, %3, x, p0;\n\t"
                                "shfl.sync.bfly.b32 x, %0, 2, 0x1c03, 0xffffffff;\n\t"
                                "shfl.sync.bfly.b32 y, %2, 2, 0x1c03, 0xffffffff;\n\t"
                                "selp.b32 %0, y, %0, p1;\n\t"
                                "selp.b32 %2, %2, x, p1;\n\t"
                                "shfl.sync.bfly.b32 x, %1, 2, 0x1c03, 0xffffffff;\n\t"
                                "shfl.sync.bfly.b32 y, %3, 2, 0x1c03, 0xffffffff;\n\t"
                                "selp.b32 %1, y, %1, p1;\n\t"
                                "selp.b32 %3, %3, x, p1;\n\t"
                                "}"
                                : "=&r"(a0), "=&r"(a2),
                                  "=&r"(a1), "=&r"(a3)
                                : "l"(W2
                                    + ((uint64_t)blockIdx.x * (uint32_t)H
                                        + (uint32_t)(h512 << 9)
                                        + (((uint32_t)threadIdx.x >> 5) << 4)
                                        + (((uint32_t)threadIdx.x & 2u) << 2)
                                        + (((uint32_t)threadIdx.x & 28u) >> 2))
                                        * ((uint32_t)I >> 1)
                                    + (uint32_t)((ip << 5) + k) * 32u
                                    + (((uint32_t)threadIdx.x & 1u) << 4))
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
                                        * ((uint32_t)I >> 4)
                                    + (uint32_t)((ip << 5) + k) * 4u)
                                : "memory");
                        }

                        // xR64 FF1 output: paired N8 fragments arrive in one v4.
                        b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                        if (xbm0 | xbm1) {
                            asm volatile(
                                "ld.global.ca.nc.v4.u32 "
                                "{%0,%1,%2,%3}, [%4];"
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
                        // FF1 producer [r5 r4 | r3] -> MMA-B [r4 r3 | r5].
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p0, p1;\n\t"
                            ".reg .b32 lane, x, y;\n\t"
                            "mov.u32 lane, %%laneid;\n\t"
                            "and.b32 x, lane, 1;\n\t"
                            "setp.ne.u32 p0, x, 0;\n\t"
                            "and.b32 y, lane, 2;\n\t"
                            "setp.ne.u32 p1, y, 0;\n\t"
                            "shfl.sync.bfly.b32 x, %0, 1, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %1, 1, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %0, y, %0, p0;\n\t"
                            "selp.b32 %1, %1, x, p0;\n\t"
                            "shfl.sync.bfly.b32 x, %2, 1, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %3, 1, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %2, y, %2, p0;\n\t"
                            "selp.b32 %3, %3, x, p0;\n\t"
                            "shfl.sync.bfly.b32 x, %0, 2, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %1, 2, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %0, y, %0, p1;\n\t"
                            "selp.b32 %1, %1, x, p1;\n\t"
                            "shfl.sync.bfly.b32 x, %2, 2, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %3, 2, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %2, y, %2, p1;\n\t"
                            "selp.b32 %3, %3, x, p1;\n\t"
                            "}"
                            : "+r"(b00), "+r"(b01), "+r"(b10), "+r"(b11));
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

                        // n16b: same xR64 packet, masks select its two N8 rows.
                        b00 = 0u; b01 = 0u; b10 = 0u; b11 = 0u;
                        if (xbm2 | xbm3) {
                            asm volatile(
                                "ld.global.ca.nc.v4.u32 "
                                "{%0,%1,%2,%3}, [%4];"
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
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p0, p1;\n\t"
                            ".reg .b32 lane, x, y;\n\t"
                            "mov.u32 lane, %%laneid;\n\t"
                            "and.b32 x, lane, 1;\n\t"
                            "setp.ne.u32 p0, x, 0;\n\t"
                            "and.b32 y, lane, 2;\n\t"
                            "setp.ne.u32 p1, y, 0;\n\t"
                            "shfl.sync.bfly.b32 x, %0, 1, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %1, 1, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %0, y, %0, p0;\n\t"
                            "selp.b32 %1, %1, x, p0;\n\t"
                            "shfl.sync.bfly.b32 x, %2, 1, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %3, 1, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %2, y, %2, p0;\n\t"
                            "selp.b32 %3, %3, x, p0;\n\t"
                            "shfl.sync.bfly.b32 x, %0, 2, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %1, 2, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %0, y, %0, p1;\n\t"
                            "selp.b32 %1, %1, x, p1;\n\t"
                            "shfl.sync.bfly.b32 x, %2, 2, 0x1c03, 0xffffffff;\n\t"
                            "shfl.sync.bfly.b32 y, %3, 2, 0x1c03, 0xffffffff;\n\t"
                            "selp.b32 %2, y, %2, p1;\n\t"
                            "selp.b32 %3, %3, x, p1;\n\t"
                            "}"
                            : "+r"(b00), "+r"(b01), "+r"(b10), "+r"(b11));
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
                }

                    // One warp owns 32 packed tokens x H16.  Each phase has
                    // eight u32 stores and maps one writer to one bank.
                    out0 = (uint32_t)__cvta_generic_to_shared(
                        smem + ((uint32_t)threadIdx.x >> 5) * 1024u);

                    if (live16 & 0xffu) {
                        o0 = s00 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a0) : "f"(o1), "f"(o0));
                        o0 = s01 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a1) : "f"(o1), "f"(o0));
                        o0 = s02 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a2) : "f"(o1), "f"(o0));
                        o0 = s03 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a3) : "f"(o1), "f"(o0));
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p0, p1;\n\t"
                            ".reg .b32 lane, off;\n\t"
                            "mov.u32 lane, %%tid.x;\n\t"
                            "and.b32 off, lane, 6;\n\t"
                            "setp.eq.u32 p0, off, 0;\n\t"
                            "setp.eq.u32 p1, off, 2;\n\t"
                            "bfe.u32 off, lane, 0, 2;\n\t"
                            "shl.b32 off, off, 6;\n\t"
                            "and.b32 lane, lane, 28;\n\t"
                            "shr.u32 lane, lane, 1;\n\t"
                            "add.u32 off, off, lane;\n\t"
                            "add.u32 off, off, %0;\n\t"
                            "@p0 st.shared.u32 [off], %1;\n\t"
                            "@p0 st.shared.u32 [off + 32], %2;\n\t"
                            "@p0 st.shared.u32 [off + 16], %3;\n\t"
                            "@p0 st.shared.u32 [off + 48], %4;\n\t"
                            "@p1 st.shared.u32 [off], %1;\n\t"
                            "@p1 st.shared.u32 [off + 32], %2;\n\t"
                            "@p1 st.shared.u32 [off + 16], %3;\n\t"
                            "@p1 st.shared.u32 [off + 48], %4;\n\t"
                            "}"
                            :
                            : "r"(out0), "r"(a0), "r"(a1),
                              "r"(a2), "r"(a3)
                            : "memory");
                    }

                    if (live16 & 0xff00u) {
                        o0 = s10 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a0) : "f"(o1), "f"(o0));
                        o0 = s11 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a1) : "f"(o1), "f"(o0));
                        o0 = s12 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a2) : "f"(o1), "f"(o0));
                        o0 = s13 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a3) : "f"(o1), "f"(o0));
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p0, p1;\n\t"
                            ".reg .b32 lane, off;\n\t"
                            "mov.u32 lane, %%tid.x;\n\t"
                            "and.b32 off, lane, 6;\n\t"
                            "setp.eq.u32 p0, off, 0;\n\t"
                            "setp.eq.u32 p1, off, 2;\n\t"
                            "bfe.u32 off, lane, 0, 2;\n\t"
                            "shl.b32 off, off, 6;\n\t"
                            "and.b32 lane, lane, 28;\n\t"
                            "shr.u32 lane, lane, 1;\n\t"
                            "add.u32 off, off, lane;\n\t"
                            "add.u32 off, off, %0;\n\t"
                            "@p0 st.shared.u32 [off + 256], %1;\n\t"
                            "@p0 st.shared.u32 [off + 288], %2;\n\t"
                            "@p0 st.shared.u32 [off + 272], %3;\n\t"
                            "@p0 st.shared.u32 [off + 304], %4;\n\t"
                            "@p1 st.shared.u32 [off + 256], %1;\n\t"
                            "@p1 st.shared.u32 [off + 288], %2;\n\t"
                            "@p1 st.shared.u32 [off + 272], %3;\n\t"
                            "@p1 st.shared.u32 [off + 304], %4;\n\t"
                            "}"
                            :
                            : "r"(out0), "r"(a0), "r"(a1),
                              "r"(a2), "r"(a3)
                            : "memory");
                    }

                    if (live16b & 0xffu) {
                        o0 = t00 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a0) : "f"(o1), "f"(o0));
                        o0 = t01 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a1) : "f"(o1), "f"(o0));
                        o0 = t02 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a2) : "f"(o1), "f"(o0));
                        o0 = t03 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a3) : "f"(o1), "f"(o0));
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p0, p1;\n\t"
                            ".reg .b32 lane, off;\n\t"
                            "mov.u32 lane, %%tid.x;\n\t"
                            "and.b32 off, lane, 6;\n\t"
                            "setp.eq.u32 p0, off, 0;\n\t"
                            "setp.eq.u32 p1, off, 2;\n\t"
                            "bfe.u32 off, lane, 0, 2;\n\t"
                            "shl.b32 off, off, 6;\n\t"
                            "and.b32 lane, lane, 28;\n\t"
                            "shr.u32 lane, lane, 1;\n\t"
                            "add.u32 off, off, lane;\n\t"
                            "add.u32 off, off, %0;\n\t"
                            "@p0 st.shared.u32 [off + 512], %1;\n\t"
                            "@p0 st.shared.u32 [off + 544], %2;\n\t"
                            "@p0 st.shared.u32 [off + 528], %3;\n\t"
                            "@p0 st.shared.u32 [off + 560], %4;\n\t"
                            "@p1 st.shared.u32 [off + 512], %1;\n\t"
                            "@p1 st.shared.u32 [off + 544], %2;\n\t"
                            "@p1 st.shared.u32 [off + 528], %3;\n\t"
                            "@p1 st.shared.u32 [off + 560], %4;\n\t"
                            "}"
                            :
                            : "r"(out0), "r"(a0), "r"(a1),
                              "r"(a2), "r"(a3)
                            : "memory");
                    }

                    if (live16b & 0xff00u) {
                        o0 = t10 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a0) : "f"(o1), "f"(o0));
                        o0 = t11 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a1) : "f"(o1), "f"(o0));
                        o0 = t12 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a2) : "f"(o1), "f"(o0));
                        o0 = t13 * post;
                        o1 = __shfl_xor_sync(0xffffffffu, o0, 4);
                        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
                                     : "=r"(a3) : "f"(o1), "f"(o0));
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p0, p1;\n\t"
                            ".reg .b32 lane, off;\n\t"
                            "mov.u32 lane, %%tid.x;\n\t"
                            "and.b32 off, lane, 6;\n\t"
                            "setp.eq.u32 p0, off, 0;\n\t"
                            "setp.eq.u32 p1, off, 2;\n\t"
                            "bfe.u32 off, lane, 0, 2;\n\t"
                            "shl.b32 off, off, 6;\n\t"
                            "and.b32 lane, lane, 28;\n\t"
                            "shr.u32 lane, lane, 1;\n\t"
                            "add.u32 off, off, lane;\n\t"
                            "add.u32 off, off, %0;\n\t"
                            "@p0 st.shared.u32 [off + 768], %1;\n\t"
                            "@p0 st.shared.u32 [off + 800], %2;\n\t"
                            "@p0 st.shared.u32 [off + 784], %3;\n\t"
                            "@p0 st.shared.u32 [off + 816], %4;\n\t"
                            "@p1 st.shared.u32 [off + 768], %1;\n\t"
                            "@p1 st.shared.u32 [off + 800], %2;\n\t"
                            "@p1 st.shared.u32 [off + 784], %3;\n\t"
                            "@p1 st.shared.u32 [off + 816], %4;\n\t"
                            "}"
                            :
                            : "r"(out0), "r"(a0), "r"(a1),
                              "r"(a2), "r"(a3)
                            : "memory");
                    }

                    asm volatile(
                        "bar.warp.sync 0xffffffff;\n\t"
                        "fence.proxy.async.shared::cta;"
                        ::: "memory");

                    if ((uint32_t)(n16 << 4)
                            + ((uint32_t)threadIdx.x & 31u) < count
                        && (uint32_t)(h512 << 9)
                            + (((uint32_t)threadIdx.x >> 5) << 4) + 15u
                            < (uint32_t)H) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 token;\n\t"
                            "ldu.global.u16 token, [%1];\n\t"
                            "cvt.u32.u16 %0, token;\n\t"
                            "}"
                            : "=r"(out0)
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  expert_token_idx
                                  + (uint64_t)blockIdx.x
                                      * ((uint32_t)NP + 1u) + 1u
                                  + (uint32_t)(n16 << 4)
                                  + ((uint32_t)threadIdx.x & 31u)))
                            : "memory");
                        asm volatile(
                            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
                            "[%0], [%1], 32;\n\t"
                            "cp.async.bulk.commit_group;"
                            :
                            : "l"((uint64_t)__cvta_generic_to_global(
                                  Y + (uint64_t)out0 * (uint32_t)H
                                  + (uint32_t)(h512 << 9)
                                  + (((uint32_t)threadIdx.x >> 5) << 4))),
                              "r"((uint32_t)__cvta_generic_to_shared(
                                  smem
                                  + ((uint32_t)threadIdx.x >> 5) * 1024u
                                  + ((uint32_t)threadIdx.x & 31u) * 32u))
                            : "memory");
                    }

                    asm volatile(
                        "cp.async.bulk.wait_group.read 0;\n\t"
                        "bar.warp.sync 0xffffffff;"
                        ::: "memory");
                }
        n16 = n16b;
    }
}
