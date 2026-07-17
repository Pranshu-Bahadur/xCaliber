#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define XR64_ABS_CTA 1024
#define XR64_ABS_WARPS 32
#define XR64_ABS_PLANE 256
#define XR64_ABS_SMEM_BYTES 32768
#define XR64_ABS_ROUTE_CTA 256
#define XR64_ABS_TOPK 8

#define XR64_LD_CA_V4 "ld.global.ca.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
#define XR64_LD_CA_U32 "ld.global.ca.nc.u32 %0, [%1];"
#define XR64_LD_CA_F32 "ld.global.ca.nc.f32 %0, [%1];"
#define XR64_LD_CS_U32 "ld.global.cs.nc.u32 %0, [%1];"
#define XR64_LD_CS_F32 "ld.global.cs.nc.f32 %0, [%1];"

__device__ __forceinline__ void xr64_mma_m8n8k32_s4(
    int32_t& d0, int32_t& d1, uint32_t a, uint32_t b) {
    asm volatile(
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
        "{%0,%1}, {%2}, {%3}, {0,0};"
        : "=r"(d0), "=r"(d1)
        : "r"(a), "r"(b));
}

__device__ __forceinline__ uint32_t xr64_q4(float x, float inv) {
    int q = __float2int_rn(x * inv);
    q = max(-7, min(7, q));
    return (uint32_t)q & 15u;
}

__device__ __forceinline__ float xr64_bf16(uint16_t x) {
    return __uint_as_float((uint32_t)x << 16);
}

__device__ __forceinline__ uint16_t xr64_to_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

__device__ __forceinline__ void xr64_ld_ca_nc_v4(
    const void* p, uint32_t& x0, uint32_t& x1,
    uint32_t& x2, uint32_t& x3) {
    asm volatile(
        XR64_LD_CA_V4
        : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
        : "l"(p)
        : "memory");
}

__device__ __forceinline__ uint32_t xr64_ld_ca_nc_u32(const void* p) {
    uint32_t x;
    asm volatile(
        XR64_LD_CA_U32
        : "=r"(x) : "l"(p) : "memory");
    return x;
}

__device__ __forceinline__ float xr64_ld_ca_nc_f32(const void* p) {
    float x;
    asm volatile(
        XR64_LD_CA_F32
        : "=f"(x) : "l"(p) : "memory");
    return x;
}

__device__ __forceinline__ uint32_t xr64_ld_cs_nc_u32(const void* p) {
    uint32_t x;
    asm volatile(
        XR64_LD_CS_U32
        : "=r"(x) : "l"(p) : "memory");
    return x;
}

__device__ __forceinline__ uint32_t xr64_ld_cs_nc_pf_u32(const void* p) {
    uint32_t x;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 lp;\n\t"
        "mov.u32 lp, %%laneid;\n\t"
        "and.b32 lp, lp, 3;\n\t"
        "setp.eq.u32 p, lp, 0;\n\t"
        "@p prefetch.global.L2 [%1];\n\t"
        XR64_LD_CS_U32 "\n\t"
        "}"
        : "=r"(x) : "l"(p) : "memory");
    return x;
}

__device__ __forceinline__ float xr64_ld_cs_nc_f32(const void* p) {
    float x;
    asm volatile(
        XR64_LD_CS_F32
        : "=f"(x) : "l"(p) : "memory");
    return x;
}

__device__ __forceinline__ float xr64_silu(float x) {
    float d;
    asm volatile(
        "ex2.approx.ftz.f32 %0, %1;"
        : "=f"(d) : "f"(-x * 1.4426950408889634f));
    d += 1.0f;
    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(d));
    return x * d;
}

__device__ __forceinline__ float xr64_gelu_tanh(float x) {
    float d = x * x;
    d *= x;
    d = fmaf(d, 0.044715f, x);
    d *= 0.7978845608028654f;
    asm volatile("tanh.approx.f32 %0, %0;" : "+f"(d));
    return x * fmaf(d, 0.5f, 0.5f);
}

template <bool GELU_TANH>
__device__ __forceinline__ float xr64_gate(float x) {
    if constexpr (GELU_TANH) return xr64_gelu_tanh(x);
    return xr64_silu(x);
}

void launch_xr64_abs_preamble(
    const uint16_t* X,
    const int32_t* topk_idx,
    const int32_t* topk_off,
    uint32_t* X4,
    float* SX,
    int N,
    int NS,
    int H,
    int TOPK,
    cudaStream_t stream);

void launch_xr64_abs_topk(
    const uint16_t* logits,
    const uint16_t* expert_scale,
    int32_t* topk_idx,
    uint16_t* topk_weight,
    int N,
    int E,
    cudaStream_t stream);

void launch_xr64_abs_route_pack(
    const int32_t* topk_idx,
    const uint16_t* topk_weight,
    int32_t* topk_off,
    uint16_t* route_weight,
    int32_t* route_token,
    int32_t* count,
    int N,
    int NS,
    int E,
    cudaStream_t stream);

void launch_xr64_abs_ff1(
    const uint32_t* W13,
    const float* S13,
    const uint32_t* X4,
    const float* SX,
    const uint16_t* route_weight,
    const int32_t* count,
    uint32_t* Y4,
    float* SY,
    int E,
    int NS,
    int I,
    int H,
    bool gelu_tanh,
    cudaStream_t stream);

void launch_xr64_abs_ff2(
    const uint32_t* W2,
    const float* S2,
    const uint32_t* Y4,
    const float* SY,
    const int32_t* route_token,
    const int32_t* count,
    float* Y,
    int E,
    int NS,
    int I,
    int H,
    cudaStream_t stream);

void launch_xr64_abs_repack_gptq(
    const uint32_t* src,
    uint32_t* dst,
    int E,
    int K,
    int O,
    cudaStream_t stream);
