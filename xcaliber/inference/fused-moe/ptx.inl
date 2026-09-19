#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define f322b(x) __float_as_uint(x)
#define u162bf16(x) __ushort_as_bfloat16(x)

__device__ __forceinline__ uint32_t softmax_bf16x2(
    uint32_t x
){
    float w = __uint_as_float(x << 16);
    float z = __uint_as_float(x & 0xffff'0000u);
    float y = 1.4426950408889634f;
    asm volatile(
        "mul.f32 %0, %0, %2;\n\t"
        "mul.f32 %1, %1, %2;\n\t"
        "ex2.approx.ftz.f32 %0, %0;\n\t"
        "ex2.approx.ftz.f32 %1, %1;\n\t"
        : "+&f"(w), "+&f"(z)
        : "f"(y)
    );
    __nv_bfloat162_raw tmp = __floats2bfloat162_rn(w, z);
    return uint32_t(tmp.x) | (uint32_t(tmp.y) << 16);
}

__device__ __forceinline__ void add_bf16x2(
    uint32_t& x, uint32_t y
){
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "add.bf16x2 %0, %0, %1;\n\t"
        : "+r"(x)
        : "r"(y)
    );
#else
    float w = __uint_as_float(x << 16) + __uint_as_float(y << 16);
    float z = __uint_as_float(x & 0xffff'0000u) + __uint_as_float(y & 0xffff'0000u);
    __nv_bfloat162_raw tmp = __floats2bfloat162_rn(w, z);
    x = uint32_t(tmp.x) | (uint32_t(tmp.y) << 16);
#endif
}

__device__ __forceinline__ void rcp_bf16x2(
    uint32_t& x
){
    float w = __uint_as_float(x << 16);
    float z = __uint_as_float(x & 0xffff'0000u);
    asm volatile(
        "rcp.approx.ftz.f32 %0, %0;\n\t"
        "rcp.approx.ftz.f32 %1, %1;\n\t"
        : "+&f"(w), "+&f"(z)
    );
    __nv_bfloat162_raw tmp = __floats2bfloat162_rn(w, z);
    x = uint32_t(tmp.x) | (uint32_t(tmp.y) << 16);
}

__device__ __forceinline__ void add_bf16x2x1(
    uint32_t x, uint32_t& y
){
    uint32_t tmp = x << 16;
    add_bf16x2(tmp, x & 0xffff'0000u);
    add_bf16x2(y, tmp & 0xffff'0000u);
}

__device__ __forceinline__ uint32_t softmax_mul(
    uint32_t x, uint32_t y
){
    uint32_t idx = x & 0x0000'ffffu;
    x &= 0xffff'0000u;
#if __CUDA_ARCH__ >= 900
    asm volatile(
        "mul.bf16x2 %0, %0, %1;\n\t"
        : "+r"(x)
        : "r"(y)
    );
#else
    float w = __uint_as_float(x) * __uint_as_float(y & 0xffff'0000u);
    x = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(w))) << 16;
#endif
    return (x & 0xffff'0000u) | idx;
}


__device__ __forceinline__ void ldcg_b32v4(
    const void* src,
    uint32_t* dst
){
    asm volatile(
#if __CUDA_ARCH__ >= 750
        "ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%4];\n\t"
#else
        "ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];\n\t"
#endif
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "l"((uint64_t)__cvta_generic_to_global(src))
        : "memory"
    );
}

__device__ __forceinline__ void ldcg_b32v8(
    const void* src,
    uint32_t* dst
){
#if __CUDA_ARCH__ >= 1000 && CUDART_VERSION >= 12090
    asm volatile(
        "ld.global.cg.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n\t"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
          "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "l"((uint64_t)__cvta_generic_to_global(src))
        : "memory"
    );
#else
    ldcg_b32v4(src, dst);
    ldcg_b32v4((const char*)src + 16, dst + 4);
#endif
}
