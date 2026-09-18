
#define f322b(x) __float_as_uint(x)

template<bool BF16>
__device__ void softmax_v2bf16(
    uint32_t x
){ 
    if (ftz) {

        uint64_t tmp = ((uint64_t)x) << 16;
        asm volatile(
            "ex2.approx.ftz.bf16 %0, %1;\n\t"
            : "=f"()
            : "l"()
        );
    }
    else {
        float y = 1.4426950408889634f;
        asm volatile( //no bf16x2 -> f32x2 cvt...
            ".reg .b64 t, s;\n\t"
            ".reg .b32 a, b;\n\t"
            "and.b64 t, %1, 0x000000000000ffff;\n\t"
            "and.b64 s, %1, 0x00000000ffff0000;\n\t"
            "shl.b64 t, t, 16;\n\t"
            "shl.b64 s, s, 32;\n\t"
            "or.b64 t, t, s;\n\t"
            "mov.b64 {a, b}, t;\n\t"
            "fma.f32 a, a, %2, 0x0;\n\t"
            "ex2.approx.ftz.f32 a, a;\n\t"
            "ex2.approx.ftz.f32 b, b;\n\t"
            "cvt.rn.bf16x2.f32 %0, b, a;\n\t"
            : "=r"(x)
            : "r"(x), "f"(y)
        );
    }
}


template<bool V8>
__device__ void ldcg_b32v8(
    const uint32_t* src,
    uint32_t* dst
){
    if (V8) {
        asm volatile(
            "ld.aquire.global.gpu.cg.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n\t"
            : "=r"(dst[i]), "=r"(dst[i + 1]), "=r"(dst[i + 2]), "=r"(dst[i + 3]), 
              "=r"(dst[i + 4]), "=r"(dst[i + 5]), "=r"(dst[i + 6]), "=r"(dst[i + 7])
            : "l"((uint64_t)__cvta_generic_to_global(src))
        );
    }
    else {
        asm volatile(
            "ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
            "ld.global.cg.L2::128B.v4.b32 {%4, %5, %6, %7}, [%8 + 16];\n\t"
            : "=r"(dst[i]), "=r"(dst[i + 1]), "=r"(dst[i + 2]), "=r"(dst[i + 3]), 
              "=r"(dst[i + 4]), "=r"(dst[i + 5]), "=r"(dst[i + 6]), "=r"(dst[i + 7])
            : "l"((uint64_t)__cvta_generic_to_global(src))
        );
    }
}