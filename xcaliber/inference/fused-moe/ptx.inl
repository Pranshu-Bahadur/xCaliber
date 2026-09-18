//imports
#define f322b(x) __float_as_uint(x)

template<bool SM90P>
__device__ void softmax_bf16x2(
    uint32_t x
){ 
    float y = 1.4426950408889634f;
    asm volatile(
        ".reg .b32 w, x;\n\t"
        ".reg .b16 a, b;\n\t"
        "mov.b16 {a, b}, %1;\n\t"
        "mov.b32 w, {a, _};\n\t"
        "and.b32 %1, w, 0x00001000;\n\t"
        "and.b32 w, w, 0x00007fff;\n\t"
        "shl.b32 %1, %1, 16;\n\t"
        "or.b32  w, w, %1;\n\t"
        "mov.b32 x, {b, _};\n\t"
        "and.b32 %1, x, 0x00001000;\n\t"
        "and.b32 x, x, 0x00007fff;\n\t"
        "shl.b32 %1, %1, 16;\n\t"
        "or.b32  x, x, %1;\n\t"
        "mul.f32 w, w, %2;\n\t"
        "mul.f32 x, x, %2;\n\t"
        "ex2.approx.ftz.f32 w, w;\n\t"
        "ex2.approx.ftz.f32 x, x;\n\t"
        "cvt.rn.bf16x2.f32 %0, x, w;\n\t"
        : "=r"(x)
        : "r"(x), "f"(y)
    );
}

template<bool SM90P>
__device__ void add_bf16x2(
    uint32_t x, uint32_t y
){
    if (SM90P) {
        asm volatile(
            "add.bf16x2 %0, %0, %1;\n\t"
            : "=r"(x)
            : "r"(x), "f"(y)
        );
    }
    else {
        asm volatile(
            ".reg .b32 w, x, y, z;\n\t"
            ".reg .b16 a, b, c, d;\n\t"
            "mov.b16 {a, b}, %1;\n\t"
            "mov.b16 {c, d}, %2;\n\t"
            "mov.b32 w, {a, _};\n\t"
            "and.b32 %1, w, 0x00001000;\n\t"
            "and.b32 w, w, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  w, w, %1;\n\t"
            "mov.b32 x, {b, _};\n\t"
            "and.b32 %1, x, 0x00001000;\n\t"
            "and.b32 x, x, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  x, x, %1;\n\t"
            "mov.b32 y, {c, _};\n\t"
            "and.b32 %1, y, 0x00001000;\n\t"
            "and.b32 y, y, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  y, y, %1;\n\t"
            "mov.b32 z, {d, _};\n\t"
            "and.b32 %1, z, 0x00001000;\n\t"
            "and.b32 z, z, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  z, z, %1;\n\t"
            "add.f32 w, w, y;\n\t"
            "add.f32 x, x, z;\n\t"
            "cvt.rn.bf16x2.f32 %0, x, w;\n\t"
            : "=r"(x)
            : "r"(x), "f"(y)
        );
    }
}

__device__ void rcp_bf16x2(
    uint32_t x
){
    asm volatile(
            ".reg .b32 w, x;\n\t"
            ".reg .b16 a, b;\n\t"
            "mov.b16 {a, b}, %1;\n\t"
            "mov.b32 w, {a, _};\n\t"
            "and.b32 %1, w, 0x00001000;\n\t"
            "and.b32 w, w, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  w, w, %1;\n\t"
            "mov.b32 x, {b, _};\n\t"
            "and.b32 %1, x, 0x00001000;\n\t"
            "and.b32 x, x, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  x, x, %1;\n\t"
            "rcp.approx.ftz.f32 w, w;\n\t"
            "rcp.approx.ftz.f32 x, x;\n\t"
            "cvt.rn.bf16x2.f32 %0, x, w;\n\t"
            : "=r"(x)
            : "r"(x), "f"(y)
    );
}

template<bool SM90P>
__device__ void add_bf16x2x1(
    uint32_t x, uint32_t y
){
    if (SM90P) {
        asm volatile(
            ".reg .b16 a, b;\n\t"
            "add.bf16x2 %1, %2, %1;\n\t"
            "mov.b32 {a, b}, t;\n\t"
            "add.bf16 t, a, b;\n\t"
            "mov.b32 %0, {t, _};\n\t"
            : "=r"(y)
            : "r"(x), "f"(y)
        );
    }
    else {
        asm volatile(
            ".reg .b32 w, x;\n\t"
            ".reg .b16 a, b;\n\t"
            "mov.b16 {a, b}, %1;\n\t"
            "mov.b32 w, {a, _};\n\t"
            "and.b32 %1, w, 0x00001000;\n\t"
            "and.b32 w, w, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  w, w, %1;\n\t"
            "mov.b32 x, {b, _};\n\t"
            "and.b32 %1, x, 0x00001000;\n\t"
            "and.b32 x, x, 0x00007fff;\n\t"
            "shl.b32 %1, %1, 16;\n\t"
            "or.b32  x, x, %1;\n\t"
            "add.f32 x, w, x;\n\t"
            "add.f32 x, x, %2;\n\t"
            "cvt.rn.bf16.f32 a, x;\n\t"
            "mov.b32 %0, {a, _};\n\t"
            : "=r"(y)
            : "r"(x), "f"(y)
        );
    }
}

__device__ void ldcg_b32v4(
    const uint32_t* src,
    uint32_t* dst
){
    asm volatile(
            "ld.global.cg.L2::128B.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
            : "=r"(dst[i]), "=r"(dst[i + 1]), "=r"(dst[i + 2]), "=r"(dst[i + 3])
            : "l"((uint64_t)__cvta_generic_to_global(src))
        );
}

template<bool SM100P>
__device__ void ldcg_b32v8(
    const uint32_t* src,
    uint32_t* dst
){
    if (SM100P) {
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