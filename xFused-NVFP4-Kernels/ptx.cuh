__device__ __forceinline__ void e2m1x8_to_bf16x8_neg(
    uint32_t packed,
    uint32_t& bf01,
    uint32_t& bf23,
    uint32_t& bf45,
    uint32_t& bf67
) {

    asm volatile(
        "{\n\t"
        ".reg .b32 x, wt, ws, t1, t2, t3, out45, out67;\n\t"
        "mov.b32 x, %4;\n\t"
        "lop3.b32 ws, x, 0x00080008, 0x00080008, 0x6A;\n\t"
        "shl.b32 ws, ws, 12;\n\t"
        "shl.b32 wt, x, 6;\n\t"
        "lop3.b32 wt, wt, 0x01C001C0, ws, 0xEA;\n\t"
        "lop3.b32 ws, x, 0x80008000, 0x80008000, 0x6A;\n\t"
        "shr.b32 t3, x, 6;\n\t"
        "lop3.b32 t3, t3, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 x, x, 4;\n\t"
        "lop3.b32 ws, x, 0x80008000, 0x80008000, 0x6A;\n\t"
        "shr.b32 t2, x, 6;\n\t"
        "lop3.b32 t2, t2, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 x, x, 4;\n\t"
        "lop3.b32 ws, x, 0x80008000, 0x80008000, 0x6A;\n\t"
        "shr.b32 t1, x, 6;\n\t"
        "lop3.b32 t1, t1, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 ws, t1, 16;\n\t"
        "lop3.b32 %0, ws, 0xFFFF0000, wt, 0xE2;\n\t"
        "shr.b32 wt, wt, 16;\n\t"
        "lop3.b32 out45, t1, 0xFFFF0000, wt, 0xE2;\n\t"
        "shr.b32 ws, t2, 16;\n\t"
        "lop3.b32 out67, t3, 0xFFFF0000, ws, 0xE2;\n\t"
        "shl.b32 %1, t3, 16;\n\t"
        "lop3.b32 %1, %1, 0xFFFF0000, t2, 0xE2;\n\t"
        "and.b32 %2, out45, 0x81C081C0;\n\t"
        "and.b32 %3, out67, 0x81C081C0;\n\t"
        "}\n\t"
        : "=r"(bf01), "=r"(bf23), "=r"(bf45), "=r"(bf67)
        : "r"(packed));
}

__device__ __forceinline__ void e2m1x8_to_bf16x8(
    uint32_t packed,
    uint32_t& bf01,
    uint32_t& bf23,
    uint32_t& bf45,
    uint32_t& bf67
) {

    asm volatile(
        "{\n\t"
        ".reg .b32 x, wt, ws, t1, t2, t3, out45, out67;\n\t"
        "mov.b32 x, %4;\n\t"
        "and.b32 ws, x, 0x00080008;\n\t"
        "shl.b32 ws, ws, 12;\n\t"
        "shl.b32 wt, x, 6;\n\t"
        "lop3.b32 wt, wt, 0x01C001C0, ws, 0xEA;\n\t"
        "and.b32 ws, x, 0x80008000;\n\t"
        "shr.b32 t3, x, 6;\n\t"
        "lop3.b32 t3, t3, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 x, x, 4;\n\t"
        "and.b32 ws, x, 0x80008000;\n\t"
        "shr.b32 t2, x, 6;\n\t"
        "lop3.b32 t2, t2, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 x, x, 4;\n\t"
        "and.b32 ws, x, 0x80008000;\n\t"
        "shr.b32 t1, x, 6;\n\t"
        "lop3.b32 t1, t1, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 ws, t1, 16;\n\t"
        "lop3.b32 %0, ws, 0xFFFF0000, wt, 0xE2;\n\t"
        "shr.b32 wt, wt, 16;\n\t"
        "lop3.b32 out45, t1, 0xFFFF0000, wt, 0xE2;\n\t"
        "shr.b32 ws, t2, 16;\n\t"
        "lop3.b32 out67, t3, 0xFFFF0000, ws, 0xE2;\n\t"
        "shl.b32 %1, t3, 16;\n\t"
        "lop3.b32 %1, %1, 0xFFFF0000, t2, 0xE2;\n\t"
        "and.b32 %2, out45, 0x81C081C0;\n\t"
        "and.b32 %3, out67, 0x81C081C0;\n\t"
        "}\n\t"
        : "=r"(bf01), "=r"(bf23), "=r"(bf45), "=r"(bf67)
        : "r"(packed));
}
