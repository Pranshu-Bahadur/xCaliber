//CTA=1024 (1 CTA = 1 SM = 1 cluster)

__device__ __constant__ uint32_t XbLUT[32] = {
    0x00000001u, 0x00000002u, 0x00000004u, 0x00000008u,
    0x00000010u, 0x00000020u, 0x00000040u, 0x00000080u,
    0x00000100u, 0x00000200u, 0x00000400u, 0x00000800u,
    0x00001000u, 0x00002000u, 0x00004000u, 0x00008000u,
    0x00010000u, 0x00020000u, 0x00040000u, 0x00080000u,
    0x00100000u, 0x00200000u, 0x00400000u, 0x00800000u,
    0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u,
    0x10000000u, 0x20000000u, 0x40000000u, 0x80000000u
};

__global__ void kernel(
		        const uint32_t* __restrict__ W13, //[E, (H + 127) >> 7, I << 4] = I << 4 u32
                const uint32_t* __restrict__ S13, //[E, (H + 127) >> 7, I << 2]
                const float* __restrict__ W13GS, //[E, 2]
                const uint32_t* __restrict__ X4, //[N, H >> 3]
                const uint32_t* __restrict__ SX, //[N, H >> 6]
                const float __restrict__ XGSINV,
                const uint32_t* __restrict__ Xb, //[E, 1 + (N + 31) >> 5] bitplanes
                const __nv_bfloat16* __restrict__ topk_W, //[N, TOPK] @TODO multiply to partials 
                __nv_bfloat16* Y, //[I, N] or [N, I]
                const int32_t E,
                const int32_t N,
                const int32_t I,
                const int32_t H,
                const int32_t TOPK
)
{

	uint32_t rmem[52]; //0 route | 1..40 W13 | 41 S13 | 42..51 consumer
	extern __shared__ alignas(16) unsigned char smem[]; //98304B W13 surplus

    asm volatile(
                    "ldu.global.u32 %0, [%1];\n\t"
                    :
	                    "=r"(rmem[0]),
                    :
                    "l"(
                            (uint64_t)__cvta_generic_to_global(
                            Xb + (blockIdx.x * (1 + (N + 31) >> 5))
                        )
                    )
                );

	if (!__popc(rmem[0])) return;

    for (int i = 0; i < (I >> 10); i++) {

            /*
                use kernel1.cu prefetch pattern on l2 but only waitgroup for matches
            */
            for (int n256 = 0; n256 < ((N + 255) >> 8); n256++) {

                for (int kt = 0; kt < ((H + 511) >> 9); kt++) { //4xmma

                    if (!kt && !n256) {
	                        // p0..p9: 10 H512 steps x v4 = 40 parked registers.
	                        asm volatile(
	                            "{\n\t"
	                            ".reg .pred p;\n\t"
	                            ".reg .u32 h;\n\t"
	                            ".reg .b64 a;\n\t"
	                            "mov.u32 h, %%tid.x;\n\t"
	                            "shr.u32 h, h, 8;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R0_done;\n\t"
	                            "mov.b64 a, %20;\n\t"
	                            "ld.global.nc.cs.L2::64B.v4.u32 {%0, %1, %2, %3}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R0_done;\n\t"
	                            "mad.wide.u32 a, %21, 256, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%4, %5, %6, %7}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R0_done;\n\t"
	                            "mad.wide.u32 a, %21, 512, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%8, %9, %10, %11}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R0_done;\n\t"
	                            "mad.wide.u32 a, %21, 768, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%12, %13, %14, %15}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R0_done;\n\t"
	                            "mad.wide.u32 a, %21, 1024, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%16, %17, %18, %19}, [a];\n\t"
	                            "W13R0_done:\n\t"
	                            "}\n"
	                            : "=&r"(rmem[1]),  "=&r"(rmem[2]),  "=&r"(rmem[3]),  "=&r"(rmem[4]),
	                              "=&r"(rmem[5]),  "=&r"(rmem[6]),  "=&r"(rmem[7]),  "=&r"(rmem[8]),
	                              "=&r"(rmem[9]),  "=&r"(rmem[10]), "=&r"(rmem[11]), "=&r"(rmem[12]),
	                              "=&r"(rmem[13]), "=&r"(rmem[14]), "=&r"(rmem[15]), "=&r"(rmem[16]),
	                              "=&r"(rmem[17]), "=&r"(rmem[18]), "=&r"(rmem[19]), "=&r"(rmem[20])
	                            : "l"((uint64_t)__cvta_generic_to_global(
	                                    W13
	                                    + (int64_t)blockIdx.x * (H >> 7) * (I << 4)
	                                    + (int64_t)((uint32_t)threadIdx.x >> 8) * (I << 4)
	                                    + (int64_t)(i << 14)
	                                    + (int64_t)(((uint32_t)threadIdx.x & 255u) << 2))),
	                              "r"((uint32_t)I),
	                              "r"((uint32_t)H >> 7)
	                            : "memory"
	                        );

	                        asm volatile(
	                            "{\n\t"
	                            ".reg .pred p;\n\t"
	                            ".reg .u32 h;\n\t"
	                            ".reg .b64 a;\n\t"
	                            "mov.u32 h, %%tid.x;\n\t"
	                            "shr.u32 h, h, 8;\n\t"
	                            "add.u32 h, h, 20;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R1_done;\n\t"
	                            "mad.wide.u32 a, %21, 1280, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%0, %1, %2, %3}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R1_done;\n\t"
	                            "mad.wide.u32 a, %21, 1536, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%4, %5, %6, %7}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R1_done;\n\t"
	                            "mad.wide.u32 a, %21, 1792, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%8, %9, %10, %11}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R1_done;\n\t"
	                            "mad.wide.u32 a, %21, 2048, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%12, %13, %14, %15}, [a];\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.ge.u32 p, h, %22;\n\t"
	                            "@p bra.uni W13R1_done;\n\t"
	                            "mad.wide.u32 a, %21, 2304, %20;\n\t"
	                            "ld.global.nc.cs.v4.u32 {%16, %17, %18, %19}, [a];\n\t"
	                            "W13R1_done:\n\t"
	                            "}\n"
	                            : "=&r"(rmem[21]), "=&r"(rmem[22]), "=&r"(rmem[23]), "=&r"(rmem[24]),
	                              "=&r"(rmem[25]), "=&r"(rmem[26]), "=&r"(rmem[27]), "=&r"(rmem[28]),
	                              "=&r"(rmem[29]), "=&r"(rmem[30]), "=&r"(rmem[31]), "=&r"(rmem[32]),
	                              "=&r"(rmem[33]), "=&r"(rmem[34]), "=&r"(rmem[35]), "=&r"(rmem[36]),
	                              "=&r"(rmem[37]), "=&r"(rmem[38]), "=&r"(rmem[39]), "=&r"(rmem[40])
	                            : "l"((uint64_t)__cvta_generic_to_global(
	                                    W13
	                                    + (int64_t)blockIdx.x * (H >> 7) * (I << 4)
	                                    + (int64_t)((uint32_t)threadIdx.x >> 8) * (I << 4)
	                                    + (int64_t)(i << 14)
	                                    + (int64_t)(((uint32_t)threadIdx.x & 255u) << 2))),
	                              "r"((uint32_t)I),
	                              "r"((uint32_t)H >> 7)
	                            : "memory"
	                        );

	                        // p10..p15: six H512 steps x 16KB = 96KB shared.
	                        asm volatile(
	                            "{\n\t"
	                            ".reg .pred p;\n\t"
	                            ".reg .u32 h, s;\n\t"
	                            ".reg .b64 g, step;\n\t"
	                            "mov.u32 h, %%tid.x;\n\t"
	                            "shr.u32 h, h, 8;\n\t"
	                            "add.u32 h, h, 40;\n\t"
	                            "setp.ge.u32 p, h, %3;\n\t"
	                            "@p bra.uni W13S_done;\n\t"
	                            "mov.u32 s, %0;\n\t"
	                            "mov.b64 g, %1;\n\t"
	                            "cvt.u64.u32 step, %2;\n\t"
	                            "shl.b64 step, step, 8;\n\t"
	                            "W13S_loop:\n\t"
	                            "cp.async.cg.shared.global [s], [g], 16;\n\t"
	                            "add.u32 s, s, 16384;\n\t"
	                            "add.u64 g, g, step;\n\t"
	                            "add.u32 h, h, 4;\n\t"
	                            "setp.lt.u32 p, h, %3;\n\t"
	                            "@p bra.uni W13S_loop;\n\t"
	                            "cp.async.commit_group;\n\t"
	                            "W13S_done:\n\t"
	                            "}\n"
	                            :
	                            : "r"((uint32_t)__cvta_generic_to_shared(
	                                    smem + ((uint32_t)threadIdx.x << 4))),
	                              "l"((uint64_t)__cvta_generic_to_global(
	                                    W13
	                                    + (int64_t)blockIdx.x * (H >> 7) * (I << 4)
	                                    + (int64_t)(40 + ((uint32_t)threadIdx.x >> 8)) * (I << 4)
	                                    + (int64_t)(i << 14)
	                                    + (int64_t)(((uint32_t)threadIdx.x & 255u) << 2))),
	                              "r"((uint32_t)I),
	                              "r"((uint32_t)H >> 7)
	                            : "memory"
	                        );
	                    }

	                    // 4 kt x 4 scale planes x 32B per warp = 16KB CTA-wide.
	                    if (!n256 && !(kt & 3) && ((threadIdx.x & 31) < 16)
	                        && ((((kt + ((threadIdx.x & 31) >> 2)) << 2)
	                            + ((uint32_t)threadIdx.x >> 8)) < (H >> 7))) {
	                        asm volatile(
	                            "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
	                            "cp.async.bulk.commit_group;\n\t"
	                            :
	                            : "l"((uint64_t)__cvta_generic_to_global(
	                                S13
	                                + (int64_t)blockIdx.x * (H >> 7) * (I << 2)
	                                + (int64_t)(((kt + ((threadIdx.x & 31) >> 2)) << 2)
	                                    + ((uint32_t)threadIdx.x >> 8)) * (I << 2)
	                                + (int64_t)(i << 12)
	                                + (int64_t)(((uint32_t)threadIdx.x & 3u) << 8)
	                                + (int64_t)((((uint32_t)threadIdx.x & 255u) >> 5) << 3))))
	                            : "memory"
	                        );
	                    }

	                    rmem[41] = 0u;
	                    if (((kt << 2) + ((uint32_t)threadIdx.x >> 8)) < (H >> 7)) {
	                        asm volatile(
	                            "ld.global.nc.cs.b32 %0, [%1];\n\t"
	                            : "=r"(rmem[41])
	                            : "l"((uint64_t)__cvta_generic_to_global(
	                                S13
	                                + (int64_t)blockIdx.x * (H >> 7) * (I << 2)
	                                + (int64_t)((kt << 2)
	                                    + ((uint32_t)threadIdx.x >> 8)) * (I << 2)
	                                + (int64_t)(i << 12)
	                                + (int64_t)(((uint32_t)threadIdx.x & 3u) << 8)
	                                + (int64_t)(((uint32_t)threadIdx.x & 255u) >> 2)))
	                            : "memory"
	                        );
	                    }
                        
	                        rmem[0] = 0u;
                        
                        if ((n256 << 3) + (((uint32_t)threadIdx.x & 255u) >> 5)
                            < ((N + 31) >> 5)) {
                            asm volatile(
                                "ldu.global.u32 %0, [%1];\n\t"
                                : "=r"(rmem[0])
                                : "l"((uint64_t)__cvta_generic_to_global(
                                    Xb
                                    + blockIdx.x * (1 + ((N + 31) >> 5))
                                    + 1
                                    + (n256 << 3)
                                    + (((uint32_t)threadIdx.x & 255u) >> 5)))
                                : "memory"
                            );
                        }

                        rmem[0] &= XbLUT[(uint32_t)threadIdx.x & 31u];
                        
                        if ((n256 << 8) + ((uint32_t)threadIdx.x & 255u) >= N)
                            rmem[0] = 0u;
                        
	                        // gpu-wide prefetch; all four bands keep q and own one H128.
                        if (rmem[0]) {
                            asm volatile(
                                "cp.async.bulk.prefetch.L2.global [%0], 64;\n\t"
                                "cp.async.bulk.commit_group;\n\t"
                                :
                                :
                                "l"(
                                    (uint64_t)__cvta_generic_to_global(
                                    X4 
                                    + (int64_t)((n256 << 8)
                                        + ((uint32_t)threadIdx.x & 255u)) * (H >> 3)
                                    + (kt << 6)
                                    + (((uint32_t)threadIdx.x >> 8) << 4))
                                )
                            );

                            if (!(threadIdx.x >> 8)) {

                                asm volatile(
                                    "cp.async.bulk.prefetch.L2.global [%0], 32;\n\t"
                                    "cp.async.bulk.commit_group;\n\t"
                                    :
                                    :
                                    "l"(
                                        (uint64_t)__cvta_generic_to_global(
                                        SX 
                                        + (int64_t)((n256 << 8)
                                            + ((uint32_t)threadIdx.x & 255u)) * (H >> 6)
                                        + (kt << 3)
                                    )
                                );
                            }
                        }
                        __syncthreads();


	            }
	        }
}
}
