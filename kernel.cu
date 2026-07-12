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

		uint32_t rmem[57]; //16 32-bit for indexing
		__shared__ alignas(16) unsigned char smem[32768]; //32768B : 98304B
	    __shared__ alignas(16) uint64_t mbar[2];

    asm volatile(
                    "ldu.global.u32 %0, [%1];\n\t"
                    :
                    "=r"(rmem),
                    :
                    "l"(
                            (uint64_t)__cvta_generic_to_global(
                            Xb + (blockIdx.x * (1 + (N + 31) >> 5))
                        )
                    )
                );

	if (!__popc(rmem[0])) return;

	if (!(threadIdx.x & 31) && ((threadIdx.x >> 5) < 2)) {
		asm volatile(
			"mbarrier.init.shared::cta.b64 [%0], 1;\n\t"
            :
			: "r"((uint32_t)__cvta_generic_to_shared(mbar + (threadIdx.x >> 5)))
			: "memory"
		);
	}

    __syncthreads();


    for (int i = 0; i < (I >> 10); i++) {

            /*
                use kernel1.cu prefetch pattern on l2 but only waitgroup for matches
            */
            for (int n256 = 0; n256 < ((N + 255) >> 8); n256++) {

                for (int kt = 0; kt < (H >> 9); kt++) { //4xmma

                    if (!kt) {
                        asm volatile(
                                "ld.global.nc.cs.L2::64B.b128 %0, [%20];"
                                "ld.global.nc.cs.b128 %1, [%20 + 16384];"
                                "ld.global.nc.cs.b128 %2, [%20 + 32768];"
                                "ld.global.nc.cs.b128 %3, [%20 + 49152];" //@TODO instead of static add a !p bra loop here to load all H, as H <= 8192
                                :
                                "=r"(&(rmem + 1)),
                                "=r"(&((rmem + 1) << 2)),
                                "=r"(&((rmem + 9) << 2)),
                                "=r"(&((rmem + 13) << 2))
                                :
                                "l"(
                                    W13
                                    + (int64_t)blockIdx.x * ((H + 127) >> 7) * (I << 4)
                                    + (int64_t)((kt << 2) + ((uint32_t)threadIdx.x >> 8)) * (I << 4)
                                    + (int64_t)(i << 14)
                                    + (int64_t)(((uint32_t)threadIdx.x & 255u) << 2)
                                )
                            );
                            if (!((threadIdx.x ^ 4) & 3)) {
                                asm volatile(
                                    "ld.global.nc.cs.b32 %0, [%1];"
                                    "ld.global.nc.cs.b32 %1, [%1 + 1024];" //@TODO its 16 weight packets per 1 scale packet, this might be off
                                    "ld.global.nc.cs.b32 %2, [%1 + 2048];" //@TODO map S13 pairs onto packet-interleaved W13
                                    "ld.global.nc.cs.b32 %3, [%1 + 3072];" //@TODO W1/W3 need distinct scale ownership
                                    :
                                    "=r"(&(rmem + 17)),
                                    "=r"(&(rmem + 18)),
                                    "=r"(&(rmem + 19)),
                                    "=r"(&(rmem + 20))
                                    :
                                    "l"(
                                        S13
                                        + (int64_t)blockIdx.x * ((H + 127) >> 7) * (I << 2)
                                        + (int64_t)((kt << 2) + ((uint32_t)threadIdx.x >> 8)) * (I << 2)
                                        + (int64_t)(i << 12)
                                        + (int64_t)(((uint32_t)threadIdx.x & 255u) >> 2)
                                    )
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
}
