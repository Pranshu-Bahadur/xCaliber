//CTA=1024 (1 CTA = 1 SM = 1 cluster)
//@TODO profile vs 2SM / cluster (W1, W3 variant)
__global__ void kernel(
		        const uint32_t* __restrict__ W13, //[E, (H + 63) >> 6, I << 5]
                const uint32_t* __restrict__ S13, //[E, (H + 63) >> 6, I << 2]
                const __nv_bfloat16* __restrict__ W13GS, //[E, 2]
                const __nv_bfloat16* __restrict__ X, //[N, H] <-atomicAdd
                const __nv_bfloat16 __restrict__ XGSINV,
                const int32_t* __restrict__ Xb, //[E, 1 + (N + 31) >> 5] bitplanes
                const __nv_bfloat16* __restrict__ topk_W, //[N, TOPK] @TODO multiply to partials
                __nv_bfloat16* Y, //[I, N]
                const int32_t E,
                const int32_t N,
                const int32_t I,
                const int32_t H,
                const int32_t TOPK
)
{

	uint32_t rmem[48]; //16 32-bit for indexing
	__shared__ alignas(16) unsigned char smem[32768]; //32768B : 98304B
    __shared__ alignas(16) uint64_t mbar[5];

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

	if (!(threadIdx.x & 31) && ((threadIdx.x >> 5) < 5)) {
		asm volatile(
			"mbarrier.init.shared::cta.b64 [%0], 512;\n\t"
			:
			: "r"((uint32_t)__cvta_generic_to_shared(mbar + (threadIdx.x >> 5)))
			: "memory"
		);
	}

    __syncthreads();

	for (int kt = 0; kt < (H >> 6); kt++) {

        for (int i = 0; i < ((I << 5) >> 8); i++) {

            asm volatile(
                "ld.global.nc.ca.L2::64B.b128 %0, [%20];"
                "ld.global.nc.L1::evict_first.b128 %1, [%20 + 16384];"
                "ld.global.nc.L1::evict_first.b128 %2, [%20 + 32768];"
                "ld.global.nc.L1::evict_first.b128 %3, [%20 + 49152];"
                :
                "=r"(&(rmem + 1)), 
                "=r"(&((rmem + 1) << 2)), 
                "=r"(&((rmem + 9) << 2)), 
                "=r"(&((rmem + 13) << 2)),
                :
                "l"(
                    W13
                    + (int64_t)(blockIdx.x * (H >> 6) * (I << 5))
                    + (int64_t)(threadIdx.x << 2)
                )
            );
            if (!((threadIdx.x ^ 4) & 3)) {
                asm volatile(
                    "ld.global.nc.ca.b32 %0, [%1];"
                    "ld.global.nc.ca.b32 %1, [%1 + 2048];"
                    "ld.global.nc.ca.b32 %2, [%1 + 4096];"
                    "ld.global.nc.ca.b32 %3, [%1 + 6144];"
                    :
                    "=r"(&(rmem + 17)),
                    "=r"(&(rmem + 18)),
                    "=r"(&(rmem + 19)),
                    "=r"(&(rmem + 20))
                    :
                    "l"(
                        S13
                        + (int64_t)(blockIdx.x * (H >> 6) * (I << 2))
                        + (int64_t)(threadIdx.x >> 2)
                    )
                );
            }

            /*
                use kernel1.cu prefetch pattern on l2 but only waitgroup for matches 
            */

            for (int n256 = 0; n256 < ((N + 31) >> 5) >> 3; n256++) {
                
                // load using + 8 offset next Xb panel each warp within 256 threads loads 1 packet

                asm volatile(
                    "ldu.global.u32 %0, [%1];\n\t"
                    :
                    "=r"(rmem),
                    :
                    "l"(
                            (uint64_t)__cvta_generic_to_global(
                            Xb + (blockIdx.x * (1 + (N + 31) >> 5)) + (((threadIdx.x >> 5) ^ 8) & 7) + (n256 << 3)
                        )
                    )
                );

                //gpu-wide prefetch
                
                asm volatile(
                    "cp.async.bulk.prefetch.L2.global [%0], 128;\n\t"
                    "cp.async.bulk.commit_group;"
                    :
                    :
                    "l"(
                        (uint64_t)__cvta_generic_to_global(
                        X 
                        + (((n256 << 8) + (threadIdx.x ^ 256) & 255) * (H >> 1))
                        + (kt << 7) + (threadIdx.x >> 8))
                    )
                );

                rmem[0] = ((1u >> ((uint32_t)((threadIdx.x ^ 32) & 31))) & rmem[0]);

                //@TODO lut

                //ld bf16 scattered compression / smem distribute compression


            }    

                
        }
        
    }
}
