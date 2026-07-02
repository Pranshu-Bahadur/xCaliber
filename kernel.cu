//CTA=1024 (1 CTA = 1 SM = 1 cluster)
//@TODO profile vs 2SM / cluster (W1, W3 variant)
__global__ void kernel(
		        const uint32_t* __restrict__ W13, //[E, (H + 127) >> 7, I << 4]
                const uint32_t* __restrict__ M13, //[E, (H + 127) >> 7, I << 2]
                const uint32_t* __restrict__ S13, //[E, (H + 127) >> 7, I << 1]
                const __nv_bfloat16* __restrict__ W13GS, //[E, 2]
                const uint32_t* __restrict__ X4, //[N, H >> 3]
                const uint32_t* __restrict__ SX, //[N, H >> 7]
                const uint32_t* __restrict__ Xb, //[E,(8 + ((N + 31) >> 5))]
                const __nv_bfloat16* __restrict__ topk_W, //[N, TOPK]
                __nv_bfloat16* Y, //[E, N, I]
                __nv_bfloat16* X, //[N, H]
                const int32_t E,
                const int32_t N,
                const int32_t I,
                const int32_t H,
                const int32_t TOPK
)
{
	uint32_t rmem[48]; //16 32-bit for indexing
	__shared__ uchar smem[32728]; //32768B : 98304B
    __shared__ alignas(16) uint64_t mbar[5];

    if (!((threadIdx.x ^ 4) & 3)){
		rmem[0] = Xb[(int64_t)(blockIdx.x * (8 + ((N + 31) >> 5)) + (((threadIdx.x ^ 32) & 31) >> 2))];
	}
	rmem[0] = __shfl_sync(0xFFFF'FFFF, rmem[0], 0, 4);
	if (!__popc(rmem[0])) return;

	if (!(threadIdx.x & 31) && ((threadIdx.x >> 5) < 5)) {
		asm volatile(
			"mbarrier.init.layout::v0.shared::cta.b64 [%0], 4;\n\t"
			:
			: "r"((uint32_t)__cvta_generic_to_shared(mbar + (threadIdx.x >> 5)))
			: "memory"
		);
	}

    __syncthreads();

	for (int kt = 0; kt < (H >> 9); kt += 4) {

		//Ix128x4 panel prefetch
		//every 8 warps does Ix128 
		

		if (!(threadIdx.x >> 7) && !((threadIdx.x ^ 32) & 31)) {
			asm volatile(
                 	"cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], %1;\n\t" //2048*8=16384
                        "cp.async.bulk.commit_group;\n\t"
			            "cp.async.bulk.wait_group 0;\n\t"
                        ::"l"(
                                (uint64_t)__cvta_generic_to_global(
                                W13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 4))
                                + (int64_t)((kt + (threadIdx.x >> 5)) * (I << 4))
                            )
                        ),
			"n"((uint32_t)(I << 2)) //I based OOB @TODO make const
			);
		}

        if ((threadIdx.x >> 7) < 2) {
            __syncthreads();
        }

         if (((threadIdx.x >> 7)==1) && !((threadIdx.x ^ 32) & 31)) {
                        asm volatile(
                            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0], [%1], %3, [%2];\n\t"
                            :
                            :"r"(smem + ),//@TODO add indexing
                             "l"(
                                (uint64_t)__cvta_generic_to_global(
                                W13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 4))
                                + (int64_t)((kt + (threadIdx.x >> 5)) * (I << 4))
                            ),
                            "r"(mbar),
                            "n"((uint32_t)(I << 2))
                            : "memory"
                );
        }
        

		if (((threadIdx.x >> 7)==2) && !((threadIdx.x ^ 32) & 31)) {
                        asm volatile(
                        "cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], %1;\n\t" //2048*8=16384
                        "cp.async.bulk.commit_group;\n\t"
			            "cp.async.bulk.wait_group 0;\n\t"
                        ::"l"(
                                (uint64_t)__cvta_generic_to_global(
                                M13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
                                + (int64_t)((kt + (threadIdx.x >> 5)) * (I << 2))
                            )
                        ),
                        "n"((uint32_t)(I << 1)) //I based OOB @TODO make const
			);
        }

       


		if (((threadIdx.x >> 7)==4) && !((threadIdx.x ^ 32) & 31)) {
                        asm volatile(
                        "cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], %1;\n\t" //2048*8=16384
                        "cp.async.bulk.commit_group;\n\t"
                        "cp.async.bulk.wait_group 0;\n\t"
                        ::"l"(
                                (uint64_t)__cvta_generic_to_global(
                                S13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 1))
                                + (int64_t)((kt + (threadIdx.x >> 5)) * (I << 1))
                            )
                        ),
                        "n"((uint32_t)(I >> 1)) //I based OOB @TODO make const
			);
        }

        if (!kt) {
            __syncthreads();
        }
		
		//L2->L1
		//L1->smem
		
		//load activations 8x128x4		


	}


	

}
