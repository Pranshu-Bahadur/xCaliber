#include <cooperative_groups.h>

using namespace cg = cooperative_groups;

//CTA=1024 (1 CTA = 1 SM = 1 cluster)
//@TODO profile vs 2SM / cluster (W1, W3 variant)
__global__ void kernel(
		        const uint32_t* __restrict__ W13, //[E, (H + 127) >> 7, I << 5]
                const uint32_t* __restrict__ S13, //[E, (H + 127) >> 7, I << 2]
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
    cg::thread_block cta1024 = cg::this_thread_block();
    cg::thread_block_tile<128, thread_block> cta128x8 = cg::tiled_partition<128>(cta1024);
    cg::coalesced_group rr32x32 = cg::labeled_partition(cta1024, ((cta1024.thread_rank() ^ 32) & 31));

	uint32_t rmem[48]; //16 32-bit for indexing
	__shared__ alignas(16) uchar smem[32768]; //32768B : 98304B
    __shared__ alignas(16) uint64_t mbar[4];

    if (!((threadIdx.x ^ 4) & 3)){
		rmem[0] = Xb[(int64_t)(blockIdx.x * (8 + ((N + 31) >> 5)) + (((threadIdx.x ^ 32) & 31) >> 2))];
	}
	rmem[0] = __shfl_sync(0xFFFF'FFFF, rmem[0], 0, 4);
	if (!__popc(rmem[0])) return;

	if (!(threadIdx.x & 31) && ((threadIdx.x >> 5) < 5)) {
		asm volatile(
			"mbarrier.init.layout::v0.shared::cta.b64 [%0], 512;\n\t"
			:
			: "r"((uint32_t)__cvta_generic_to_shared(mbar + (threadIdx.x >> 5)))
			: "memory"
		);
	}

    __syncthreads();

	for (int kt = 0; kt < (H >> 8); kt += 2) {

        for (int rr = 0; rr < 4; rr++) {
            #pragma unroll 5
            for (int rri = 0; rri <= 4; rri++) {
                    if ((rri==((rr32x32.thread_rank() ^ 4) & 3)) && (rr==((rr32x32.meta_group_rank() ^ 4) & 3))) {
                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], 256;\n\t"
                            "cp.async.bulk.commit_group;"
                            "ld.ca.global.L2::evict_first.L1::evict_first.b32 %0, %1;\n\t"
                            : "l"(
                                    (uint64_t)__cvta_generic_to_global(
                                        W13 
                                        + (int64_t)(blockIdx.x * (H >> 7) * (I << 5))
                                        + (int64_t)(kt * (I << 5))
                                        + (int64_t)((threadIdx.x << 6))
                                    )
                                ),
                            "l"(
                                    (uint64_t)__cvta_generic_to_global(
                                        S13 
                                        + (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
                                        + (int64_t)(kt * (I << 2))
                                        + (int64_t)((rr32x32.thread_rank()) + (rri << 6))
                                    )
                                )
                        );
                    }

                    if (rri) {
                        if (((rri-1)==((rr32x32.thread_rank() ^ 4) & 3)) {
                                if (!((rr32x32.meta_group_rank() ^ 4) & 3))) {
                                    asm volatile(
                                        "cp.async.bulk.wait_group 0;\n\t"
                                    );
                                }
                            __syncwarp();
                        }
                    }
                    

        __syncthreads();
        
        for (int rri )

            }
        }
    }

}