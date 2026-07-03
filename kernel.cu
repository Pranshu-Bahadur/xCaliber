#include <cooperative_groups.h>

using namespace cg = cooperative_groups;

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
    cg::thread_block cta1024 = cg::this_thread_block();
    cg::thread_block_tile<128, thread_block> cta128x8 = cg::tiled_partition<128>(cta1024);
    cg::coalesced_group rr32x32 = cg::labeled_partition(cta1024, ((cta1024.thread_rank() ^ 32) & 31));

	uint32_t rmem[48]; //16 32-bit for indexing
	__shared__ alignas(16) uchar smem[32768]; //32768B : 98304B
    __shared__ alignas(16) uint64_t mbar[5];

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

        if (rr32x32.meta_group_rank() < 2) {
            asm volatile(            
                     	"cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], 4096;\n\t"
                        "cp.async.bulk.commit_group;\n\t"
                        ::"l"(
                                (uint64_t)__cvta_generic_to_global(
                                W13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 4))
                                + (int64_t)((kt + ((rr32x32.meta_group_rank() ^ 2) & 1))) * (I << 4))
                                + (rr32x32.thread_rank() << 10)
                            )
                        ),
			            "n"((uint32_t)(I << 2)) //@TODO add condition based bytes calc (for tail of each panel)
			);
        }

        if ((rr32x32.meta_group_rank() >= 2) && (rr32x32.meta_group_rank() < 4)) {
            asm volatile(
                     	"cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], 1024;\n\t"
                        "cp.async.bulk.commit_group;\n\t"
                        "cp.async.bulk.prefetch.L2.global.L2::evict_last [%1], 512;\n\t"
                        "cp.async.bulk.commit_group;\n\t"
                        ::
                        "l"(
                                (uint64_t)__cvta_generic_to_global(
                                M13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
                                + (int64_t)((kt + ((rr32x32.meta_group_rank() ^ 2) & 1)) * (I << 2))
                                + (rr32x32.thread_rank() << 8)
                            )
                        ),
                        "l"(
                                (uint64_t)__cvta_generic_to_global(
                                S13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 1))
                                + (int64_t)((kt + ((rr32x32.meta_group_rank() ^ 2) & 1)) * (I << 1))
                                + (rr32x32.thread_rank() << 7)
                            )
                        ),
                        "n"((uint32_t)(I)) //@TODO add condition based bytes calc (for tail of each panel)
                        "n"((uint32_t)(I >> 1))
			);
        }

        for (int rri = 0; rri < 8; rri+=4) {
            for (int k = 0; k <= 2; k+=1) {

                if (k) {

                    if (
                            (rr32x32.meta_group_rank() == (4 + k + 1))
                    ) {
                        if ((!rr32x32.thread_rank())) {
                            asm volatile(
                                "mbarrier.arrive_drop.expect_tx.shared::cta.release.b64 _, [%0], 64;
                                ::
                                "r"((uint32_t)__cvta_generic_to_shared(mbar + 1))
                            );
                        }
                    }

                    if (
                           (rr32x32.meta_group_rank() == (4 + k))
                    ) {
                        if ((!rr32x32.thread_rank())) {
                            asm volatile(
                                "mbarrier.arrive_drop.expect_tx.shared::cta.release.b64 _, [%0], 128;
                                ::
                                "r"((uint32_t)__cvta_generic_to_shared(mbar + 2))
                            );
                        }
                    }

                    __syncthreads();

                    if (threadIdx.x & 1) {
                         asm volatile(
                            "ld.shared.acquire.b32 %0, %1;\n\t"
                            :"r"((uint32_t)(rmem + 1 + ((k - 1) * 6))),
                            :"r"((uint32_t)__cvta_generic_to_shared(smem + (threadIdx.x >> 1)))
                        );
                    }

                    asm volatile(
                            "ld.shared.acquire.b32 %0, %1;\n\t"
                            :"r"((uint32_t)(rmem + 2 + ((k - 1) * 6))),
                            :"r"((uint32_t)__cvta_generic_to_shared(smem + 2048 + threadIdx.x))
                    );

                    __syncthreads();
                   

                    if ((!rr32x32.thread_rank())) {
                            asm volatile(
                                "mbarrier.arrive_drop.expect_tx.shared::cta.release.b64 state, [%0], 16;
                                ::
                                "r"((uint32_t)__cvta_generic_to_shared(mbar))
                            );
                    }
                    __syncthreads();

                    asm volatile(
                            "ld.shared.acquire.v4.b32 %0, %1;\n\t"
                            :"r"((uint32_t)(rmem + 3 + ((k - 1) * 6))),
                            :"r"((uint32_t)__cvta_generic_to_shared(smem + 6144 + threadIdx.x))
                    );

                    __syncthreads();

                }

                if (k < 2) {
                    for (int rrip = 0; rrip < 4; rrip++) {
                        if (
                            (rr32x32.meta_group_rank() == (2 + k))
                            && (((rri << 2) + rrip) == rr32x32.thread_rank())
                        ) 
                        {
                            asm volatile(
                                "cp.async.bulk.wait_group 1;\n\t"
                            );
                        }
                    }

                    if (
                        (rr32x32.meta_group_rank() == (2 + k)) || (rr32x32.meta_group_rank() == (4 + k + 1))
                    ) 
                    {
                        rr32x32.sync();
                    }

                    if (
                            (rr32x32.meta_group_rank() == (4 + k + 1))
                    ) 
                    {

                        asm volatile(
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0], [%1], 32, [%2];\n\t"
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0 + 32], [%1 + 1024], 32, [%2];\n\t"
                                    :
                                    "r"(
                                        (uint32_t)__cvta_generic_to_shared(
                                            smem + ((rr32x32.thread_rank() << 6))
                                        )
                                    )
                                    :
                                    "l"(
                                            (uint64_t)__cvta_generic_to_global(
                                            S13
                                            + (int64_t)(blockIdx.x * (H >> 7) * (I << 1))
                                            + (int64_t)((kt + ((rr32x32.meta_group_rank() ^ 2) & 1)) * (I << 1))
                                            + (rr32x32.thread_rank() << 4)
                                        )
                                    ),
                                    "r"((uint32_t)__cvta_generic_to_shared(mbar + 1))
                            );
                    }

                    for (int rrip = 0; rrip < 4; rrip++) {
                        if (
                            (rr32x32.meta_group_rank() == (2 + k))
                            && (((rri << 2) + rrip) == rr32x32.thread_rank())
                        ) 
                        {
                            asm volatile(
                                "cp.async.bulk.wait_group 0;\n\t"
                            );
                        }
                    }

                    if (
                        (rr32x32.meta_group_rank() == (2 + k)) || (rr32x32.meta_group_rank() == (4 + k)) || (rr32x32.meta_group_rank() == (4 + k + 2))
                    ) 
                    {
                        rr32x32.sync();
                    }

                    if (
                            (rr32x32.meta_group_rank() == (4 + k)) || (rr32x32.meta_group_rank() == (4 + k + 2))
                    )
                    {

                        asm volatile(
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0], [%1], 32, [%2];\n\t"
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0 + 32], [%1 + 1024], 32, [%2];\n\t"
                                    :
                                    "r"(
                                        (uint32_t)__cvta_generic_to_shared(
                                            smem + (2048) + (((rr32x32.thread_rank() + (rr32x32.meta_group_rank() / 6)) << 6))
                                        )
                                    )
                                    :
                                    "l"(
                                            (uint64_t)__cvta_generic_to_global(
                                            M13
                                            + (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
                                            + (int64_t)((kt + ((rr32x32.meta_group_rank() ^ 2) & 1)) * (I << 2))
                                            + (((rr32x32.thread_rank() + (rr32x32.meta_group_rank() / 6)) << 4))
                                        )
                                    ),
                                    "r"((uint32_t)__cvta_generic_to_shared(mbar + 2))
                            );
                    }

                    for (int rrip = 0; rrip < 4; rrip++) {
                        if (
                            (rr32x32.meta_group_rank() == k)
                            && (((rri << 2) + rrip) == rr32x32.thread_rank())
                        ) 
                        {
                            asm volatile(
                                "cp.async.bulk.wait_group 0;\n\t"
                            );
                        }
                    }

                    if (rr32x32.meta_group_rank() == k || (rr32x32.meta_group_rank() >= 16)) {

                        rr32x32.sync();

                    }

                    if ((rr32x32.meta_group_rank() >= 16)) {

                        asm volatile(
                                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0], [%1], 32, [%2];\n\t"
                                    :
                                    "r"(
                                        (uint32_t)__cvta_generic_to_shared(
                                            smem + (6144) + ((rr32x32.meta_group_rank() - 16) + (rr32x32.thread_rank() << 5))
                                        )
                                    )
                                    :
                                    "l"(
                                            (uint64_t)__cvta_generic_to_global(
                                            W13
                                            + (int64_t)(blockIdx.x * (H >> 7) * (I << 4))
                                            + (int64_t)((kt + ((rr32x32.meta_group_rank() ^ 2) & 1)) * (I << 4))
                                            + ((rr32x32.meta_group_rank() - 16) + (rr32x32.thread_rank() << 4))
                                        )
                                    ),
                                    "r"((uint32_t)__cvta_generic_to_shared(mbar))
                            );
                    }
                }
            }
        }
    }
}