#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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
    cg::thread_block_tile<128, cg::thread_block> cta128x8 = cg::tiled_partition<128>(cta1024);
    cg::coalesced_group rr32x32 = cg::labeled_partition(cta1024, ((cta1024.thread_rank() ^ 32) & 31));

	uint32_t rmem[48]; //16 32-bit for indexing
	__shared__ alignas(16) unsigned char smem[32768]; //32768B : 98304B
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

    /*
      Round-robin W13 L2 prefetch board
      ---------------------------------

      CTA1024 = 32 warps x 32 lanes = 1024 threads.

      chunk256:
        chunk256 = rr_rank*32 + rr_lane = threadIdx.x
        1 chunk = 256B = 64 u32

      4x4 phase grid:
        rr  = lane mod 4  = 0..3
        rri = rank mod 4  = 0..3

      One phase (rr,rri):
        lanes: rr, rr+4, rr+8, ..., rr+28        = 8 lanes / warp
        ranks: rri, rri+4, rri+8, ..., rri+28    = 8 warps
        active threads = 8 * 8 = 64
        bytes/thread   = 256B
        bytes/phase    = 64 * 256B = 16384B

      All phases:
        4 * 4 phases = 16
        16 * 16384B = 262144B = 256KB

      Staggered wait_group:

        rri slot 0: issue r%4=0
        rri slot 1: issue r%4=1 | wait r%4=0
        rri slot 2: issue r%4=2 | wait r%4=1
        rri slot 3: issue r%4=3 | wait r%4=2
        rri slot 4: drain        | wait r%4=3

        wait lane:
          lane mod4 == 0 is the control lane per physical warp

        no final blunt wait:
          the fifth slot drains the previous issued group.

      Horizontal phase map:

        phase rr0/rri0:
          r00 l00,l04,...,l28 | r04 l00,l04,...,l28 | ... | r28 l00,l04,...,l28

        phase rr1/rri0:
          r00 l01,l05,...,l29 | r04 l01,l05,...,l29 | ... | r28 l01,l05,...,l29

        phase rr2/rri0:
          r00 l02,l06,...,l30 | r04 l02,l06,...,l30 | ... | r28 l02,l06,...,l30

        phase rr3/rri0:
          r00 l03,l07,...,l31 | r04 l03,l07,...,l31 | ... | r28 l03,l07,...,l31

        phase rr0/rri1:
          r01 l00,l04,...,l28 | r05 l00,l04,...,l28 | ... | r29 l00,l04,...,l28

        ...

        phase rr3/rri3:
          r03 l03,l07,...,l31 | r07 l03,l07,...,l31 | ... | r31 l03,l07,...,l31

      Address map:

        W13 u32 offset = chunk256 << 6
        W13 byte span  = [chunk256*256B, chunk256*256B + 255B]

      Therefore each phase issues a 16KB distributed prefetch over the 256KB
      tile, and all 16 phases cover each 256B chunk exactly once.

      S13 below is a one-u32 side probe tied to the round-robin phase. It is
      not the full S13 staging contract.
    */

	for (int kt = 0; kt < (H >> 8); kt += 2) {

        for (int rr = 0; rr < 4; rr++) {
            #pragma unroll 5
            for (int rri = 0; rri <= 4; rri++) {
                if (
                    (rri < 4)
                    &&
                    (rr == (rr32x32.meta_group_rank() & 3))
                    && (rri == (rr32x32.thread_rank() & 3))
                ) {
                    asm volatile(
                        "cp.async.bulk.prefetch.L2.global.L2::evict_last [%1], 256;\n\t"
                        "cp.async.bulk.commit_group;\n\t"
                        "ld.ca.global.L2::evict_first.L1::evict_first.b32 %0, [%2];\n\t"
                        : "=r"(rmem[1 + (rr << 2) + rri])
                        : "l"(
                                (uint64_t)__cvta_generic_to_global(
                                    W13
                                    + (int64_t)(blockIdx.x * (H >> 7) * (I << 5))
                                    + (int64_t)(kt * (I << 5))
                                    + (int64_t)(threadIdx.x << 6)
                                )
                            ),
                          "l"(
                                (uint64_t)__cvta_generic_to_global(
                                    S13
                                    + (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
                                    + (int64_t)(kt * (I << 2))
                                    + (int64_t)(rr32x32.thread_rank() + (rri << 6))
                                )
                            )
                        : "memory"
                    );
                }

                if (rri) {
                    if ((rri - 1) == (rr32x32.thread_rank() & 3)) {
                        if (!((rr32x32.meta_group_rank() ^ 4) & 3)) {
                            asm volatile(
                                "cp.async.bulk.wait_group 0;\n\t"
                                ::: "memory"
                            );
                        }

                        __syncwarp();
                    }
                }
            }
        }

        __syncthreads();
    }

}
