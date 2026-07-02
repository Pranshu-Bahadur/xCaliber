//CTA=1024
__global__ void ff1(
		const uint32_t* __restrict__ W13, //[E, (H + 127) >> 7, I << 4]
  		const uint32_t* __restrict__ M13, //[E, (H + 127) >> 7, I << 2]
		const uint32_t* __restrict__ S13, //[E, (H + 127) >> 7, I << 1]
        const float* __restrict__ W13GS, //[E, 2]
		const uint32_t* __restrict__ X4, //[N, H >> 3]
		const uint32_t* __restrict__ SX, //[N, H >> 7]
		const uint32_t* __restrict__ topk_bitplanes, //[E, 1 + ((N + 31) >> 5)]
		const __nv_bfloat16* __restrict__ topk_W, //[N, TOPK]
		__nv_bfloat16* Y, //[E, N, I]
		__nv_bfloat16* X, //[N, H]
		const uint16 E, 
		const uint32 N, 
		const uint16 I,
		const uint16 H,
		const uint8_t TOPK

){

    uint32_t rmem[64];
    __shared__ uchar smem[98304]; //98304B : 32768B

    if (!((threadIdx.x ^ 4) & 3)){
        rmem[0] = Xb[(int64_t)(blockIdx.x * (8 + ((N + 31) >> 5)) + (((threadIdx.x ^ 32) & 31) >> 2))];
    }
    rmem[0] = __shfl_sync(0xFFFF'FFFF, rmem[0], 0, 4);

    if (!__popc(rmem[0])) return;

    /*
		   0 w1-i0-h015, w1-i0-h1631, w1-i0-h3247, w1-i0-h4863, ..., 127, (8),  w3-i0-h015, w3-i0-h1631, w3-i0-h3247, w3-i0-h4863, ..., repeat pattern until 2I
		   1 128-255
		   2 256-383
		   3 384-511
		   .
		   .
		   . repeat until H / 128
	*/

    //@TODO fix indexing details
    //@TODO wire L2->smem->rmem (W13, M13) [in panels]
    //each warp will stride through I128' instead of spatial?
    //eitherway 5mb*i L2 cache used...so i gotta figure that out too

    for (int kt = 0; kt < (H >> 7) + 1; kt++) {
        for (int i = 0; i < (I >> 8); i++) {
            if (!((threadIdx.x ^ 32) & 31)) {
                if (kt < (H >> 7)) {
                    //I * 16 based prefetch for OOB
                    //4096B -> 1024 (I128' / warp) I (512 W1, 512 W3)
                    //2048B -> I512' 256, 256
                    asm volatile(
                        "cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], 2048;\n\t" //2048*32=16384
                        "cp.async.bulk.commit_group;\n\t"
                        ::"l"(
                                (uint64_t)__cvta_generic_to_global(
                                W13
                                + (int64_t)(blockIdx.x * (H >> 7) * (I << 4))
                                + (int64_t)(kt * (I << 4))
                                + (int64_t)((i << 12) + ((threadIdx.x >> 5) << 9))
                            )
                        )
                    );
                    if (i & 1) {
                        asm volatile(
									"cp.async.bulk.wait_group 0;"
						);

                        asm volatile(
                            "cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], 2048;\n\t" //2048*8=16384
                            "cp.async.bulk.commit_group;\n\t"
                            ::"l"(
                                    (uint64_t)__cvta_generic_to_global(
                                    M13
                                    + (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
                                    + (int64_t)(kt * (I << 2))
                                    + (int64_t)((i << 12) + ((threadIdx.x >> 5) << 9))
                                )
                            )
                        );
                    }
                }
            }
    }
}