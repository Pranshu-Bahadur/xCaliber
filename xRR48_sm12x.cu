#include <cuda_bf16.h>
#include <stdint.h>

using uchar = unsigned char;
using uint16 = uint16_t;
using uint32 = uint32_t;

/*
    CTA=256
*/
__global__ void ff1SwiGLUxtopkWff2_w4a4w4216a16sp48( //nvfp4 sparse 4:8 (pair-wise), bf16
		const uint32_t* __restrict__ W13, //[E, (H + 127) >> 7, I << 4]
		/*
		   0 w1-i0-h015, w1-i0-h1631, w1-i0-h3247, w1-i0-h4863, ..., 127, (8),  w3-i0-h015, w3-i0-h1631, w3-i0-h3247, w3-i0-h4863, ..., repeat pattern until 2I
		   1 128-255
		   2 256-383
		   3 384-511
		   .
		   .
		   . repeat until H / 128
		*/
		const uint32_t* __restrict__ M13, //[E, (H + 63) >> 6, I << 2]
		/*
		  0 m1i0h063, m1i0h64127, m3i0h063, m3i0h64127
		  1 
		  2
		  3 
		  metadata is 4:8 in pairs (like 2:4 but half the size)
		  Figure 267 shows 0...63, but since nvfp4 is 4:8 (in pairs), in our case this is 0...127
		  note: the figure only shows M=31
		*/
		const uint32_t* __restrict__ S13, //[E, (H + 127) >> 7, I << 1]
		/*
		  source, same logic as M13:
		    0 s1i0h0127, s3i0h0127
		    1 s1i1h0127, s3i1h0127
		    2 s1i2h0127, s3i2h0127
		    ...

		  required dst / F233 pane:
		    row    c0          c1          c2          c3
		    00     s1 i00      s1 i32      s3 i00      s3 i32
		    01     s1 i01      s1 i33      s3 i01      s3 i33
		    ...
		    31     s1 i31      s1 i63      s3 i31      s3 i63

		  one uint32_t scale packet = 4x u8 scales = 128 sparse cols
		  SFA ID = 00
		  ref: figure 233 (ptx isa 9.3) -> sparse variant
		*/
		//@TODO add W2, M2, S2
		const float* __restrict__ W132GS, //[E, 3]
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
	/*
	   pre-emption sentinel:
	     gmem: topk_bitplanes[e][0]
	     w0:   lane0 ld.acquire.cta.L1::evict_last.u32 -> shfl all lanes
	     sync
	     w1..7 lane0 ld.acquire.cta.L1::no_allocate.u32 -> shfl all lanes
	     sync
	     0 -> return before mtbar/smem/tmem/rmem
	*/
	uint32 route_sentinel = 0u;

	if (!(threadIdx.x >> 5)) {
		if (!(threadIdx.x & 31)) {
			asm volatile(
				"ld.global.acquire.cta.L1::evict_last.u32 %0, [%1];\n\t"
				: "=r"(route_sentinel)
				: "l"((uint64_t)__cvta_generic_to_global(
						topk_bitplanes
						+ ((uint64_t)blockIdx.x * (1 + ((N + 31) >> 5)))))
				: "memory"
			);
		}

		asm volatile(
			"shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;\n\t"
			: "=r"(route_sentinel)
			: "r"(route_sentinel)
		);
	}

	__syncthreads();

	if (threadIdx.x >> 5) {
		if (!(threadIdx.x & 31)) {
			asm volatile(
				"ld.global.acquire.cta.L1::no_allocate.u32 %0, [%1];\n\t"
				: "=r"(route_sentinel)
				: "l"((uint64_t)__cvta_generic_to_global(
						topk_bitplanes
						+ ((uint64_t)blockIdx.x * (1 + ((N + 31) >> 5)))))
				: "memory"
			);
		}

		asm volatile(
			"shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;\n\t"
			: "=r"(route_sentinel)
			: "r"(route_sentinel)
		);
	}

	__syncthreads();

	if (!route_sentinel) {
		return;
	}

	/*
	   mtbar:
	     [0] stage mbar
	     [1..7] rsv

	   smem:
	     M13 stage = smem+0000..1023
	     S13 raw   = smem+1024..1535

	     m0 smem+0000..0255 -> i00..15
	     m1 smem+0256..0511 -> i16..31
	     m2 smem+0512..0767 -> i32..47
	     m3 smem+0768..1023 -> i48..63

	     s0 smem+1024..1279 -> i00..31
	     s1 smem+1280..1535 -> i32..63

	     M13:
	       row = ((warp << 1) + lane01)
	       dst = smem + (m << 8) + (row << 4)

	     S13:
	       raw = smem + 1024 + (s << 8) + (row << 4)
	       row = 16B = {s1 i2r, s3 i2r, s1 i2r+1, s3 i2r+1}
	       ldmatrix.x4.trans -> 4x 8x8.b16 sub-tile transpose / warp
	*/
	__shared__ alignas(16) uint64_t mtbar[8];
	__shared__ alignas(16) uchar smem[131072];

	if (!threadIdx.x) {
		asm volatile(
			"mbarrier.init.layout::v0.shared::cta.b64 [%0], 1;\n\t"
			:
			: "r"((uint32)__cvta_generic_to_shared(mtbar))
			: "memory"
		);
	}

	__syncthreads();

	/*
	   load routine:

	     CTA=256 = 8 warps
	     i = I64 == I128'
	     kt = H128

	     W13 / i row:
	       16 u32 = 64B
	       8 W1 packets + 8 W3 packets

	     W13 / panel:
	       I64 x 64B = 4096B
	       lane0 / warp x 8 warps x 512B = 4096B

	     M13 / i row:
	       4 u32 = 16B
	       {m1 h000..063, m1 h064..127, m3 h000..063, m3 h064..127}

	     M13 / panel:
	       I64 x 16B = 1024B
	       lane0,lane1 / warp = 16 lanes
	       16 lanes x 16B = 256B

	     S13 / i row:
	       2 u32 = 8B
	       {s1 h000..127, s3 h000..127}

	     S13 / panel:
	       I64 x 8B = 512B
	       lane0,lane1 / warp = 16 lanes
	       16 lanes x 16B = 256B

	     why m < 4:
	       4 x 256B = 1024B
	       m0 -> i00..15
	       m1 -> i16..31
	       m2 -> i32..47
	       m3 -> i48..63

	     why s < 2:
	       2 x 256B = 512B
	       s0 -> i00..31
	       s1 -> i32..63
	*/

	/*
	   vertical W13:

	     baseW = W13 + e*PH*(I<<4) + kt*(I<<4) + i*1024

	     warp.lane   gmem u32     bytes        i rows
	     ---------   --------     ---------    -------
	     w0.l0       +0000        0000..0511   i00..07
	     w1.l0       +0128        0512..1023   i08..15
	     w2.l0       +0256        1024..1535   i16..23
	     w3.l0       +0384        1536..2047   i24..31
	     w4.l0       +0512        2048..2559   i32..39
	     w5.l0       +0640        2560..3071   i40..47
	     w6.l0       +0768        3072..3583   i48..55
	     w7.l0       +0896        3584..4095   i56..63
	*/

	/*
	   horizontal W13:

	     lane       w0.l0       w1.l0       w2.l0       w3.l0       w4.l0       w5.l0       w6.l0       w7.l0
	     ----       -----       -----       -----       -----       -----       -----       -----       -----
	     gmem u32   +0000       +0128       +0256       +0384       +0512       +0640       +0768       +0896
	     bytes      0000..0511  0512..1023  1024..1535  1536..2047  2048..2559  2560..3071  3072..3583  3584..4095
	     i rows     i00..07     i08..15     i16..23     i24..31     i32..39     i40..47     i48..55     i56..63
	*/

	/*
	   vertical M13:

	     baseM = M13 + e*MH*(I<<2) + kt*(I<<2) + i*256
	     mid   = (warp << 1) + lane01
	     row   = mid
	     dst   = smem + (m << 8) + (row << 4)
	     src   = baseM + (m << 6) + (mid << 2)

	     m-stage   smem bytes    gmem u32     i rows
	     -------   ----------    --------     -------
	     m0        0000..0255    +000..063    i00..15
	     m1        0256..0511    +064..127    i16..31
	     m2        0512..0767    +128..191    i32..47
	     m3        0768..1023    +192..255    i48..63

	     one m-stage:

	     warp.lane   mid   row   i        smem+      gmem u32
	     ---------   ---   ---   -----    -----      --------
	     w0.l0       00    00    16m+00   +000       64m+00..03
	     w0.l1       01    01    16m+01   +016       64m+04..07
	     w1.l0       02    02    16m+02   +032       64m+08..11
	     w1.l1       03    03    16m+03   +048       64m+12..15
	     w2.l0       04    04    16m+04   +064       64m+16..19
	     w2.l1       05    05    16m+05   +080       64m+20..23
	     w3.l0       06    06    16m+06   +096       64m+24..27
	     w3.l1       07    07    16m+07   +112       64m+28..31
	     w4.l0       08    08    16m+08   +128       64m+32..35
	     w4.l1       09    09    16m+09   +144       64m+36..39
	     w5.l0       10    10    16m+10   +160       64m+40..43
	     w5.l1       11    11    16m+11   +176       64m+44..47
	     w6.l0       12    12    16m+12   +192       64m+48..51
	     w6.l1       13    13    16m+13   +208       64m+52..55
	     w7.l0       14    14    16m+14   +224       64m+56..59
	     w7.l1       15    15    16m+15   +240       64m+60..63
	*/

	/*
	   horizontal M13, stage view:

	     m-stage     m0           m1           m2           m3
	     -------     ----------   ----------   ----------   ----------
	     smem bytes  0000..0255   0256..0511   0512..0767   0768..1023
	     gmem u32    +000..063    +064..127    +128..191    +192..255
	     i rows      i00..15      i16..31      i32..47      i48..63
	*/

	/*
	   horizontal M13, one m-stage, lanes 00..07:

	     lane       w0.l0       w0.l1       w1.l0       w1.l1       w2.l0       w2.l1       w3.l0       w3.l1
	     ----       -----       -----       -----       -----       -----       -----       -----       -----
	     mid        00          01          02          03          04          05          06          07
	     row        00          01          02          03          04          05          06          07
	     i          16m+00      16m+01      16m+02      16m+03      16m+04      16m+05      16m+06      16m+07
	     smem+      +000        +016        +032        +048        +064        +080        +096        +112
	     gmem u32   64m+00..03  64m+04..07  64m+08..11  64m+12..15  64m+16..19  64m+20..23  64m+24..27  64m+28..31
	*/

	/*
	   horizontal M13, one m-stage, lanes 08..15:

	     lane       w4.l0       w4.l1       w5.l0       w5.l1       w6.l0       w6.l1       w7.l0       w7.l1
	     ----       -----       -----       -----       -----       -----       -----       -----       -----
	     mid        08          09          10          11          12          13          14          15
	     row        08          09          10          11          12          13          14          15
	     i          16m+08      16m+09      16m+10      16m+11      16m+12      16m+13      16m+14      16m+15
	     smem+      +128        +144        +160        +176        +192        +208        +224        +240
	     gmem u32   64m+32..35  64m+36..39  64m+40..43  64m+44..47  64m+48..51  64m+52..55  64m+56..59  64m+60..63
	*/

	/*
	   vertical S13 raw:

	     baseS = S13 + e*PH*(I<<1) + kt*(I<<1) + i*128
	     mid   = (warp << 1) + lane01
	     row   = mid
	     raw   = smem + 1024 + (s << 8) + (row << 4)
	     src   = baseS + (s << 6) + (mid << 2)

	     s-stage   smem bytes    gmem u32     i rows
	     -------   ----------    --------     -------
	     s0        1024..1279    +000..063    i00..31
	     s1        1280..1535    +064..127    i32..63

	     one s-stage:

	     warp.lane   mid   row   i          smem+      gmem u32
	     ---------   ---   ---   -------    -----      --------
	     w0.l0       00    00    32s+00/01  +000       64s+00..03
	     w0.l1       01    01    32s+02/03  +016       64s+04..07
	     w1.l0       02    02    32s+04/05  +032       64s+08..11
	     w1.l1       03    03    32s+06/07  +048       64s+12..15
	     ...
	     w7.l0       14    14    32s+28/29  +224       64s+56..59
	     w7.l1       15    15    32s+30/31  +240       64s+60..63
	*/

	/*
	   horizontal S13 stage view:

	     s-stage     s0             s1
	     -------     ------------   ------------
	     smem bytes  1024..1279     1280..1535
	     gmem u32    +000..063      +064..127
	     i rows      i00..31        i32..63
	*/

	/*
	   S13 ldmatrix sub-tile transpose:

	     raw smem:
	       row00 = {s1 i00, s3 i00, s1 i01, s3 i01}
	       row01 = {s1 i02, s3 i02, s1 i03, s3 i03}
	       ...
	       row15 = {s1 i30, s3 i30, s1 i31, s3 i31}
	       row16 = {s1 i32, s3 i32, s1 i33, s3 i33}
	       ...
	       row31 = {s1 i62, s3 i62, s1 i63, s3 i63}

	     ldmatrix addr / warp:
	       lane00..07  -> row00..07
	       lane08..15  -> row08..15
	       lane16..23  -> row16..23
	       lane24..31  -> row24..31

	     x4.trans.b16 regs:
	       s13_0 -> rows00..07  transposed 8x8.b16
	       s13_1 -> rows08..15  transposed 8x8.b16
	       s13_2 -> rows16..23  transposed 8x8.b16
	       s13_3 -> rows24..31  transposed 8x8.b16

	     F233 target:
	       row00 = {s1 i00, s1 i32, s3 i00, s3 i32}
	       row01 = {s1 i01, s1 i33, s3 i01, s3 i33}
	       ...
	       row31 = {s1 i31, s1 i63, s3 i31, s3 i63}

	     next:
	       pack s13_0/s13_2 and s13_1/s13_3 into F233 row order
	*/

	/*
	   smem bank board, M13:

	     bank = (smem_byte >> 2) & 31
	     row bytes = 16B = 4 banks

	     row00 -> b00,b01,b02,b03
	     row01 -> b04,b05,b06,b07
	     row02 -> b08,b09,b10,b11
	     row03 -> b12,b13,b14,b15
	     row04 -> b16,b17,b18,b19
	     row05 -> b20,b21,b22,b23
	     row06 -> b24,b25,b26,b27
	     row07 -> b28,b29,b30,b31
	     row08 -> b00,b01,b02,b03
	     ...

	     LDGSTS dst is 16B row-aligned.
	     consumer bank map needs final tmem/rmem handoff board.
	*/

	/*
	   vertical:

	     topk sentinel
	       w0.l0 evict_last
	       sync
	       w1..7.l0 no_allocate
	       sync
	       empty -> return

	     mtbar
	       t0 init mtbar[0]
	       sync

	     loop i=I64, kt=H128
	       W13:
	         8 x lane0 prefetch 512B -> L2
	         total 4096B

	       M13:
	         m0 16 lanes x 16B -> smem+0000
	         m1 16 lanes x 16B -> smem+0256
	         m2 16 lanes x 16B -> smem+0512
	         m3 16 lanes x 16B -> smem+0768
	         total 1024B

	       S13:
	         s0 16 lanes x 16B -> smem+1024
	         s1 16 lanes x 16B -> smem+1280
	         total 512B
	         ldmatrix.x4.trans -> 4 regs/thread

	       sync

	     next:
	       mbar tx / wait
	       M13 smem -> tmem/rmem
	       S13 regs -> F233 pack/store board
	       W13 L2 -> smem/tmem/rmem
	*/
	for (int i = 0; i < (I >> 6); i++) {
			for (int kt = 0; kt < (H >> 7) + 1; kt++) {
					if (!((threadIdx.x ^ 32) & 31)) {
							if (kt) {
								asm volatile(
									"cp.async.bulk.wait_group 0;"
								);
							}
						if (kt < (H >> 7)) {
							asm volatile(
								"cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], 512;\n\t" //512*8=4096 -> TMA optimum
								"cp.async.bulk.commit_group;\n\t"
								::"l"(
									(uint64_t)__cvta_generic_to_global(
										W13
										+ (int64_t)(blockIdx.x * (H >> 7) * (I << 4))
										+ (int64_t)(kt * (I << 4))
										+ (int64_t)((i << 10) + ((threadIdx.x >> 5) << 7))
									)
								)
							);
						}
					}
				if (kt) {
					__syncwarp();
					//TMA L2->smem->rmem with 32B tx across 8 warps with every 2 lanes per warp doing 32B

				}

					if (kt < (H >> 7)) {
						#pragma unroll
						for (int m = 0; m < 4; m++) {
							if (!((((threadIdx.x ^ 32) & 31) >> 1))) { //LDGSTS || TMA
								asm volatile(
									"cp.async.cg.shared::cta.global [%0], [%1], 16;\n\t" //16 lanes * 16B = 256B
									"cp.async.commit_group;\n\t"
									"cp.async.wait_all;\n\t"
									:
									: "r"((uint32)__cvta_generic_to_shared(
											smem
											+ (m << 8)
											+ (((threadIdx.x >> 5) << 5) + ((threadIdx.x & 1) << 4)))),
									  "l"((uint64_t)__cvta_generic_to_global(
											M13
											+ (int64_t)(blockIdx.x * (H >> 6) * (I << 2))
											+ (int64_t)(kt * (I << 2))
											+ (int64_t)((i << 8)
												+ (m << 6)
												+ (((threadIdx.x >> 5) << 3) + ((threadIdx.x & 1) << 2)))))
									: "memory"
								);
							}
						}

						#pragma unroll
						for (int s = 0; s < 2; s++) {
							if (!((((threadIdx.x ^ 32) & 31) >> 1))) {
								asm volatile(
									"cp.async.cg.shared::cta.global [%0], [%1], 16;\n\t" //16 lanes * 16B = 256B
									"cp.async.commit_group;\n\t"
									"cp.async.wait_all;\n\t"
									:
									: "r"((uint32)__cvta_generic_to_shared(
											smem
											+ 1024
											+ (s << 8)
											+ (((threadIdx.x >> 5) << 5) + ((threadIdx.x & 1) << 4)))),
									  "l"((uint64_t)__cvta_generic_to_global(
											S13
											+ (int64_t)(blockIdx.x * (H >> 7) * (I << 1))
											+ (int64_t)(kt * (I << 1))
											+ (int64_t)((i << 7)
												+ (s << 6)
												+ (((threadIdx.x >> 5) << 3) + ((threadIdx.x & 1) << 2)))))
									: "memory"
								);
							}
						}

						__syncthreads();

						uint32 s13_0;
						uint32 s13_1;
						uint32 s13_2;
						uint32 s13_3;

						asm volatile(
							"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n\t"
							: "=r"(s13_0),
							  "=r"(s13_1),
							  "=r"(s13_2),
							  "=r"(s13_3)
							: "r"((uint32)__cvta_generic_to_shared(
									smem
									+ 1024
									+ (((threadIdx.x & 31) >> 3) << 7)
									+ ((threadIdx.x & 7) << 4)))
							: "memory"
						);
					}
				}
			}
}
