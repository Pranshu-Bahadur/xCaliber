#include <cuda_bf16.h>
#include <stdint.h>

using uchar = unsigned char;
using uint16 = uint16_t;
using uint32 = uint32_t;

/*
   sm12x sideband bring-up

   only:
     M13 gmem->smem by TMA-style cp.async.bulk + mbar
     S13 gmem->smem by LDGSTS cp.async

   not here:
     W13
     ldmatrix
     tmem
     mma

   invariant:
     CTA=256
     H % 128 == 0
     I % 64 == 0

   source regime, same as W13:

     W13: [E, H128, I<<4]
       i row = {w1 h000..015, ..., w1 h112..127,
                w3 h000..015, ..., w3 h112..127}

     M13: [E, H128, I<<2]
       i row = {m1 h000..063, m1 h064..127,
                m3 h000..063, m3 h064..127}

     S13: [E, H128, I<<1]
       i row = {s1 h000..127,
                s3 h000..127}

   one i-loop:
     I64 == I128'
     Up I64 + Gate I64

   one kt-loop:
     H128
*/
__global__ void xRR48_sm12x_M13S13_smem(
		const uint32_t* __restrict__ M13, //[E, H >> 7, I << 2]
		const uint32_t* __restrict__ S13, //[E, H >> 7, I << 1]
		const uint16 I,
		const uint16 H
){
	/*
	   smem:

	     0000..1023  M13
	     1024..1535  S13

	     M13:
	       m0 0000..0255 -> i00..15
	       m1 0256..0511 -> i16..31
	       m2 0512..0767 -> i32..47
	       m3 0768..1023 -> i48..63

	     S13:
	       s0 1024..1279 -> i00..31
	       s1 1280..1535 -> i32..63

	     row = 16B
	     bank = (smem_byte >> 2) & 31

	     row00 -> b00,b01,b02,b03
	     row01 -> b04,b05,b06,b07
	     row02 -> b08,b09,b10,b11
	     row03 -> b12,b13,b14,b15
	     row04 -> b16,b17,b18,b19
	     row05 -> b20,b21,b22,b23
	     row06 -> b24,b25,b26,b27
	     row07 -> b28,b29,b30,b31
	     row08 -> b00,b01,b02,b03
	*/
	__shared__ alignas(16) uint64_t mtbar[8];
	__shared__ alignas(16) uchar smem[2048];

	uint32 mbar_M13 = (uint32)__cvta_generic_to_shared(mtbar);
	uint32 parity = 0;

	/*
	   mbar:

	     object:
	       mtbar[0] is 64b in smem

	     pointer:
	       mbar_M13 is 32b shared address
	       all threads compute same pointer

	     phase:
	       expect_tx = 1024B
	       M13 panel = I64 x 16B
	                 = 64 x 16B
	                 = 1024B
	*/
	if (!threadIdx.x) {
		asm volatile(
			"mbarrier.init.layout::v0.shared::cta.b64 [%0], 1;\n\t"
			:
			: "r"(mbar_M13)
			: "memory"
		);
	}

	__syncthreads();

	/*
	   vertical M13 TMA:

	     baseM = M13 + e*PH*(I<<2) + kt*(I<<2) + i64*256
	     mid   = (warp << 1) + lane01
	     src   = baseM + (m << 6) + (mid << 2)
	     dst   = smem + (m << 8) + (mid << 4)

	     m-stage   smem       src u32      i rows
	     -------   ----       -------      ------
	     m0        0000..0255 +000..063    i00..15
	     m1        0256..0511 +064..127    i16..31
	     m2        0512..0767 +128..191    i32..47
	     m3        0768..1023 +192..255    i48..63

	     one m-stage:

	     lane      w0.l0      w0.l1      w1.l0      w1.l1      ... w7.l1
	     ----      -----      -----      -----      -----      --------
	     mid       00         01         02         03         ... 15
	     smem+     +000       +016       +032       +048       ... +240
	     src u32   +00..03    +04..07    +08..11    +12..15    ... +60..63
	     bytes     16B        16B        16B        16B        ... 16B

	     16 active threads x 16B = 256B / m-stage
	     4 m-stages x 256B = 1024B
	*/

	/*
	   vertical S13 LDGSTS:

	     baseS = S13 + e*PH*(I<<1) + kt*(I<<1) + i64*128
	     mid   = (warp << 1) + lane01
	     src   = baseS + (s << 6) + (mid << 2)
	     dst   = smem + 1024 + (s << 8) + (mid << 4)

	     s-stage   smem       src u32      i rows
	     -------   ----       -------      ------
	     s0        1024..1279 +000..063    i00..31
	     s1        1280..1535 +064..127    i32..63

	     one s-stage:

	     lane      w0.l0      w0.l1      w1.l0      w1.l1      ... w7.l1
	     ----      -----      -----      -----      -----      --------
	     mid       00         01         02         03         ... 15
	     smem+     +000       +016       +032       +048       ... +240
	     src u32   +00..03    +04..07    +08..11    +12..15    ... +60..63
	     i rows    32s+00/01  32s+02/03  32s+04/05  32s+06/07  ... 32s+30/31

	     16 active threads x 16B = 256B / s-stage
	     2 s-stages x 256B = 512B
	*/

	/*
	   horizontal panel:

	     payload       M13                 S13
	     -------       ---                 ---
	     source        [E,H128,I<<2]       [E,H128,I<<1]
	     panel u32     256                 128
	     panel bytes   1024                512
	     movement      TMA + mbar          LDGSTS
	     smem          0000..1023          1024..1535
	     consumer      parked              parked
	*/
	for (int i = 0; i < (I >> 6); i++) {
		for (int kt = 0; kt < (H >> 7); kt++) {
			if (!threadIdx.x) {
				asm volatile(
					"mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n\t"
					:
					: "r"(mbar_M13),
					  "r"(1024)
					: "memory"
				);
			}

			__syncthreads();

			#pragma unroll
			for (int m = 0; m < 4; m++) {
				if (!((((threadIdx.x ^ 32) & 31) >> 1))) {
					asm volatile(
						"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::evict_first [%0], [%1], 16, [%2];\n\t"
						:
						: "r"((uint32)__cvta_generic_to_shared(
								smem
								+ (m << 8)
								+ (((threadIdx.x >> 5) << 5) + ((threadIdx.x & 1) << 4)))),
						  "l"((uint64_t)__cvta_generic_to_global(
								M13
								+ (int64_t)(blockIdx.x * (H >> 7) * (I << 2))
								+ (int64_t)(kt * (I << 2))
								+ (int64_t)((i << 8)
									+ (m << 6)
									+ (((threadIdx.x >> 5) << 3) + ((threadIdx.x & 1) << 2))))),
						  "r"(mbar_M13)
						: "memory"
					);
				}
			}

			#pragma unroll
			for (int s = 0; s < 2; s++) {
				if (!((((threadIdx.x ^ 32) & 31) >> 1))) {
					asm volatile(
						"cp.async.cg.shared::cta.global [%0], [%1], 16;\n\t"
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

			asm volatile(
				"{\n\t"
				".reg .pred p;\n\t"
				"wait_%=:\n\t"
				"mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n\t"
				"@!p bra wait_%=;\n\t"
				"}\n\t"
				:
				: "r"(mbar_M13),
				  "r"(parity)
				: "memory"
			);

			parity ^= 1;

			__syncthreads();
		}
	}
}
