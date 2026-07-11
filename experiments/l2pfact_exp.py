#!/usr/bin/env python3
"""
kernel.cu activation-stream simulator for RTX PRO 6000 / sm_120.

One script owns CUDA source, build, run, rectangular CSV validation, and an
ASCII ranking.  PTX forms follow .vscode/ptx_isa_9.3.pdf.  Paths stop at the
stream, scaled weight-dequant, or activation-compression sink.  No MMA or
output reduction is modeled here.

Colab CLI run:
  colab exec -s xCalibrr -f experiments/l2pfact_exp.py --timeout 3600

The ipykernel `-f` launch selects the full N={8,256,512} sweep automatically.

Local smoke run:
  python3 experiments/l2pfact_exp.py --quick

Focused local run:
  python3 experiments/l2pfact_exp.py --paths pure \
      --methods no_pf_direct,pf64x4_current_lead1_direct,pf64x4_current_lead1_tma_smem
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path.cwd()
COLAB_SWEEP_PROFILE = "full"


CUDA_SRC = r'''
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                \
    cudaError_t e__ = (call);                                                \
    if (e__ != cudaSuccess) {                                                \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                     cudaGetErrorString(e__));                               \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

static constexpr uint32_t CTA = 1024;
static constexpr uint32_t TOPK_LOCK = 8;
static constexpr uint32_t W_BUNDLE_U32 = 16384; // 4 x 16KB W13 planes.
static constexpr uint32_t S_BUNDLE_U32 = 2048;  // 4 x 2KB S13 planes.
static constexpr uint32_t ACT_RAW_BYTES = 32768;    // N256 x K64 x BF16.
static constexpr uint32_t ACT_PACKED_BYTES = 9216;  // N256 x (X4 32B + SX 4B).

enum Path : int {
    PURE = 0,
    WDEQUANT = 1,
    ACTCOMP = 2,
    PATH_COUNT = 3,
};

enum Method : int {
    NO_PF_DIRECT = 0,
    PF128_CURRENT_DIRECT = 1,
    PF64X2_CURRENT_DIRECT = 2,
    PF128X2_CURRENT_LEAD1_DIRECT = 3,
    PF64X4_CURRENT_LEAD1_DIRECT = 4,
    PF64X4_CURRENT_LEAD1_WAIT0_DIRECT = 5,
    NO_PF_TMA_SMEM = 6,
    PF128_CURRENT_TMA_SMEM = 7,
    PF64X2_CURRENT_TMA_SMEM = 8,
    PF128X2_CURRENT_LEAD1_TMA_SMEM = 9,
    PF64X4_CURRENT_LEAD1_TMA_SMEM = 10,
    PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM = 11,
    PF32X4_CURRENT_DIRECT = 12,
    PF32X4_CURRENT_WAIT0_DIRECT = 13,
    PF32X4_CURRENT_TMA_SMEM = 14,
    PF32X4_CURRENT_WAIT0_TMA_SMEM = 15,
    NO_PF_CTA_NC_ACTCOMP_SMEM = 16,
    PF128_CURRENT_CTA_NC_ACTCOMP_SMEM = 17,
    PF32X4_CURRENT_CTA_NC_ACTCOMP_SMEM = 18,
    PF32X4_CURRENT_WAIT0_CTA_NC_ACTCOMP_SMEM = 19,
    METHOD_COUNT = 20,
};

__host__ __device__ constexpr bool method_is_tma(int method) {
    return (method >= NO_PF_TMA_SMEM && method <= PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM) ||
           method == PF32X4_CURRENT_TMA_SMEM ||
           method == PF32X4_CURRENT_WAIT0_TMA_SMEM;
}

__host__ __device__ constexpr bool method_is_cta_nc_actcomp(int method) {
    return method >= NO_PF_CTA_NC_ACTCOMP_SMEM;
}

__host__ __device__ constexpr bool method_has_prefetch(int method) {
    return method != NO_PF_DIRECT && method != NO_PF_TMA_SMEM &&
           method != NO_PF_CTA_NC_ACTCOMP_SMEM;
}

static const char* path_name(int path) {
    switch (path) {
    case PURE: return "pure";
    case WDEQUANT: return "wdequant";
    case ACTCOMP: return "actcomp";
    default: return "unknown";
    }
}

static const char* method_name(int method) {
    switch (method) {
    case NO_PF_DIRECT: return "no_pf_direct";
    case PF128_CURRENT_DIRECT: return "pf128_current_direct";
    case PF64X2_CURRENT_DIRECT: return "pf64x2_current_direct";
    case PF128X2_CURRENT_LEAD1_DIRECT: return "pf128x2_current_lead1_direct";
    case PF64X4_CURRENT_LEAD1_DIRECT: return "pf64x4_current_lead1_direct";
    case PF64X4_CURRENT_LEAD1_WAIT0_DIRECT: return "pf64x4_current_wait0_lead1_async_direct";
    case NO_PF_TMA_SMEM: return "no_pf_tma_smem";
    case PF128_CURRENT_TMA_SMEM: return "pf128_current_tma_smem";
    case PF64X2_CURRENT_TMA_SMEM: return "pf64x2_current_tma_smem";
    case PF128X2_CURRENT_LEAD1_TMA_SMEM: return "pf128x2_current_lead1_tma_smem";
    case PF64X4_CURRENT_LEAD1_TMA_SMEM: return "pf64x4_current_lead1_tma_smem";
    case PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM: return "pf64x4_current_wait0_lead1_async_tma_smem";
    case PF32X4_CURRENT_DIRECT: return "pf32x4_current_direct";
    case PF32X4_CURRENT_WAIT0_DIRECT: return "pf32x4_current_wait0_direct";
    case PF32X4_CURRENT_TMA_SMEM: return "pf32x4_current_tma_smem";
    case PF32X4_CURRENT_WAIT0_TMA_SMEM: return "pf32x4_current_wait0_tma_smem";
    case NO_PF_CTA_NC_ACTCOMP_SMEM: return "no_pf_cta_nc_actcomp_smem";
    case PF128_CURRENT_CTA_NC_ACTCOMP_SMEM: return "pf128_current_cta_nc_actcomp_smem";
    case PF32X4_CURRENT_CTA_NC_ACTCOMP_SMEM: return "pf32x4_current_cta_nc_actcomp_smem";
    case PF32X4_CURRENT_WAIT0_CTA_NC_ACTCOMP_SMEM: return "pf32x4_current_wait0_cta_nc_actcomp_smem";
    default: return "unknown";
    }
}

static const char* transport_name(int method) {
    if (method_is_cta_nc_actcomp(method)) return "cta_nc";
    if (method_is_tma(method)) return "tma";
    return "warp_direct";
}

static const char* pipeline_name(int path, int method) {
    if (method_is_cta_nc_actcomp(method)) return "cta_nc_rmem_actcomp_packed_smem";
    if (method_is_tma(method) && path == ACTCOMP) return "tma_raw_smem_cta_actcomp_packed_smem";
    if (method_is_tma(method)) return "tma_raw_smem_warp32_shared_consumer";
    if (path == ACTCOMP) return "warp32_global_actcomp_sink_diagnostic";
    return "warp32_global_consumer";
}

static const char* comparison_name(int path, int method) {
    if (path == ACTCOMP && !method_is_tma(method) && !method_is_cta_nc_actcomp(method)) {
        return "actcomp_warp_diagnostic";
    }
    if (path == ACTCOMP) return "actcomp_packed_smem_producer";
    return "warp_consumer";
}

static const char* destination_name(int path, int method) {
    if (path == ACTCOMP &&
        (method_is_tma(method) || method_is_cta_nc_actcomp(method))) {
        return "packed_smem";
    }
    if (method_is_tma(method)) return "smem_then_rmem";
    return "rmem_sink";
}

static const char* pf_shape_name(int method) {
    switch (method) {
    case NO_PF_DIRECT:
    case NO_PF_TMA_SMEM:
        return "none";
    case PF128_CURRENT_DIRECT:
    case PF128_CURRENT_TMA_SMEM:
        return "1x128B_current";
    case PF64X2_CURRENT_DIRECT:
    case PF64X2_CURRENT_TMA_SMEM:
        return "2x64B_current";
    case PF128X2_CURRENT_LEAD1_DIRECT:
    case PF128X2_CURRENT_LEAD1_TMA_SMEM:
        return "2x128B_current_lead1";
    case PF64X4_CURRENT_LEAD1_DIRECT:
    case PF64X4_CURRENT_LEAD1_TMA_SMEM:
        return "4x64B_current_lead1";
    case PF64X4_CURRENT_LEAD1_WAIT0_DIRECT:
    case PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM:
        return "4x64B_current_wait0_lead1_async";
    case PF32X4_CURRENT_DIRECT:
    case PF32X4_CURRENT_TMA_SMEM:
    case PF32X4_CURRENT_CTA_NC_ACTCOMP_SMEM:
        return "4x32B_current_thread_groups";
    case PF32X4_CURRENT_WAIT0_DIRECT:
    case PF32X4_CURRENT_WAIT0_TMA_SMEM:
    case PF32X4_CURRENT_WAIT0_CTA_NC_ACTCOMP_SMEM:
        return "4x32B_current_thread_groups_wait0";
    case NO_PF_CTA_NC_ACTCOMP_SMEM:
        return "none";
    case PF128_CURRENT_CTA_NC_ACTCOMP_SMEM:
        return "1x128B_current";
    default:
        return "unknown";
    }
}

__host__ __device__ __forceinline__ uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t x) {
    return __uint_as_float(uint32_t(x) << 16);
}

__device__ __forceinline__ void load_W13(
    uint32_t* rmem,
    const uint32_t* W13
) {
    asm volatile(
        "ld.global.ca.nc.L2::64B.v4.u32 {%0,%1,%2,%3}, [%16];\n\t"
        "ld.global.nc.L1::evict_first.v4.u32 {%4,%5,%6,%7}, [%16 + 16384];\n\t"
        "ld.global.nc.L1::evict_first.v4.u32 {%8,%9,%10,%11}, [%16 + 32768];\n\t"
        "ld.global.nc.L1::evict_first.v4.u32 {%12,%13,%14,%15}, [%16 + 49152];\n\t"
        : "=r"(rmem[1]), "=r"(rmem[2]), "=r"(rmem[3]), "=r"(rmem[4]),
          "=r"(rmem[5]), "=r"(rmem[6]), "=r"(rmem[7]), "=r"(rmem[8]),
          "=r"(rmem[9]), "=r"(rmem[10]), "=r"(rmem[11]), "=r"(rmem[12]),
          "=r"(rmem[13]), "=r"(rmem[14]), "=r"(rmem[15]), "=r"(rmem[16])
        : "l"((uint64_t)__cvta_generic_to_global(W13))
        : "memory");
}

__device__ __forceinline__ void load_S13(
    uint32_t* rmem,
    const uint32_t* S13
) {
    rmem[17] = rmem[18] = rmem[19] = rmem[20] = 0;
    rmem[21] = rmem[22] = rmem[23] = rmem[24] = 0;

    if (!(threadIdx.x & 3)) {
        asm volatile(
            "ld.global.ca.nc.v2.u32 {%0,%1}, [%8];\n\t"
            "ld.global.ca.nc.v2.u32 {%2,%3}, [%8 + 2048];\n\t"
            "ld.global.ca.nc.v2.u32 {%4,%5}, [%8 + 4096];\n\t"
            "ld.global.ca.nc.v2.u32 {%6,%7}, [%8 + 6144];\n\t"
            : "=r"(rmem[17]), "=r"(rmem[18]),
              "=r"(rmem[19]), "=r"(rmem[20]),
              "=r"(rmem[21]), "=r"(rmem[22]),
              "=r"(rmem[23]), "=r"(rmem[24])
            : "l"((uint64_t)__cvta_generic_to_global(S13))
            : "memory");
    }

    const int src = int(threadIdx.x & 31u) & ~3;
    for (int j = 17; j < 25; ++j) {
        rmem[j] = __shfl_sync(0xffffffffu, rmem[j], src);
    }
}

__device__ __forceinline__ uint4 load_X_global(const void* p) {
    uint4 x;
    asm volatile(
        "ld.global.cs.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
        : "l"((uint64_t)__cvta_generic_to_global(p))
        : "memory");
    return x;
}

__device__ __forceinline__ uint4 load_X_shared(const void* p) {
    uint4 x;
    asm volatile(
        "ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
        : "r"((uint32_t)__cvta_generic_to_shared(p))
        : "memory");
    return x;
}

template <int BYTES>
__device__ __forceinline__ void prefetch_X(const void* p) {
    const uint32_t bytes = BYTES;
    asm volatile(
        "cp.async.bulk.prefetch.L2.global [%0], %1;\n\t"
        "cp.async.bulk.commit_group;"
        :
        : "l"((uint64_t)__cvta_generic_to_global(p)), "r"(bytes)
        : "memory");
}

__device__ __forceinline__ void prefetch_wait0() {
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;\n\t"
        "fence.proxy.async.shared::cta;"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory");
}

__device__ __forceinline__ uint64_t mbarrier_expect_tx(uint64_t* mbar, uint32_t bytes) {
    uint64_t state;
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(bytes)
        : "memory");
    return state;
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint64_t state) {
    asm volatile(
        "{\n\t"
        ".reg .pred done;\n\t"
        "wait_%=:\n\t"
        "mbarrier.test_wait.shared::cta.b64 done, [%0], %1;\n\t"
        "@!done bra wait_%=;\n\t"
        "}"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(mbar)), "l"(state)
        : "memory");
}

__device__ __forceinline__ void tma_X_128B(void* smem, const void* X, uint64_t* mbar) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], 128, [%2];"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(smem)),
          "l"((uint64_t)__cvta_generic_to_global(X)),
          "r"((uint32_t)__cvta_generic_to_shared(mbar))
        : "memory");
}

__device__ __forceinline__ void e2m1x8_to_bf16x8_neg(
    uint32_t packed,
    uint32_t& bf01,
    uint32_t& bf23,
    uint32_t& bf45,
    uint32_t& bf67
) {
    asm volatile(
        "{\n\t"
        ".reg .b32 x, wt, ws, t1, t2, t3, out45, out67;\n\t"
        "mov.b32 x, %4;\n\t"
        "lop3.b32 ws, x, 0x00080008, 0x00080008, 0x6A;\n\t"
        "shl.b32 ws, ws, 12;\n\t"
        "shl.b32 wt, x, 6;\n\t"
        "lop3.b32 wt, wt, 0x01C001C0, ws, 0xEA;\n\t"
        "lop3.b32 ws, x, 0x80008000, 0x80008000, 0x6A;\n\t"
        "shr.b32 t3, x, 6;\n\t"
        "lop3.b32 t3, t3, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 x, x, 4;\n\t"
        "lop3.b32 ws, x, 0x80008000, 0x80008000, 0x6A;\n\t"
        "shr.b32 t2, x, 6;\n\t"
        "lop3.b32 t2, t2, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 x, x, 4;\n\t"
        "lop3.b32 ws, x, 0x80008000, 0x80008000, 0x6A;\n\t"
        "shr.b32 t1, x, 6;\n\t"
        "lop3.b32 t1, t1, 0x01C001C0, ws, 0xEA;\n\t"
        "shl.b32 ws, t1, 16;\n\t"
        "lop3.b32 %0, ws, 0xFFFF0000, wt, 0xE2;\n\t"
        "shr.b32 wt, wt, 16;\n\t"
        "lop3.b32 out45, t1, 0xFFFF0000, wt, 0xE2;\n\t"
        "shr.b32 ws, t2, 16;\n\t"
        "lop3.b32 out67, t3, 0xFFFF0000, ws, 0xE2;\n\t"
        "shl.b32 %1, t3, 16;\n\t"
        "lop3.b32 %1, %1, 0xFFFF0000, t2, 0xE2;\n\t"
        "and.b32 %2, out45, 0x81C081C0;\n\t"
        "and.b32 %3, out67, 0x81C081C0;\n\t"
        "}\n\t"
        : "=r"(bf01), "=r"(bf23), "=r"(bf45), "=r"(bf67)
        : "r"(packed));
}

__device__ __forceinline__ uint32_t max_u32(uint32_t a, uint32_t b) {
    return a > b ? a : b;
}

__device__ __forceinline__ uint32_t bf16_pair_absmax(uint32_t x) {
    return max_u32(x & 0x7fffu, (x >> 16) & 0x7fffu);
}

__device__ __forceinline__ uint32_t uint4_absmax(uint4 x) {
    return max_u32(max_u32(bf16_pair_absmax(x.x), bf16_pair_absmax(x.y)),
                   max_u32(bf16_pair_absmax(x.z), bf16_pair_absmax(x.w)));
}

__device__ __forceinline__ uint32_t ue4m3_from_absmax(uint32_t mx) {
    if (!mx) return 0;
    uint32_t e = (mx >> 7) & 0xffu;
    uint32_t m = (mx >> 4) & 7u;
    e = e > 120u ? e - 120u : 0u;
    return ((e > 15u ? 15u : e) << 3) | m;
}

__device__ __forceinline__ uint32_t q_e2m1(uint32_t h, uint32_t mx) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t mag = h & 0x7fffu;
    if (!mag || !mx) return sign << 3;
    int rel = int((mag >> 7) & 0xffu) - int((mx >> 7) & 0xffu) + 3;
    rel = rel < 0 ? 0 : (rel > 3 ? 3 : rel);
    return (sign << 3) | (uint32_t(rel) << 1) | ((mag >> 6) & 1u);
}

__device__ __forceinline__ uint32_t pack_e2m1x8(uint4 x, uint32_t mx) {
    const uint32_t h0 = x.x & 0xffffu;
    const uint32_t h1 = x.x >> 16;
    const uint32_t h2 = x.y & 0xffffu;
    const uint32_t h3 = x.y >> 16;
    const uint32_t h4 = x.z & 0xffffu;
    const uint32_t h5 = x.z >> 16;
    const uint32_t h6 = x.w & 0xffffu;
    const uint32_t h7 = x.w >> 16;
    return (q_e2m1(h0, mx) << 0) | (q_e2m1(h1, mx) << 4) |
           (q_e2m1(h2, mx) << 8) | (q_e2m1(h3, mx) << 12) |
           (q_e2m1(h4, mx) << 16) | (q_e2m1(h5, mx) << 20) |
           (q_e2m1(h6, mx) << 24) | (q_e2m1(h7, mx) << 28);
}

template <int METHOD>
__device__ __forceinline__ void issue_X_prefetch(
    const __nv_bfloat16* X,
    const uint32_t* Xb_s,
    uint32_t n256,
    uint32_t N,
    uint32_t kt,
    uint32_t H,
    uint32_t kt_count
) {
    if constexpr (!method_has_prefetch(METHOD)) {
        return;
    }

    const uint32_t rank = uint32_t(threadIdx.x) & 255u;
    const uint32_t band = uint32_t(threadIdx.x) >> 8;
    const uint32_t token = (n256 << 8) + rank;
    if (token >= N) return;
    const uint32_t bits = Xb_s[(n256 << 3) + (rank >> 5)];
    if (!((bits >> (rank & 31u)) & 1u)) return;

    const char* current = reinterpret_cast<const char*>(X + uint64_t(token) * H + uint64_t(kt) * 64u);
    const char* lead1 = kt + 1u < kt_count
        ? reinterpret_cast<const char*>(X + uint64_t(token) * H + uint64_t(kt + 1u) * 64u)
        : nullptr;

    if constexpr (METHOD == PF128_CURRENT_DIRECT || METHOD == PF128_CURRENT_TMA_SMEM ||
                  METHOD == PF128_CURRENT_CTA_NC_ACTCOMP_SMEM) {
        if (band == 0u) prefetch_X<128>(current);
    }
    if constexpr (METHOD == PF64X2_CURRENT_DIRECT || METHOD == PF64X2_CURRENT_TMA_SMEM) {
        if (band == 0u) prefetch_X<64>(current);
        if (band == 1u) prefetch_X<64>(current + 64);
    }
    if constexpr (METHOD == PF128X2_CURRENT_LEAD1_DIRECT || METHOD == PF128X2_CURRENT_LEAD1_TMA_SMEM) {
        if (band == 0u) prefetch_X<128>(current);
        if (band == 1u && lead1) prefetch_X<128>(lead1);
    }
    if constexpr (METHOD == PF64X4_CURRENT_LEAD1_DIRECT ||
                  METHOD == PF64X4_CURRENT_LEAD1_WAIT0_DIRECT ||
                  METHOD == PF64X4_CURRENT_LEAD1_TMA_SMEM ||
                  METHOD == PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM) {
        if (band == 0u) prefetch_X<64>(current);
        if (band == 1u) prefetch_X<64>(current + 64);
        if (band == 2u && lead1) prefetch_X<64>(lead1);
        if (band == 3u && lead1) prefetch_X<64>(lead1 + 64);
    }

    if constexpr (METHOD == PF64X4_CURRENT_LEAD1_WAIT0_DIRECT ||
                  METHOD == PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM) {
        if (band < 2u) prefetch_wait0();
    }

    if constexpr (METHOD == PF32X4_CURRENT_DIRECT ||
                  METHOD == PF32X4_CURRENT_WAIT0_DIRECT ||
                  METHOD == PF32X4_CURRENT_TMA_SMEM ||
                  METHOD == PF32X4_CURRENT_WAIT0_TMA_SMEM ||
                  METHOD == PF32X4_CURRENT_CTA_NC_ACTCOMP_SMEM ||
                  METHOD == PF32X4_CURRENT_WAIT0_CTA_NC_ACTCOMP_SMEM) {
        prefetch_X<32>(current + band * 32u);
    }

    if constexpr (METHOD == PF32X4_CURRENT_WAIT0_DIRECT ||
                  METHOD == PF32X4_CURRENT_WAIT0_TMA_SMEM ||
                  METHOD == PF32X4_CURRENT_WAIT0_CTA_NC_ACTCOMP_SMEM) {
        prefetch_wait0();
    }
}

__device__ __forceinline__ void actcomp_pack(
    uint4 x0,
    uint4 x1,
    uint32_t& b0,
    uint32_t& b1,
    uint32_t& scaleB
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    uint32_t mx0 = uint4_absmax(x0);
    uint32_t mx1 = uint4_absmax(x1);
    mx0 = max_u32(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx1 = max_u32(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    b0 = pack_e2m1x8(x0, mx0);
    b1 = pack_e2m1x8(x1, mx1);
    const int quad = int(lane & ~3u);
    scaleB =
        (ue4m3_from_absmax(__shfl_sync(0xffffffffu, mx0, quad + 0)) << 0) |
        (ue4m3_from_absmax(__shfl_sync(0xffffffffu, mx0, quad + 2)) << 8) |
        (ue4m3_from_absmax(__shfl_sync(0xffffffffu, mx1, quad + 0)) << 16) |
        (ue4m3_from_absmax(__shfl_sync(0xffffffffu, mx1, quad + 2)) << 24);
}

template <int PATH>
__device__ __forceinline__ void path_sink(
    uint32_t* rmem,
    float& path_acc,
    uint32_t& aux,
    float W13GS
) {
    if constexpr (PATH == PURE) {
        for (int j = 25; j < 33; ++j) aux = mix32(aux ^ rmem[j]);
    }

    if constexpr (PATH == WDEQUANT) {
        for (int j = 25; j < 33; ++j) aux = mix32(aux ^ rmem[j]);
        for (int wp = 0; wp < 4; ++wp) {
            for (int k16 = 0; k16 < 4; ++k16) {
                uint32_t a0, a1, a2, a3;
                e2m1x8_to_bf16x8_neg(rmem[1 + wp * 4 + k16], a0, a1, a2, a3);
                const uint32_t s = rmem[17 + wp * 2 + (k16 >> 1)];
                const float sf = (1.0f + float((s >> ((k16 & 1) * 8)) & 0xffu)) * (1.0f / 256.0f);
                const float packet =
                    bf16_bits_to_f32(uint16_t(a0)) + bf16_bits_to_f32(uint16_t(a0 >> 16)) +
                    bf16_bits_to_f32(uint16_t(a1)) + bf16_bits_to_f32(uint16_t(a1 >> 16)) +
                    bf16_bits_to_f32(uint16_t(a2)) + bf16_bits_to_f32(uint16_t(a2 >> 16)) +
                    bf16_bits_to_f32(uint16_t(a3)) + bf16_bits_to_f32(uint16_t(a3 >> 16));
                path_acc += packet * sf * W13GS;
                aux ^= a0 ^ a1 ^ a2 ^ a3 ^ s;
            }
        }
    }

    if constexpr (PATH == ACTCOMP) {
        const uint4 x0 = make_uint4(rmem[25], rmem[26], rmem[27], rmem[28]);
        const uint4 x1 = make_uint4(rmem[29], rmem[30], rmem[31], rmem[32]);
        uint32_t b0, b1, scaleB;
        actcomp_pack(x0, x1, b0, b1, scaleB);
        aux = mix32(aux ^ b0 ^ b1 ^ scaleB);
    }
}

template <int PATH, int METHOD>
__global__ __launch_bounds__(CTA, 1)
void kernel(
    const uint32_t* __restrict__ W13,
    const uint32_t* __restrict__ S13,
    const __nv_bfloat16* __restrict__ W13GS,
    const __nv_bfloat16* __restrict__ X,
    const uint16_t XGSINV,
    const int32_t* __restrict__ Xb,
    const __nv_bfloat16* __restrict__ topk_W,
    uint32_t* __restrict__ Y,
    uint64_t* __restrict__ cycles,
    const int32_t E,
    const int32_t N,
    const int32_t I,
    const int32_t H,
    const int32_t TOPK
) {
    uint32_t rmem[48];
    __shared__ alignas(16) unsigned char smem[ACT_RAW_BYTES + ACT_PACKED_BYTES];
    __shared__ alignas(16) uint64_t mbar;
    __shared__ uint64_t mstate;
    __shared__ uint32_t Xb_s[16];
    __shared__ uint32_t Xb_plane_count[2];

    const uint32_t e = uint32_t(blockIdx.x);
    const uint32_t t = uint32_t(threadIdx.x);
    const uint32_t w = t >> 5;
    const uint32_t l = t & 31u;
    const uint32_t g = l >> 2;
    const uint32_t p = l & 3u;
    const uint32_t Xb_stride = 1u + ((uint32_t(N) + 31u) >> 5);

    if (e >= uint32_t(E)) return;
    asm volatile("ldu.global.u32 %0, [%1];"
                 : "=r"(rmem[0])
                 : "l"((uint64_t)__cvta_generic_to_global(Xb + uint64_t(e) * Xb_stride))
                 : "memory");
    if (!rmem[0]) return;

    uint64_t start = 0;
    if (t == 0) start = clock64();
    if constexpr (method_is_tma(METHOD)) {
        if (t == 0) mbarrier_init(&mbar);
        __syncthreads();
    }

    if (t < 16u) {
        Xb_s[t] = t < ((uint32_t(N) + 31u) >> 5)
            ? uint32_t(Xb[uint64_t(e) * Xb_stride + 1u + t])
            : 0u;
    }
    __syncthreads();
    if (t == 0) {
        uint32_t count = 0;
        for (int j = 0; j < 16; ++j) {
            count += __popc(Xb_s[j]);
        }
        if (count != rmem[0]) __trap();
    }
    if (t < 2u) {
        uint32_t count = 0;
        for (int j = 0; j < 8; ++j) count += __popc(Xb_s[(t << 3) + j]);
        Xb_plane_count[t] = count;
    }
    __syncthreads();

    const uint32_t kt_count = uint32_t(H) >> 6;
    const uint32_t i_count = uint32_t(I) >> 9;
    const float Gw = bf16_bits_to_f32(reinterpret_cast<const uint16_t*>(W13GS)[uint64_t(e) * 2u + (g & 1u)]);
    float path_acc = 0.0f;
    uint32_t aux = mix32(t ^ (e << 16) ^ uint32_t(XGSINV));
    (void)w;
    (void)topk_W;
    (void)TOPK;

    for (uint32_t kt = 0; kt < kt_count; ++kt) {
        for (uint32_t i = 0; i < i_count; ++i) {
            const uint32_t* W13_i = W13
                + uint64_t(e) * kt_count * (uint32_t(I) << 5)
                + uint64_t(kt) * (uint32_t(I) << 5)
                + uint64_t(i) * W_BUNDLE_U32
                + (t << 2);
            const uint32_t* S13_i = S13
                + uint64_t(e) * kt_count * (uint32_t(I) << 2)
                + uint64_t(kt) * (uint32_t(I) << 2)
                + uint64_t(i) * S_BUNDLE_U32
                + ((t >> 2) << 1);

            load_W13(rmem, W13_i);
            load_S13(rmem, S13_i);

            for (int j = 1; j < 25; ++j) aux = mix32(aux ^ rmem[j]);

            for (uint32_t n256 = 0; n256 < ((uint32_t(N) + 255u) >> 8); ++n256) {
                if (!Xb_plane_count[n256]) continue;

                // b=t>>8 selects one 32B K64 subpanel; q=t&255 stays fixed.
                issue_X_prefetch<METHOD>(
                    X, Xb_s, n256, uint32_t(N), kt, uint32_t(H), kt_count);
                __syncthreads();

                if constexpr (method_is_tma(METHOD)) {
                    if (t == 0) {
                        mstate = mbarrier_expect_tx(&mbar, Xb_plane_count[n256] * 128u);
                    }
                    __syncthreads();
                    if (t < 256u) {
                        const uint32_t bits = Xb_s[(n256 << 3) + (t >> 5)];
                        if ((bits >> (t & 31u)) & 1u) {
                            const uint32_t token = (n256 << 8) + t;
                            tma_X_128B(smem + t * 128u,
                                       X + uint64_t(token) * uint32_t(H) + uint64_t(kt) * 64u,
                                       &mbar);
                        }
                    }
                    if (t == 0) mbarrier_wait(&mbar, mstate);
                    __syncthreads();
                }

                if constexpr (PATH == ACTCOMP &&
                              (method_is_tma(METHOD) || method_is_cta_nc_actcomp(METHOD))) {
                    // CTA-wide producer: 256 quads x 4T load one K64/token exactly once.
                    const uint32_t rank = t >> 2;
                    const uint32_t pp = t & 3u;
                    const uint32_t bits = Xb_s[(n256 << 3) + (rank >> 5)];
                    const uint32_t live = (bits >> (rank & 31u)) & 1u;
                    uint4 x0 = make_uint4(0, 0, 0, 0);
                    uint4 x1 = make_uint4(0, 0, 0, 0);
                    if (live) {
                        if constexpr (method_is_tma(METHOD)) {
                            const char* Xrow = reinterpret_cast<const char*>(smem + rank * 128u);
                            x0 = load_X_shared(Xrow + pp * 16u);
                            x1 = load_X_shared(Xrow + 64u + pp * 16u);
                        } else {
                            const uint32_t token = (n256 << 8) + rank;
                            const char* Xrow = reinterpret_cast<const char*>(
                                X + uint64_t(token) * uint32_t(H) + uint64_t(kt) * 64u);
                            x0 = load_X_global(Xrow + pp * 16u);
                            x1 = load_X_global(Xrow + 64u + pp * 16u);
                        }
                    }

                    uint32_t b0, b1, scaleB;
                    actcomp_pack(x0, x1, b0, b1, scaleB);
                    volatile uint32_t* packed = reinterpret_cast<volatile uint32_t*>(
                        smem + ACT_RAW_BYTES);
                    if (live) {
                        const uint32_t frag_lane = ((rank & 7u) << 2) + pp;
                        packed[(rank >> 3) * 64u + frag_lane] = b0;
                        packed[(rank >> 3) * 64u + 32u + frag_lane] = b1;
                        if (pp == 0u) packed[2048u + rank] = scaleB;
                    }
                    __syncthreads();
                    if (live) {
                        const uint32_t frag_lane = ((rank & 7u) << 2) + pp;
                        const uint32_t published =
                            packed[(rank >> 3) * 64u + frag_lane] ^
                            packed[(rank >> 3) * 64u + 32u + frag_lane] ^
                            (pp == 0u ? packed[2048u + rank] : 0u);
                        aux = mix32(aux ^ published);
                    }
                    __syncthreads();
                } else {
                    const uint32_t plane_tokens = uint32_t(N) - (n256 << 8) < 256u
                        ? uint32_t(N) - (n256 << 8) : 256u;
                    for (uint32_t n8 = 0; n8 < ((plane_tokens + 7u) >> 3); ++n8) {
                        const uint32_t active8 =
                            (Xb_s[(n256 << 3) + (n8 >> 2)] >> ((n8 & 3u) << 3)) & 0xffu;
                        if (!active8) continue;
                        const uint32_t rank = (n8 << 3) + g;
                        const uint32_t live = (active8 >> g) & 1u;
                        const uint32_t token = (n256 << 8) + rank;

                        uint4 x0 = make_uint4(0, 0, 0, 0);
                        uint4 x1 = make_uint4(0, 0, 0, 0);
                        if (live) {
                            if constexpr (method_is_tma(METHOD)) {
                                const char* Xrow = reinterpret_cast<const char*>(smem + rank * 128u);
                                x0 = load_X_shared(Xrow + p * 16u);
                                x1 = load_X_shared(Xrow + 64u + p * 16u);
                            } else {
                                const char* Xrow = reinterpret_cast<const char*>(
                                    X + uint64_t(token) * uint32_t(H) + uint64_t(kt) * 64u);
                                x0 = load_X_global(Xrow + p * 16u);
                                x1 = load_X_global(Xrow + 64u + p * 16u);
                            }
                        }

                        rmem[25] = x0.x; rmem[26] = x0.y; rmem[27] = x0.z; rmem[28] = x0.w;
                        rmem[29] = x1.x; rmem[30] = x1.y; rmem[31] = x1.z; rmem[32] = x1.w;
                        path_sink<PATH>(rmem, path_acc, aux, Gw);
                    }
                    if constexpr (method_is_tma(METHOD)) __syncthreads();
                }
            }
        }
    }

    aux ^= __float_as_uint(path_acc);
    Y[uint64_t(e) * CTA + t] = aux;
    if (t == 0) cycles[e] = clock64() - start;
}

__global__ void init_u32(uint32_t* p, uint64_t n, uint32_t seed) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) p[x] = mix32(uint32_t(x) ^ seed);
}

__global__ void init_bf16(uint16_t* p, uint64_t n, uint32_t seed) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t z = mix32(uint32_t(x) ^ seed);
        p[x] = uint16_t(0x3f00u | (z & 0x007fu) | ((z >> 16) & 0x8000u));
    }
}

__global__ void flush_l2(uint32_t* p, uint64_t n) {
    uint64_t x = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blockDim.x) * gridDim.x;
    for (; x < n; x += stride) p[x] = p[x] * 1664525u + 1013904223u;
}

struct Config {
    uint32_t E = 384;
    uint32_t N = 128;
    uint32_t I = 1024;
    uint32_t H = 1024;
    uint32_t TOPK = 8;
    uint32_t active_experts = 384;
    uint32_t route_seed = 0x6a09e667u;
    uint32_t repeats = 3;
    uint32_t flush_mb = 512;
    std::string paths = "pure,wdequant,actcomp";
    std::string methods = "all";
    const char* csv_path = "kernel_sim_results.csv";
};

static uint32_t u32(const char* s) {
    char* end = nullptr;
    unsigned long x = std::strtoul(s, &end, 0);
    if (!end || *end) { std::fprintf(stderr, "bad integer: %s\n", s); std::exit(2); }
    return uint32_t(x);
}

static Config parse_args(int argc, char** argv) {
    Config c;
    for (int a = 1; a < argc; ++a) {
        auto need = [&](const char* flag) {
            if (a + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(2); }
            return argv[++a];
        };
        if (!std::strcmp(argv[a], "--E")) c.E = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--N")) c.N = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--I")) c.I = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--H")) c.H = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--TOPK")) c.TOPK = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--active-experts")) c.active_experts = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--route-seed")) c.route_seed = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--repeats")) c.repeats = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--flush-mb")) c.flush_mb = u32(need(argv[a]));
        else if (!std::strcmp(argv[a], "--paths")) c.paths = need(argv[a]);
        else if (!std::strcmp(argv[a], "--methods")) c.methods = need(argv[a]);
        else if (!std::strcmp(argv[a], "--csv")) c.csv_path = need(argv[a]);
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[a]); std::exit(2); }
    }
    return c;
}

static std::vector<int> parse_paths(const std::string& text) {
    std::vector<int> out;
    const std::string selected = "," + text + ",";
    for (int p = 0; p < PATH_COUNT; ++p) {
        const std::string key = "," + std::string(path_name(p)) + ",";
        if (text == "all" || selected.find(key) != std::string::npos) out.push_back(p);
    }
    if (out.empty()) { std::fprintf(stderr, "no valid --paths selected: %s\n", text.c_str()); std::exit(2); }
    return out;
}

static std::vector<int> parse_methods(const std::string& text) {
    std::vector<int> out;
    const std::string selected = "," + text + ",";
    for (int m = 0; m < METHOD_COUNT; ++m) {
        const std::string key = "," + std::string(method_name(m)) + ",";
        if (text == "all" || selected.find(key) != std::string::npos) out.push_back(m);
    }
    if (out.empty()) { std::fprintf(stderr, "no valid --methods selected: %s\n", text.c_str()); std::exit(2); }
    return out;
}

static bool method_valid_for_path(int path, int method) {
    return !method_is_cta_nc_actcomp(method) || path == ACTCOMP;
}

template <int PATH, int METHOD>
static void launch_kernel(
    const Config& c,
    const uint32_t* W13,
    const uint32_t* S13,
    const __nv_bfloat16* W13GS,
    const __nv_bfloat16* X,
    const int32_t* Xb,
    const __nv_bfloat16* topk_W,
    uint32_t* Y,
    uint64_t* cycles
) {
    static bool occupancy_checked = false;
    if (!occupancy_checked) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, kernel<PATH, METHOD>, CTA, 0));
        if (blocks_per_sm != 1) {
            std::fprintf(stderr, "%s/%s: expected exactly 1 resident CTA/SM, got %d\n",
                         path_name(PATH), method_name(METHOD), blocks_per_sm);
            std::exit(2);
        }
        occupancy_checked = true;
    }
    kernel<PATH, METHOD><<<c.E, CTA>>>(
        W13, S13, W13GS, X, 0x3f80u, Xb, topk_W, Y, cycles,
        int32_t(c.E), int32_t(c.N), int32_t(c.I), int32_t(c.H), int32_t(c.TOPK));
}

#define DISPATCH_METHOD(P, M) case M: launch_kernel<P, M>(c, W13, S13, W13GS, X, Xb, topk_W, Y, cycles); break

static void launch(
    int path,
    int method,
    const Config& c,
    const uint32_t* W13,
    const uint32_t* S13,
    const __nv_bfloat16* W13GS,
    const __nv_bfloat16* X,
    const int32_t* Xb,
    const __nv_bfloat16* topk_W,
    uint32_t* Y,
    uint64_t* cycles
) {
#define METHOD_SWITCH(P)                                                      \
    switch (method) {                                                         \
    DISPATCH_METHOD(P, 0); DISPATCH_METHOD(P, 1); DISPATCH_METHOD(P, 2);      \
    DISPATCH_METHOD(P, 3); DISPATCH_METHOD(P, 4); DISPATCH_METHOD(P, 5);      \
    DISPATCH_METHOD(P, 6); DISPATCH_METHOD(P, 7); DISPATCH_METHOD(P, 8);      \
    DISPATCH_METHOD(P, 9); DISPATCH_METHOD(P, 10); DISPATCH_METHOD(P, 11);    \
    DISPATCH_METHOD(P, 12); DISPATCH_METHOD(P, 13); DISPATCH_METHOD(P, 14);   \
    DISPATCH_METHOD(P, 15); DISPATCH_METHOD(P, 16); DISPATCH_METHOD(P, 17);   \
    DISPATCH_METHOD(P, 18); DISPATCH_METHOD(P, 19);                           \
    default: std::abort();                                                     \
    }
    if (path == PURE) { METHOD_SWITCH(PURE); }
    else if (path == WDEQUANT) { METHOD_SWITCH(WDEQUANT); }
    else if (path == ACTCOMP) { METHOD_SWITCH(ACTCOMP); }
    else std::abort();
#undef METHOD_SWITCH
}

#undef DISPATCH_METHOD

static uint64_t median_nonzero(std::vector<uint64_t> x) {
    x.erase(std::remove(x.begin(), x.end(), 0ull), x.end());
    if (x.empty()) return 0;
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

static float median_ms(std::vector<float> x) {
    std::sort(x.begin(), x.end());
    const size_t mid = x.size() / 2;
    return x.size() & 1u ? x[mid] : 0.5f * (x[mid - 1] + x[mid]);
}

static uint64_t checksum_Y(uint32_t* Y, uint64_t n) {
    std::vector<uint32_t> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), Y, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint64_t s = 0xcbf29ce484222325ull;
    for (uint32_t x : h) s = (s ^ x) * 0x100000001b3ull;
    return s;
}

static double prefetch_bytes_per_assignment(uint32_t method, uint32_t kt_count, uint32_t i_count) {
    if (!method_has_prefetch(int(method))) return 0.0;
    const bool lead = method == PF128X2_CURRENT_LEAD1_DIRECT ||
                      method == PF64X4_CURRENT_LEAD1_DIRECT ||
                      method == PF64X4_CURRENT_LEAD1_WAIT0_DIRECT ||
                      method == PF128X2_CURRENT_LEAD1_TMA_SMEM ||
                      method == PF64X4_CURRENT_LEAD1_TMA_SMEM ||
                      method == PF64X4_CURRENT_LEAD1_WAIT0_TMA_SMEM;
    const double panels = lead ? double(kt_count + (kt_count ? kt_count - 1u : 0u)) : double(kt_count);
    return panels * 128.0 * double(i_count);
}

int main(int argc, char** argv) {
    Config c = parse_args(argc, argv);
    if (c.TOPK != TOPK_LOCK || c.active_experts < TOPK_LOCK || c.active_experts > c.E ||
        !c.E || !c.N || c.N > 512u || !c.repeats || (c.I & 511u) || (c.H & 63u)) {
        std::fprintf(stderr, "require TOPK=8, 8<=active-experts<=E, 1<=N<=512, nonzero repeats, I%%512=0, H%%64=0\n");
        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    const uint32_t kt_count = c.H >> 6;
    const uint32_t i_count = c.I >> 9;
    const uint32_t n32_count = (c.N + 31u) >> 5;
    const uint32_t Xb_stride = 1u + n32_count;

    const uint64_t W13_u32 = uint64_t(c.E) * kt_count * (uint64_t(c.I) << 5);
    const uint64_t S13_u32 = uint64_t(c.E) * kt_count * (uint64_t(c.I) << 2);
    const uint64_t X_u16 = uint64_t(c.N) * c.H;
    const uint64_t topk_u16 = uint64_t(c.N) * c.TOPK;
    const uint64_t Y_u32 = uint64_t(c.E) * CTA;

    std::vector<int32_t> h_Xb(uint64_t(c.E) * Xb_stride, 0);
    std::vector<uint32_t> expert_assignments(c.E, 0);
    std::vector<uint32_t> expert_pool(c.E);
    for (uint32_t e = 0; e < c.E; ++e) expert_pool[e] = e;
    uint32_t route_state = c.route_seed;
    for (uint32_t i = c.E; i > 1u; --i) {
        route_state = mix32(route_state + i);
        std::swap(expert_pool[i - 1u], expert_pool[route_state % i]);
    }

    uint64_t assignments = 0;
    for (uint32_t token = 0; token < c.N; ++token) {
        uint32_t used[TOPK_LOCK];
        for (uint32_t slot = 0; slot < c.TOPK; ++slot) {
            uint32_t pool_idx = mix32(
                c.route_seed ^ mix32(token + 1u) ^ mix32((slot + 1u) * 0x9e3779b9u)
            ) % c.active_experts;
            uint32_t e = expert_pool[pool_idx];
            bool duplicate = true;
            while (duplicate) {
                duplicate = false;
                for (uint32_t j = 0; j < slot; ++j) duplicate |= used[j] == e;
                if (duplicate) {
                    pool_idx = (pool_idx + 1u) % c.active_experts;
                    e = expert_pool[pool_idx];
                }
            }
            used[slot] = e;
            h_Xb[uint64_t(e) * Xb_stride + 1u + (token >> 5)] |= int32_t(1u << (token & 31u));
            ++expert_assignments[e];
            ++assignments;
        }
    }
    uint32_t live_experts = 0;
    uint32_t min_tokens_per_live_expert = 0xffffffffu;
    uint32_t max_tokens_per_live_expert = 0;
    for (uint32_t e = 0; e < c.E; ++e) {
        h_Xb[uint64_t(e) * Xb_stride] = int32_t(expert_assignments[e]);
        if (expert_assignments[e]) {
            ++live_experts;
            min_tokens_per_live_expert = std::min(min_tokens_per_live_expert, expert_assignments[e]);
            max_tokens_per_live_expert = std::max(max_tokens_per_live_expert, expert_assignments[e]);
        }
    }

    uint32_t *W13 = nullptr, *S13 = nullptr, *Y = nullptr, *flush = nullptr;
    __nv_bfloat16 *W13GS = nullptr, *X = nullptr, *topk_W = nullptr;
    int32_t* Xb = nullptr;
    uint64_t* cycles = nullptr;
    CUDA_CHECK(cudaMalloc(&W13, W13_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&S13, S13_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&W13GS, uint64_t(c.E) * 2u * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&X, X_u16 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Xb, h_Xb.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&topk_W, topk_u16 * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&Y, Y_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&cycles, uint64_t(c.E) * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(Xb, h_Xb.data(), h_Xb.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    uint64_t flush_u32 = uint64_t(c.flush_mb) * 1048576ull / sizeof(uint32_t);
    if (flush_u32) CUDA_CHECK(cudaMalloc(&flush, flush_u32 * sizeof(uint32_t)));
    init_u32<<<4096, 256>>>(W13, W13_u32, 0x13u);
    init_u32<<<4096, 256>>>(S13, S13_u32, 0x51u);
    init_bf16<<<4096, 256>>>(reinterpret_cast<uint16_t*>(W13GS), uint64_t(c.E) * 2u, 0x61u);
    init_bf16<<<4096, 256>>>(reinterpret_cast<uint16_t*>(X), X_u16, 0xa4u);
    init_bf16<<<4096, 256>>>(reinterpret_cast<uint16_t*>(topk_W), topk_u16, 0x77u);
    if (flush_u32) init_u32<<<4096, 256>>>(flush, flush_u32, 0xf1u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("gpu,%s\n", prop.name);
    std::printf("sms,%d\n", prop.multiProcessorCount);
    std::printf("routing,uniform_hash_scattered_pool\nroute_seed,0x%08x\n", c.route_seed);
    std::printf("E,%u\nN,%u\nI,%u\nH,%u\nTOPK,%u\nexpert_pool,%u\nlive_experts,%u\n"
                "min_tokens_per_live_expert,%u\nmax_tokens_per_live_expert,%u\n",
                c.E, c.N, c.I, c.H, c.TOPK, c.active_experts, live_experts,
                min_tokens_per_live_expert, max_tokens_per_live_expert);
    std::printf("W13_GiB,%.3f\nS13_GiB,%.3f\nX_GiB,%.3f\n",
                double(W13_u32 * 4ull) / 1073741824.0,
                double(S13_u32 * 4ull) / 1073741824.0,
                double(X_u16 * 2ull) / 1073741824.0);
    std::printf("traffic_scope,logical_requested_bytes_not_physical_HBM\n");
    std::printf("cycle_scope,median_live_CTA_all_repeats\n");
    std::printf("occupancy_contract,validated_1_CTA_per_SM_per_mode\n");
    std::printf("token_batching,Xb_global_N256_planes_with_sparse_N8_warp_tiles\n");
    std::printf("prefetch_ownership,4x256T_same_Xb_token_4x32B_K64_subpanels\n");
    std::printf("actcomp_ownership,256x4T_Xb_predicated_token_K64_then_packed_smem\n");
    std::printf("wait_scope,live_current_issuers_then_CTA_barrier\n");
    std::printf("pipeline_scope,pre_MMA_transport_or_transform_sink\n");

    FILE* csv = std::fopen(c.csv_path, "w");
    if (!csv) { std::perror("fopen csv"); return 2; }
    std::fprintf(csv,
        "gpu,sms,path,method,transport,pipeline,comparison,destination,pf_shape,routing,route_seed,repeats,flush_mb,E,N,I,H,TOPK,expert_pool,live_experts,assignments,"
        "min_tokens_per_live_expert,max_tokens_per_live_expert,"
        "ms,median_live_CTA_cyc,W_bytes,S_bytes,X_useful_bytes,X_consumer_bytes,X_global_bytes,X_packed_bytes,"
        "X_smem_read_bytes,X_smem_write_bytes,pf_requested_bytes,pipeline_requested_bytes,"
        "X_useful_GBps,X_consumer_GBps,X_global_GBps,X_packed_GBps,X_smem_read_GBps,X_smem_write_GBps,"
        "pf_requested_GBps,pipeline_requested_GBps,checksum\n");

    const auto paths = parse_paths(c.paths);
    const auto methods = parse_methods(c.methods);
    for (int path : paths) {
        for (int method : methods) {
            if (!method_valid_for_path(path, method)) continue;
            launch(path, method, c, W13, S13, W13GS, X, Xb, topk_W, Y, cycles);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> ms_samples;
            std::vector<uint64_t> all_cycles;
            ms_samples.reserve(c.repeats);
            all_cycles.reserve(uint64_t(c.repeats) * live_experts);
            for (uint32_t r = 0; r < c.repeats; ++r) {
                if (flush_u32) {
                    flush_l2<<<4096, 256>>>(flush, flush_u32);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                CUDA_CHECK(cudaMemset(Y, 0, Y_u32 * sizeof(uint32_t)));
                CUDA_CHECK(cudaMemset(cycles, 0, uint64_t(c.E) * sizeof(uint64_t)));
                cudaEvent_t a, b;
                CUDA_CHECK(cudaEventCreate(&a));
                CUDA_CHECK(cudaEventCreate(&b));
                CUDA_CHECK(cudaEventRecord(a));
                launch(path, method, c, W13, S13, W13GS, X, Xb, topk_W, Y, cycles);
                CUDA_CHECK(cudaEventRecord(b));
                CUDA_CHECK(cudaEventSynchronize(b));
                CUDA_CHECK(cudaGetLastError());
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                ms_samples.push_back(ms);
                CUDA_CHECK(cudaEventDestroy(a));
                CUDA_CHECK(cudaEventDestroy(b));
                std::vector<uint64_t> run_cycles(c.E);
                CUDA_CHECK(cudaMemcpy(run_cycles.data(), cycles, uint64_t(c.E) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                for (uint64_t x : run_cycles) if (x) all_cycles.push_back(x);
            }
            const float ms = median_ms(ms_samples);
            const uint64_t med = median_nonzero(all_cycles);
            const uint64_t checksum = checksum_Y(Y, Y_u32);

            const double W_bytes = double(live_experts) * kt_count * i_count * 65536.0;
            const double S_bytes = double(live_experts) * kt_count * i_count * 8192.0;
            const double X_useful = double(assignments) * double(c.H) * 2.0 * i_count;
            const bool actcomp_producer = path == ACTCOMP &&
                (method_is_tma(method) || method_is_cta_nc_actcomp(method));
            const double X_consumer = X_useful * (actcomp_producer ? 1.0 : 32.0);
            const double X_global = method_is_tma(method) ? X_useful : X_consumer;
            const double X_packed = actcomp_producer
                ? double(assignments) * double(kt_count) * 36.0 * double(i_count)
                : 0.0;
            const double X_smem_write = (method_is_tma(method) ? X_useful : 0.0) + X_packed;
            const double X_smem_read = method_is_tma(method)
                ? (actcomp_producer ? X_useful : X_consumer) + X_packed
                : X_packed;
            const double pf_bytes = double(assignments) * prefetch_bytes_per_assignment(method, kt_count, i_count);
            const double pipeline_bytes = W_bytes + S_bytes + X_global + X_smem_read +
                X_smem_write + pf_bytes;
            const double denom = double(ms) * 1.0e6;

            std::fprintf(csv,
                "%s,%d,%s,%s,%s,%s,%s,%s,%s,uniform_hash_scattered_pool,0x%08x,%u,%u,%u,%u,%u,%u,%u,%u,%u,%llu,%u,%u,%.6f,%llu,"
                "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,"
                "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,0x%016llx\n",
                prop.name, prop.multiProcessorCount, path_name(path), method_name(method),
                transport_name(method), pipeline_name(path, method), comparison_name(path, method),
                destination_name(path, method), pf_shape_name(method),
                c.route_seed, c.repeats, c.flush_mb,
                c.E, c.N, c.I, c.H, c.TOPK,
                c.active_experts, live_experts, (unsigned long long)assignments,
                min_tokens_per_live_expert, max_tokens_per_live_expert, ms,
                (unsigned long long)med, W_bytes, S_bytes, X_useful, X_consumer, X_global,
                X_packed, X_smem_read, X_smem_write, pf_bytes, pipeline_bytes,
                X_useful / denom, X_consumer / denom, X_global / denom, X_packed / denom,
                X_smem_read / denom, X_smem_write / denom, pf_bytes / denom,
                pipeline_bytes / denom, (unsigned long long)checksum);
            std::fflush(csv);
            std::printf("%-8s %-60s %9.3f ms  X useful %8.1f  X global %8.1f GB/s\n",
                        path_name(path), method_name(method), ms, X_useful / denom, X_global / denom);
        }
    }
    std::fclose(csv);

    CUDA_CHECK(cudaFree(W13));
    CUDA_CHECK(cudaFree(S13));
    CUDA_CHECK(cudaFree(W13GS));
    CUDA_CHECK(cudaFree(X));
    CUDA_CHECK(cudaFree(Xb));
    CUDA_CHECK(cudaFree(topk_W));
    CUDA_CHECK(cudaFree(Y));
    CUDA_CHECK(cudaFree(cycles));
    if (flush) CUDA_CHECK(cudaFree(flush));
    return 0;
}
'''


METHODS = [
    "no_pf_direct",
    "pf128_current_direct",
    "pf64x2_current_direct",
    "pf128x2_current_lead1_direct",
    "pf64x4_current_lead1_direct",
    "pf64x4_current_wait0_lead1_async_direct",
    "no_pf_tma_smem",
    "pf128_current_tma_smem",
    "pf64x2_current_tma_smem",
    "pf128x2_current_lead1_tma_smem",
    "pf64x4_current_lead1_tma_smem",
    "pf64x4_current_wait0_lead1_async_tma_smem",
    "pf32x4_current_direct",
    "pf32x4_current_wait0_direct",
    "pf32x4_current_tma_smem",
    "pf32x4_current_wait0_tma_smem",
    "no_pf_cta_nc_actcomp_smem",
    "pf128_current_cta_nc_actcomp_smem",
    "pf32x4_current_cta_nc_actcomp_smem",
    "pf32x4_current_wait0_cta_nc_actcomp_smem",
]
PATHS = ["pure", "wdequant", "actcomp"]


def normalized_selection(raw: str, choices: list[str], flag: str) -> str:
    if raw == "all":
        return raw
    picked = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in picked if item not in choices]
    if unknown:
        raise SystemExit(f"{flag}: unknown value(s): {','.join(unknown)}")
    if not picked:
        raise SystemExit(f"{flag}: empty selection")
    return ",".join(dict.fromkeys(picked))


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run(cmd: list[str]) -> None:
    print("+", command_text(cmd), flush=True)
    subprocess.run(cmd, check=True)


def audit_ptx() -> None:
    forbidden = {
        r"mma\.sync": "this experiment stops before MMA",
        r"ld\.global\.nc\.(?:ca|cg|cs)": "cache operator must precede .nc",
        r"ld\.global\.(?:ca|cg|cs)\.nc\.L[12]::evict_":
            "cache operator and explicit eviction priority are separate ld.global.nc forms",
        r"cp\.async\.bulk\.prefetch[^;\n]*\.L2::(?:64B|128B|256B)": "bulk PF has no L2 prefetch-size modifier",
        r"cp\.async\.bulk\.prefetch[^;\n]*(?:evict_|level::eviction_priority)": "bulk PF has no eviction modifier",
    }
    for pattern, message in forbidden.items():
        hit = re.search(pattern, CUDA_SRC)
        if hit:
            line = CUDA_SRC.count("\n", 0, hit.start()) + 1
            raise SystemExit(f"PTX audit line {line}: {message}")
    for size in re.findall(r"prefetch_X<(\d+)>", CUDA_SRC):
        if int(size) % 16:
            raise SystemExit(f"PTX audit: prefetch size {size} is not 16B aligned")
    block = re.search(
        r"static const char\* method_name\(int method\) \{(.*?)\n\}\n\nstatic const char\* transport_name",
        CUDA_SRC,
        re.S,
    )
    if not block:
        raise SystemExit("PTX audit: method_name table not found")
    cuda_methods = [name for name in re.findall(r'return "([^"]+)"', block.group(1)) if name != "unknown"]
    if cuda_methods != METHODS:
        raise SystemExit("PTX audit: CUDA and Python method tables differ")


def validate_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise SystemExit("CSV header missing or duplicated")
        rows = list(reader)
    if not rows:
        raise SystemExit("CSV has no rows")

    pairs = [
        ("X_useful_bytes", "X_useful_GBps"),
        ("X_consumer_bytes", "X_consumer_GBps"),
        ("X_global_bytes", "X_global_GBps"),
        ("X_packed_bytes", "X_packed_GBps"),
        ("X_smem_read_bytes", "X_smem_read_GBps"),
        ("X_smem_write_bytes", "X_smem_write_GBps"),
        ("pf_requested_bytes", "pf_requested_GBps"),
        ("pipeline_requested_bytes", "pipeline_requested_GBps"),
    ]
    checksums: dict[tuple[str, str, str, str, str], set[str]] = {}
    for line, row in enumerate(rows, 2):
        ms = float(row["ms"])
        if ms <= 0:
            raise SystemExit(f"CSV line {line}: non-positive ms")
        for byte_key, rate_key in pairs:
            expected = float(row[byte_key]) / (ms * 1.0e6)
            actual = float(row[rate_key])
            if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-5):
                raise SystemExit(
                    f"CSV line {line}: {rate_key}={actual}, expected {expected}"
                )

        N = int(row["N"])
        I = int(row["I"])
        H = int(row["H"])
        assignments = int(row["assignments"])
        if assignments != N * int(row["TOPK"]):
            raise SystemExit(f"CSV line {line}: assignments != N*TOPK")
        kt_count = H // 64
        i_count = I // 512
        useful = assignments * H * 2 * i_count
        producer = row["comparison"] == "actcomp_packed_smem_producer"
        consumer = useful if producer else useful * 32
        global_bytes = useful if row["transport"] in {"tma", "cta_nc"} else consumer
        packed = assignments * kt_count * 36 * i_count if producer else 0
        no_pf = {
            "no_pf_direct", "no_pf_tma_smem", "no_pf_cta_nc_actcomp_smem",
        }
        lead1_pf = {
            "pf128x2_current_lead1_direct",
            "pf64x4_current_lead1_direct",
            "pf64x4_current_wait0_lead1_async_direct",
            "pf128x2_current_lead1_tma_smem",
            "pf64x4_current_lead1_tma_smem",
            "pf64x4_current_wait0_lead1_async_tma_smem",
        }
        pf_panels = 0 if row["method"] in no_pf else (
            2 * kt_count - 1 if row["method"] in lead1_pf else kt_count
        )
        pf_expected = assignments * pf_panels * 128 * i_count
        smem_write = (useful if row["transport"] == "tma" else 0) + packed
        smem_read = (
            ((useful if producer else consumer) if row["transport"] == "tma" else 0)
            + packed
        )
        expected_bytes = {
            "W_bytes": int(row["live_experts"]) * kt_count * i_count * 65536,
            "S_bytes": int(row["live_experts"]) * kt_count * i_count * 8192,
            "X_useful_bytes": useful,
            "X_consumer_bytes": consumer,
            "X_global_bytes": global_bytes,
            "X_packed_bytes": packed,
            "X_smem_read_bytes": smem_read,
            "X_smem_write_bytes": smem_write,
            "pf_requested_bytes": pf_expected,
        }
        for key, expected in expected_bytes.items():
            if not math.isclose(float(row[key]), expected, rel_tol=0.0, abs_tol=0.5):
                raise SystemExit(f"CSV line {line}: {key}={row[key]}, expected {expected}")
        pipeline = (
            float(row["W_bytes"]) + float(row["S_bytes"]) + global_bytes + smem_read +
            smem_write + float(row["pf_requested_bytes"])
        )
        if not math.isclose(
            float(row["pipeline_requested_bytes"]), pipeline, rel_tol=0.0, abs_tol=0.5
        ):
            raise SystemExit(f"CSV line {line}: pipeline_requested_bytes mismatch")
        checksum_key = (
            row["N"], row["expert_pool"], row["route_seed"], row["path"], row["comparison"],
        )
        checksums.setdefault(checksum_key, set()).add(row["checksum"])
    mismatched = [key for key, values in checksums.items() if len(values) != 1]
    if mismatched:
        raise SystemExit(f"CSV checksum mismatch in fair group: {mismatched[0]}")
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    tma_baselines = {
        (row["path"], row["comparison"]): float(row["ms"])
        for row in rows
        if row["method"] == "no_pf_tma_smem"
    }
    print()
    print("+-----+----------+-------------+--------------------------------------------------------------+----------+----------+----------+----------+")
    print("| N   | path     | transport   | method                                                       | ms       | vs TMA0  | X useful | X global |")
    print("+-----+----------+-------------+--------------------------------------------------------------+----------+----------+----------+----------+")
    for path in ("pure", "wdequant", "actcomp"):
        part = sorted((r for r in rows if r["path"] == path), key=lambda r: float(r["ms"]))
        for row in part:
            baseline = tma_baselines.get((row["path"], row["comparison"]))
            vs_tma = "diagnostic" if baseline is None else f"{100.0 * (1.0 - float(row['ms']) / baseline):+7.2f}%"
            print(
                f"| {int(row['N']):3d} | {path:<8} | {row['transport']:<11} | {row['method']:<60} | {float(row['ms']):8.3f} "
                f"| {vs_tma:>8} | {float(row['X_useful_GBps']):8.1f} | {float(row['X_global_GBps']):8.1f} |"
            )
        if part:
            print("+-----+----------+-------------+--------------------------------------------------------------+----------+----------+----------+----------+")


def u32_list(raw: str, flag: str) -> list[int]:
    try:
        values = [int(item.strip(), 0) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{flag}: expected comma-separated integers") from exc
    if not values:
        raise SystemExit(f"{flag}: empty selection")
    return list(dict.fromkeys(values))


def native_command(
    args: argparse.Namespace,
    exe: Path,
    csv_path: Path,
    N: int,
    expert_pool: int,
    route_seed: int,
    repeats: int,
    flush_mb: int,
) -> list[str]:
    return [
        str(exe),
        "--E", str(args.E),
        "--N", str(N),
        "--I", str(args.I),
        "--H", str(args.H),
        "--TOPK", str(args.TOPK),
        "--active-experts", str(expert_pool),
        "--route-seed", hex(route_seed),
        "--repeats", str(repeats),
        "--flush-mb", str(flush_mb),
        "--paths", args.paths,
        "--methods", args.methods,
        "--csv", str(csv_path.resolve()),
    ]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_sweep(rows: list[dict[str, str]], out_dir: Path) -> None:
    transport_baseline_methods = {
        "warp_direct": "no_pf_direct",
        "tma": "no_pf_tma_smem",
        "cta_nc": "no_pf_cta_nc_actcomp_smem",
    }
    tma_baselines: dict[tuple[str, str, str, str, str, str], float] = {}
    transport_baselines: dict[tuple[str, str, str, str, str, str, str], float] = {}
    for row in rows:
        if row["method"] == "no_pf_tma_smem":
            key = (
                row["N"], row["expert_pool"], row["route_seed"], row["flush_mb"],
                row["path"], row["comparison"],
            )
            tma_baselines[key] = float(row["ms"])
        expected = transport_baseline_methods[row["transport"]]
        if row["method"] == expected:
            key = (
                row["N"], row["expert_pool"], row["route_seed"], row["flush_mb"],
                row["path"], row["comparison"], row["transport"],
            )
            transport_baselines[key] = float(row["ms"])

    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["N"], row["expert_pool"], row["flush_mb"], row["path"],
            row["comparison"], row["transport"], row["method"],
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, str]] = []
    for (N, pool, flush_mb, path, comparison, transport, method), group in sorted(grouped.items()):
        times = [float(row["ms"]) for row in group]
        tma_speedups = []
        tma_reductions = []
        tma_times = []
        transport_speedups = []
        transport_reductions = []
        for row in group:
            tma_key = (N, pool, row["route_seed"], flush_mb, path, comparison)
            if tma_key in tma_baselines:
                tma_times.append(tma_baselines[tma_key])
                tma_speedups.append(tma_baselines[tma_key] / float(row["ms"]))
                tma_reductions.append(
                    100.0 * (1.0 - float(row["ms"]) / tma_baselines[tma_key])
                )
            transport_key = tma_key + (transport,)
            transport_speedups.append(
                transport_baselines[transport_key] / float(row["ms"])
            )
            transport_reductions.append(
                100.0 * (
                    1.0 - float(row["ms"]) / transport_baselines[transport_key]
                )
            )
        tma_speedup = statistics.median(tma_speedups) if tma_speedups else None
        transport_speedup = statistics.median(transport_speedups)
        summary.append({
            "N": N,
            "expert_pool": pool,
            "flush_mb": flush_mb,
            "path": path,
            "comparison": comparison,
            "transport": transport,
            "method": method,
            "seeds": str(len(group)),
            "live_experts_min": str(min(int(row["live_experts"]) for row in group)),
            "live_experts_max": str(max(int(row["live_experts"]) for row in group)),
            "tokens_per_live_expert_min": str(min(int(row["min_tokens_per_live_expert"]) for row in group)),
            "tokens_per_live_expert_max": str(max(int(row["max_tokens_per_live_expert"]) for row in group)),
            "median_ms": f"{statistics.median(times):.6f}",
            "min_ms": f"{min(times):.6f}",
            "max_ms": f"{max(times):.6f}",
            "standard_baseline": "no_pf_tma_smem" if tma_speedup is not None else "not_comparable",
            "tma_baseline_median_ms": f"{statistics.median(tma_times):.6f}" if tma_times else "",
            "median_speedup_vs_tma": f"{tma_speedup:.6f}" if tma_speedup is not None else "",
            "median_time_reduction_vs_tma_pct":
                f"{statistics.median(tma_reductions):.4f}" if tma_reductions else "",
            "transport_no_pf_baseline": transport_baseline_methods[transport],
            "median_speedup_vs_transport_no_pf": f"{transport_speedup:.6f}",
            "median_time_reduction_vs_transport_no_pf_pct":
                f"{statistics.median(transport_reductions):.4f}",
        })

    summary_path = out_dir / "summary.csv"
    write_csv(summary_path, summary, list(summary[0]))

    print()
    print("TMA-standard fair winners (different output contracts are never mixed):")
    print("+-----+------+-------+----------+-------------------------------+-------------+--------------------------------------------------------------+----------+----------+")
    print("| N   | pool | flush | path     | fair comparison               | transport   | winning method                                               | med ms   | vs TMA0  |")
    print("+-----+------+-------+----------+-------------------------------+-------------+--------------------------------------------------------------+----------+----------+")
    fair_keys = sorted({
        (row["N"], row["expert_pool"], row["flush_mb"], row["path"], row["comparison"])
        for row in summary
        if row["standard_baseline"] == "no_pf_tma_smem"
    }, key=lambda x: (int(x[0]), int(x[1]), int(x[2]), PATHS.index(x[3]), x[4]))
    for N, pool, flush_mb, path, comparison in fair_keys:
        part = [
            row for row in summary
            if row["N"] == N and row["expert_pool"] == pool and row["flush_mb"] == flush_mb and row["path"] == path and
               row["comparison"] == comparison
        ]
        best = max(
            part,
            key=lambda row: (
                float(row["median_time_reduction_vs_tma_pct"]),
                -float(row["median_ms"]),
            ),
        )
        print(
            f"| {int(N):3d} | {int(pool):4d} | {int(flush_mb):5d} | {path:<8} | {comparison:<29} | {best['transport']:<11} "
            f"| {best['method']:<60} | {float(best['median_ms']):8.3f} "
            f"| {float(best['median_time_reduction_vs_tma_pct']):+7.2f}% |"
        )
    print("+-----+------+-------+----------+-------------------------------+-------------+--------------------------------------------------------------+----------+----------+")
    diagnostics = sum(row["standard_baseline"] == "not_comparable" for row in summary)
    if diagnostics:
        print(f"Excluded diagnostic rows without an equal-work TMA baseline: {diagnostics}")
    print(f"Sweep summary: {summary_path.resolve()}")


def run_sweep(args: argparse.Namespace, exe: Path, dry_run: bool = False) -> None:
    Ns = u32_list(args.sweep_n, "--sweep-n")
    pools = u32_list(args.sweep_pools, "--sweep-pools")
    seeds = u32_list(args.sweep_seeds, "--sweep-seeds")
    flushes = u32_list(args.sweep_flush_mb, "--sweep-flush-mb") \
        if args.sweep_flush_mb else [args.flush_mb]
    for N in Ns:
        if N < 1 or N > 512:
            raise SystemExit(f"--sweep-n: require 1 <= N <= 512, got {N}")
    for pool in pools:
        if pool < args.TOPK or pool > args.E:
            raise SystemExit(f"--sweep-pools: require TOPK <= pool <= E, got {pool}")

    commands: list[tuple[Path, list[str]]] = []
    for N in Ns:
        for pool in pools:
            for seed in seeds:
                for flush_mb in flushes:
                    csv_path = args.sweep_dir / (
                        f"N{N}_pool{pool}_seed{seed:08x}_flush{flush_mb}.csv"
                    )
                    commands.append((csv_path, native_command(
                        args, exe, csv_path, N, pool, seed, args.sweep_repeats, flush_mb
                    )))

    if dry_run:
        for _, command in commands:
            print("+", command_text(command))
        return

    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for csv_path, command in commands:
        print(f"\n=== {csv_path.stem} ===", flush=True)
        run(command)
        current = validate_csv(csv_path)
        if fieldnames is None:
            fieldnames = list(current[0])
        all_rows.extend(current)

    if not all_rows or fieldnames is None:
        raise SystemExit("sweep produced no rows")
    combined = args.sweep_dir / "combined.csv"
    write_csv(combined, all_rows, fieldnames)
    summarize_sweep(all_rows, args.sweep_dir)
    print(f"Sweep raw rows: {combined.resolve()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-f", dest="_jupyter_connection_file", help=argparse.SUPPRESS)
    ap.add_argument("--arch", default="sm_120a")
    ap.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    ap.add_argument("--workdir", type=Path, default=Path("/content/xcaliber_kernel_sim"))
    ap.add_argument("--csv", type=Path, default=Path("/content/kernel_sim_results.csv"))
    ap.add_argument("--E", type=int, default=384)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--I", type=int, default=2048)
    ap.add_argument("--H", type=int, default=7168)
    ap.add_argument("--TOPK", type=int, default=8)
    ap.add_argument("--active-experts", type=int, default=384)
    ap.add_argument("--route-seed", type=lambda x: int(x, 0), default=0x6A09E667)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--flush-mb", type=int, default=512)
    ap.add_argument("--paths", default="pure,wdequant,actcomp")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-n", default="8,256,512")
    ap.add_argument("--sweep-pools", default="128,384")
    ap.add_argument("--sweep-seeds", default="0x6a09e667,0xbb67ae85,0x3c6ef372")
    ap.add_argument("--sweep-flush-mb", default="")
    ap.add_argument("--sweep-repeats", type=int, default=20)
    ap.add_argument("--sweep-dir", type=Path, default=Path("/content/l2pfact_sweep_v2"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args._jupyter_connection_file:
        args.sweep = True
        if COLAB_SWEEP_PROFILE == "flush_sensitivity":
            args.paths = "actcomp"
            args.methods = ",".join([
                "no_pf_tma_smem",
                "pf128_current_tma_smem",
                "pf64x2_current_tma_smem",
                "pf32x4_current_tma_smem",
                "pf32x4_current_wait0_tma_smem",
                "no_pf_cta_nc_actcomp_smem",
                "pf128_current_cta_nc_actcomp_smem",
                "pf32x4_current_cta_nc_actcomp_smem",
                "pf32x4_current_wait0_cta_nc_actcomp_smem",
            ])
            args.sweep_flush_mb = "0,128,512"
            args.sweep_dir = Path("/content/l2pfact_flush_sensitivity")
    if args.list:
        print("paths: pure,wdequant,actcomp")
        for i, name in enumerate(METHODS):
            print(f"{i:2d} {name}")
        return 0

    audit_ptx()
    args.paths = normalized_selection(args.paths, PATHS, "--paths")
    args.methods = normalized_selection(args.methods, METHODS, "--methods")
    if args.quick:
        args.E = 32
        args.N = 128
        args.I = 512
        args.H = 128
        args.active_experts = 24
        args.repeats = 1
        args.flush_mb = 64

    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        if not args.dry_run:
            args.sweep_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
    cu = args.workdir / "kernel_sim.cu"
    exe = args.workdir / "kernel_sim"
    cu.write_text(CUDA_SRC, encoding="utf-8")

    build = [
        args.nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={args.arch}",
        "-lineinfo",
        "-Xptxas=-v",
        str(cu),
        "-o",
        str(exe),
    ]
    launch = native_command(
        args, exe, args.csv, args.N, args.active_experts, args.route_seed,
        args.repeats, args.flush_mb
    )

    if args.dry_run:
        print("PTX audit: pass")
        print("+", command_text(build))
        if args.sweep:
            run_sweep(args, exe, dry_run=True)
        else:
            print("+", command_text(launch))
        return 0
    if not args.no_build:
        run(build)
    elif not exe.is_file():
        raise SystemExit(f"--no-build executable missing: {exe}")
    if args.compile_only:
        return 0
    if args.sweep:
        run_sweep(args, exe)
        return 0
    run(launch)
    rows = validate_csv(args.csv)
    print_table(rows)
    print(f"CSV: {args.csv.resolve()}")
    return 0


if __name__ == "__main__":
    status = main()
    if "ipykernel" not in sys.modules:
        raise SystemExit(status)
