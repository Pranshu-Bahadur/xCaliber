#include "xR64ABS.cuh"

// CTA256 = 8 tokens. lane l owns experts {2l,2l+1} + 64j.
__global__ __launch_bounds__(XR64_ABS_ROUTE_CTA)
void xR64ABSTopK(
    const uint16_t* __restrict__ logits,
    const uint16_t* __restrict__ expert_scale,
    int32_t* __restrict__ topk_idx,
    uint16_t* __restrict__ topk_weight,
    int N,
    int E) {

    uint32_t lane = threadIdx.x & 31u;
    uint32_t token = blockIdx.x * 8u + ((uint32_t)threadIdx.x >> 5);
    if (token >= (uint32_t)N) return;

    uint32_t dead = 0u;
    uint32_t selected = 0u;
    for (uint32_t rank = 0u; rank < XR64_ABS_TOPK; rank++) {
        uint32_t best = 0u;
        for (uint32_t e = lane << 1, slot = 0u;
             e < (uint32_t)E; e += 64u, slot += 2u) {
            uint32_t x = xr64_ld_ca_nc_u32(
                logits + (uint64_t)token * (uint32_t)E + e);
            if (!(dead & (1u << slot))) {
                uint32_t raw = x & 0xffffu;
                uint32_t key = raw ^ ((0u - (raw >> 15)) | 0x8000u);
                best = max(best, (key << 16) | (0xffffu - e));
            }
            if (!(dead & (2u << slot))) {
                uint32_t raw = x >> 16;
                uint32_t key = raw ^ ((0u - (raw >> 15)) | 0x8000u);
                best = max(best, (key << 16) | (0xfffeu - e));
            }
        }

        for (uint32_t d = 16u; d; d >>= 1)
            best = max(best,
                __shfl_xor_sync(0xffffffffu, best, d));
        uint32_t expert = 0xffffu - (best & 0xffffu);
        dead |= (uint32_t)(lane == ((expert >> 1) & 31u))
             << (((expert >> 6) << 1) | (expert & 1u));
        if (lane == rank) selected = best;
    }

    float x = 0.0f;
    uint32_t expert = 0u;
    if (lane < XR64_ABS_TOPK) {
        expert = 0xffffu - (selected & 0xffffu);
        uint32_t ordered = selected >> 16;
        uint32_t raw = ordered
            ^ (0xffffu ^ ((0u - (ordered >> 15)) & 0x7fffu));
        x = xr64_bf16((uint16_t)raw);
    }

    float top1 = __shfl_sync(0xffffffffu, x, 0);
    float weight = 0.0f;
    if (lane < XR64_ABS_TOPK)
        asm volatile(
            "ex2.approx.ftz.f32 %0, %1;"
            : "=f"(weight)
            : "f"((x - top1) * 1.4426950408889634f));

    float denom = weight;
    denom += __shfl_xor_sync(0xffffffffu, denom, 16);
    denom += __shfl_xor_sync(0xffffffffu, denom, 8);
    denom += __shfl_xor_sync(0xffffffffu, denom, 4);
    denom += __shfl_xor_sync(0xffffffffu, denom, 2);
    denom += __shfl_xor_sync(0xffffffffu, denom, 1);
    asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(denom));

    if (lane < XR64_ABS_TOPK) {
        if (expert_scale) {
            uint32_t s = xr64_ld_ca_nc_u32(
                expert_scale + (expert & ~1u));
            weight *= xr64_bf16(
                (uint16_t)(s >> ((expert & 1u) << 4)));
        }
        topk_idx[(uint64_t)token * XR64_ABS_TOPK + lane] = expert;
        topk_weight[(uint64_t)token * XR64_ABS_TOPK + lane]
            = xr64_to_bf16(weight * denom);
    }
}

// One CTA/expert: stable token order, exact topk_off, no global atomics.
__global__ __launch_bounds__(XR64_ABS_ROUTE_CTA)
void xR64ABSRoutePack(
    const int32_t* __restrict__ topk_idx,
    const uint16_t* __restrict__ topk_weight,
    int32_t* __restrict__ topk_off,
    uint16_t* __restrict__ route_weight,
    int32_t* __restrict__ route_token,
    int32_t* __restrict__ count,
    int N,
    int NS) {

    __shared__ uint32_t warp_count[XR64_ABS_ROUTE_CTA >> 5];
    __shared__ uint32_t packed_base;
    uint32_t e = blockIdx.x;
    uint32_t warp = (uint32_t)threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31u;
    if (!threadIdx.x) packed_base = 0u;
    __syncthreads();

    for (uint32_t first = 0u; first < ((uint32_t)N + 31u) >> 5;
         first += XR64_ABS_ROUTE_CTA >> 5) {
        uint32_t token = ((first + warp) << 5) + lane;
        uint32_t hit = 0u, slot = 0u, weight = 0u;
        if (token < (uint32_t)N) {
            for (uint32_t k = 0u; k < XR64_ABS_TOPK; k++) {
                if (xr64_ld_ca_nc_u32(
                        topk_idx + (uint64_t)token * XR64_ABS_TOPK + k)
                    == e) {
                    hit = 1u;
                    slot = k;
                }
            }
            if (hit) {
                uint32_t p = token * XR64_ABS_TOPK + slot;
                uint32_t x = xr64_ld_ca_nc_u32(
                    topk_weight + (p & ~1u));
                weight = (x >> ((p & 1u) << 4)) & 0xffffu;
            }
        }

        uint32_t bits = __ballot_sync(0xffffffffu, hit != 0u);
        if (!lane) warp_count[warp] = __popc(bits);
        __syncthreads();

        uint32_t rank = packed_base;
        for (uint32_t w = 0u; w < warp; w++) rank += warp_count[w];
        rank += __popc(bits & ((1u << lane) - 1u));
        if (hit) {
            topk_off[(uint64_t)token * XR64_ABS_TOPK + slot] = rank;
            route_weight[(uint64_t)e * NS + rank] = (uint16_t)weight;
            route_token[(uint64_t)e * NS + rank] = token;
        }
        __syncthreads();

        if (!threadIdx.x)
            for (uint32_t w = 0u; w < XR64_ABS_ROUTE_CTA >> 5; w++)
                packed_base += warp_count[w];
        __syncthreads();
    }

    if (!threadIdx.x) count[e] = packed_base;
}

void launch_xr64_abs_topk(
    const uint16_t* logits,
    const uint16_t* expert_scale,
    int32_t* topk_idx,
    uint16_t* topk_weight,
    int N,
    int E,
    cudaStream_t stream) {

    xR64ABSTopK<<<((uint32_t)N + 7u) >> 3,
                    XR64_ABS_ROUTE_CTA, 0, stream>>>(
        logits, expert_scale, topk_idx, topk_weight, N, E);
}

void launch_xr64_abs_route_pack(
    const int32_t* topk_idx,
    const uint16_t* topk_weight,
    int32_t* topk_off,
    uint16_t* route_weight,
    int32_t* route_token,
    int32_t* count,
    int N,
    int NS,
    int E,
    cudaStream_t stream) {

    xR64ABSRoutePack<<<E, XR64_ABS_ROUTE_CTA, 0, stream>>>(
        topk_idx, topk_weight, topk_off, route_weight,
        route_token, count, N, NS);
}
