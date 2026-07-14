#include "xR57F1_contiguous_graph.cu"
#include "xR57F2_direct_reduce_graph.cu"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                              \
    cudaError_t e__ = (call);                                              \
    if (e__ != cudaSuccess) {                                              \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,          \
                     cudaGetErrorString(e__));                             \
        std::exit(1);                                                      \
    }                                                                      \
} while (0)

struct Shape {
    int E, N, NP, I, H, TOPK;
    uint32_t abs_blocks;
};

struct RouteStats {
    uint64_t assignments = 0;
    uint64_t active_n8 = 0;
    uint64_t active_n16 = 0;
    uint64_t paired_n16 = 0;
    int live_experts = 0;
    uint32_t max_tokens_per_expert = 0;
    uint32_t max_active_n8_per_expert = 0;
    uint32_t max_active_n16_per_expert = 0;
    uint32_t max_paired_n16_per_expert = 0;
};

struct Buffers {
    uint32_t *W13 = nullptr, *S13 = nullptr;
    uint16_t* W13GS = nullptr;
    uint32_t *W2 = nullptr, *S2 = nullptr;
    uint16_t* W2GS = nullptr;
    uint16_t* X = nullptr;
    int32_t* topk_idx = nullptr;
    uint16_t* topk_W = nullptr;
    uint32_t *X4 = nullptr, *Sx = nullptr;
    uint16_t* topk_off = nullptr;
    uint16_t* expert_topk_W = nullptr;
    uint16_t* expert_token_idx = nullptr;
    uint32_t *Y4 = nullptr, *SY = nullptr, *YGSINV = nullptr;
    uint16_t* O = nullptr;
    uint32_t* partial = nullptr;
    MoeActScale* act_scale = nullptr;
};

struct GraphExec {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
};

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint16_t bf16_raw(float x) {
    __nv_bfloat16 b = __float2bfloat16(x);
    uint16_t raw;
    std::memcpy(&raw, &b, sizeof(raw));
    return raw;
}

static void add_unique(std::vector<int>& row, int e) {
    if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
}

static std::vector<std::vector<int>> make_routes(
    int E, int N, int TOPK, const std::string& mode, uint32_t seed
) {
    std::mt19937 rng(seed);
    std::vector<std::vector<int>> routes(N);
    std::vector<double> weights(E);
    std::vector<int> perm(E);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int rank = 0; rank < E; rank++)
        weights[perm[rank]] = 1.0 / std::pow(double(rank + 1), 1.10);
    std::discrete_distribution<int> zipf(weights.begin(), weights.end());
    std::uniform_int_distribution<int> uniform(0, E - 1);

    int burst_center = uniform(rng);
    int local_span = std::max(TOPK * 3, std::max(8, E / 16));
    for (int n = 0; n < N; n++) {
        if (mode == "burst" && !(n & 15)) burst_center = zipf(rng);
        while ((int)routes[n].size() < TOPK) {
            int e;
            if (mode == "uniform") {
                e = uniform(rng);
            } else if (mode == "zipf") {
                e = zipf(rng);
            } else if (mode == "burst") {
                if ((int)routes[n].size() < (TOPK * 3 + 3) / 4) {
                    int d = int(rng() % (uint32_t)local_span) - local_span / 2;
                    e = ((burst_center + d) % E + E) % E;
                } else {
                    e = zipf(rng);
                }
            } else {
                std::fprintf(stderr, "unknown routing profile: %s\n", mode.c_str());
                std::exit(2);
            }
            add_unique(routes[n], e);
        }
    }
    return routes;
}

static RouteStats encode_topk_inputs(
    const std::vector<std::vector<int>>& routes,
    const Shape& s,
    std::vector<int32_t>& topk_idx,
    std::vector<uint16_t>& topk_W,
    std::vector<uint16_t>& topk_off,
    std::vector<uint16_t>& expert_topk_W,
    std::vector<uint16_t>& expert_token_idx
) {
    topk_idx.resize((uint64_t)s.N * s.TOPK);
    topk_W.resize((uint64_t)s.N * s.TOPK);
    topk_off.resize((uint64_t)s.N * s.TOPK);
    expert_topk_W.assign((uint64_t)s.E * s.NP, 0u);
    expert_token_idx.assign((uint64_t)s.E * (s.NP + 1u), 0u);
    std::vector<uint32_t> counts(s.E, 0u);
    uint16_t weight = bf16_raw(1.0f / float(s.TOPK));

    for (int n = 0; n < s.N; n++) {
        for (int k = 0; k < s.TOPK; k++) {
            int e = routes[n][k];
            uint32_t p = counts[e]++;
            topk_idx[(uint64_t)n * s.TOPK + k] = e;
            topk_W[(uint64_t)n * s.TOPK + k] = weight;
            topk_off[(uint64_t)n * s.TOPK + k] = (uint16_t)p;
            expert_topk_W[(uint64_t)e * s.NP + p] = weight;
            expert_token_idx[(uint64_t)e * (s.NP + 1u) + 1u + p]
                = (uint16_t)n;
        }
    }

    RouteStats out;
    out.assignments = (uint64_t)s.N * s.TOPK;
    for (int e = 0; e < s.E; e++) {
        expert_token_idx[(uint64_t)e * (s.NP + 1u)]
            = (uint16_t)counts[e];
        out.live_experts += counts[e] != 0u;
        out.max_tokens_per_expert = std::max(
            out.max_tokens_per_expert, counts[e]);
        uint32_t active_n8 = (counts[e] + 7u) >> 3;
        uint32_t active_n16 = (counts[e] + 15u) >> 4;
        uint32_t paired_n16 = (counts[e] + 31u) >> 5;
        out.active_n8 += active_n8;
        out.active_n16 += active_n16;
        out.paired_n16 += paired_n16;
        out.max_active_n8_per_expert = std::max(
            out.max_active_n8_per_expert, active_n8);
        out.max_active_n16_per_expert = std::max(
            out.max_active_n16_per_expert, active_n16);
        out.max_paired_n16_per_expert = std::max(
            out.max_paired_n16_per_expert, paired_n16);
    }
    return out;
}

__global__ void init_W13_preflipped(uint32_t* p, uint64_t n) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; x < n; x += (uint64_t)blockDim.x * gridDim.x) {
        uint32_t q = ((x & 1u) ? 10u : (((uint32_t)(x >> 12) & 3u) + 9u));
        p[x] = q * 0x11111111u;
    }
}

__global__ void init_W2_native(uint32_t* p, uint64_t n) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; x < n; x += (uint64_t)blockDim.x * gridDim.x)
        p[x] = (1u + ((uint32_t)x & 3u)) * 0x11111111u;
}

__global__ void init_u32(uint32_t* p, uint64_t n, uint32_t value) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; x < n; x += (uint64_t)blockDim.x * gridDim.x) p[x] = value;
}

__global__ void init_u16(uint16_t* p, uint64_t n, uint16_t value) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; x < n; x += (uint64_t)blockDim.x * gridDim.x) p[x] = value;
}

static cudaGraphNode_t add_kernel(
    cudaGraph_t graph,
    const std::vector<cudaGraphNode_t>& deps,
    void* fn,
    dim3 grid,
    dim3 block,
    size_t smem,
    void** args
) {
    cudaKernelNodeParams p{};
    p.func = fn;
    p.gridDim = grid;
    p.blockDim = block;
    p.sharedMemBytes = smem;
    p.kernelParams = args;
    cudaGraphNode_t node;
    CUDA_CHECK(cudaGraphAddKernelNode(
        &node, graph, deps.empty() ? nullptr : deps.data(), deps.size(), &p));
    return node;
}

static GraphExec build_act_graph(const Shape& s, const Buffers& b) {
    GraphExec out;
    CUDA_CHECK(cudaGraphCreate(&out.graph, 0));
    uint64_t pairs = (uint64_t)s.N * s.H / 2u;
    const uint32_t* X = reinterpret_cast<const uint32_t*>(b.X);
    uint32_t* partial = b.partial;
    void* a0[] = {&X, &pairs, &partial};
    cudaGraphNode_t partial_node = add_kernel(
        out.graph, {}, (void*)moe_act_absmax_partial,
        s.abs_blocks, MOE_PREAMBLE_CTA, 0, a0);

    uint32_t count = s.abs_blocks;
    MoeActScale* act_scale = b.act_scale;
    void* a1[] = {&partial, &count, &act_scale};
    cudaGraphNode_t scale_node = add_kernel(
        out.graph, {partial_node}, (void*)moe_act_scale_finalize,
        1, MOE_PREAMBLE_CTA, 0, a1);

    (void)scale_node;
    return out;
}

static GraphExec build_route_graph(const Shape& s, const Buffers& b) {
    GraphExec out;
    CUDA_CHECK(cudaGraphCreate(&out.graph, 0));
    const int32_t* topk_idx = b.topk_idx;
    const uint16_t* topk_W = b.topk_W;
    uint16_t* topk_off = b.topk_off;
    uint16_t* expert_topk_W = b.expert_topk_W;
    uint16_t* expert_token_idx = b.expert_token_idx;
    int N = s.N, NP = s.NP, TOPK = s.TOPK;
    void* r0[] = {&topk_idx, &topk_W, &topk_off, &expert_topk_W,
                  &expert_token_idx, &N, &NP, &TOPK};
    add_kernel(out.graph, {}, (void*)moe_route_pack_contiguous,
               s.E, MOE_PREAMBLE_CTA, 0, r0);
    return out;
}

static GraphExec build_expert_graph(const Shape& s, const Buffers& b) {
    GraphExec out;
    CUDA_CHECK(cudaGraphCreate(&out.graph, 0));

    const uint32_t* X = reinterpret_cast<const uint32_t*>(b.X);
    const int32_t* topk_idx = b.topk_idx;
    const uint16_t* topk_off = b.topk_off;
    MoeActScale* act_scale = b.act_scale;
    uint32_t *X4out = b.X4, *Sxout = b.Sx;
    int N = s.N, NP = s.NP, I = s.I, H = s.H, TOPK = s.TOPK;
    void* pa[] = {&X, &act_scale, &topk_idx, &topk_off, &X4out, &Sxout,
                  &N, &NP, &H, &TOPK};
    cudaGraphNode_t pack = add_kernel(
        out.graph, {}, (void*)moe_act_pack_expert_contiguous,
        (s.H >> 6) * (s.NP >> 4), MOE_PREAMBLE_CTA, 0, pa);

    cudaMemsetParams z{};
    z.dst = b.O;
    z.elementSize = 1;
    z.width = (uint64_t)s.H * s.NP * sizeof(uint16_t);
    z.height = 1;
    cudaGraphNode_t zero;
    CUDA_CHECK(cudaGraphAddMemsetNode(&zero, out.graph, nullptr, 0, &z));

    const uint32_t *W13 = b.W13, *S13 = b.S13;
    const __nv_bfloat16* W13GS
        = reinterpret_cast<const __nv_bfloat16*>(b.W13GS);
    const uint32_t *X4 = b.X4, *Sx = b.Sx;
    const uint32_t* XGSINV2 = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const char*>(b.act_scale)
        + offsetof(MoeActScale, XGSINV2));
    const __nv_bfloat16* expert_topk_W
        = reinterpret_cast<const __nv_bfloat16*>(b.expert_topk_W);
    uint32_t *Y4 = b.Y4, *SY = b.SY, *YGSINV = b.YGSINV;
    const uint16_t* expert_token_idx = b.expert_token_idx;
    void* f1a[] = {&W13, &S13, &W13GS, &X4, &Sx, &XGSINV2,
                   &expert_topk_W, &expert_token_idx, &Y4, &SY, &YGSINV,
                   &NP, &I, &H};
    cudaGraphNode_t f1 = add_kernel(
        out.graph, {pack}, (void*)xR57F1_contiguous_graph,
        s.E, CTA, F1Y4_SMEM_BYTES, f1a);

    const uint32_t *W2 = b.W2, *S2 = b.S2;
    const __nv_bfloat16* W2GS
        = reinterpret_cast<const __nv_bfloat16*>(b.W2GS);
    const uint32_t *Y4in = b.Y4, *SYin = b.SY, *YGSINVin = b.YGSINV;
    __nv_bfloat16* O = reinterpret_cast<__nv_bfloat16*>(b.O);
    void* f2a[] = {&W2, &S2, &W2GS, &Y4in, &SYin, &YGSINVin,
                   &expert_token_idx, &O, &NP, &I, &H};
    add_kernel(out.graph, {f1, zero}, (void*)xR57F2_direct_reduce_graph,
               s.E, F2_CTA, F2_CTA * 8u * sizeof(uint32_t), f2a);

    return out;
}

static GraphExec build_e2e_graph(
    const GraphExec& act,
    const GraphExec& route,
    const GraphExec& expert
) {
    GraphExec out;
    CUDA_CHECK(cudaGraphCreate(&out.graph, 0));
    cudaGraphNode_t act_node, route_node, expert_node;
    CUDA_CHECK(cudaGraphAddChildGraphNode(
        &act_node, out.graph, nullptr, 0, act.graph));
    CUDA_CHECK(cudaGraphAddChildGraphNode(
        &route_node, out.graph, nullptr, 0, route.graph));
    cudaGraphNode_t preamble[] = {act_node, route_node};
    CUDA_CHECK(cudaGraphAddChildGraphNode(
        &expert_node, out.graph, preamble, 2, expert.graph));
    CUDA_CHECK(cudaGraphInstantiate(&out.exec, out.graph, nullptr, nullptr, 0));
    return out;
}

static double time_graph(GraphExec& graph, cudaStream_t stream, int iters) {
    for (int i = 0; i < 3; i++) CUDA_CHECK(cudaGraphLaunch(graph.exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> times(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        CUDA_CHECK(cudaGraphLaunch(graph.exec, stream));
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }
    std::sort(times.begin(), times.end());
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return times[times.size() / 2];
}

static void destroy_graph(GraphExec& graph) {
    if (graph.exec) CUDA_CHECK(cudaGraphExecDestroy(graph.exec));
    if (graph.graph) CUDA_CHECK(cudaGraphDestroy(graph.graph));
}

int main(int argc, char** argv) {
    if (argc != 10) {
        std::fprintf(stderr,
            "usage: %s E N I H TOPK {uniform|zipf|burst} seed iters output.csv\n",
            argv[0]);
        return 2;
    }

    Shape s{};
    s.E = std::atoi(argv[1]);
    s.N = std::atoi(argv[2]);
    s.I = std::atoi(argv[3]);
    s.H = std::atoi(argv[4]);
    s.TOPK = std::atoi(argv[5]);
    std::string routing = argv[6];
    uint32_t seed = (uint32_t)std::strtoul(argv[7], nullptr, 0);
    int iters = std::atoi(argv[8]);
    const char* csv_path = argv[9];
    // X4 is n16-packed, but Sx/SY are scale-quad n32 layouts.
    s.NP = (s.N + 31) & ~31;
    uint64_t pairs = (uint64_t)s.N * s.H / 2u;
    s.abs_blocks = (uint32_t)std::max<uint64_t>(
        1u, std::min<uint64_t>(4096u,
            (pairs + MOE_PREAMBLE_CTA * 8u - 1u)
            / (MOE_PREAMBLE_CTA * 8u)));

    if (s.E < s.TOPK || s.N <= 0 || s.N > 65535
        || s.I <= 0 || s.H <= 0
        || (s.I & 63) || (s.H & 63) || s.TOPK <= 0 || s.TOPK > 8
        || iters <= 0) {
        std::fprintf(stderr,
            "requires E>=TOPK, 0<N<=65535, I%%64=H%%64=0, "
            "1<=TOPK<=8, iters>0\n");
        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 12) {
        std::fprintf(stderr, "requires SM120+, got %d.%d\n",
                     prop.major, prop.minor);
        return 2;
    }

    auto routes = make_routes(s.E, s.N, s.TOPK, routing, seed);
    std::vector<int32_t> hTopkIdx;
    std::vector<uint16_t> hTopkW;
    std::vector<uint16_t> hTopkOffRef;
    std::vector<uint16_t> hExpertTopkRef;
    std::vector<uint16_t> hExpertTokenRef;
    RouteStats rs = encode_topk_inputs(
        routes, s, hTopkIdx, hTopkW, hTopkOffRef, hExpertTopkRef,
        hExpertTokenRef);
    std::string route_path = std::string(csv_path) + ".routes.i32";
    FILE* route_file = std::fopen(route_path.c_str(), "wb");
    if (!route_file
        || std::fwrite(hTopkIdx.data(), sizeof(int32_t), hTopkIdx.size(),
                       route_file) != hTopkIdx.size()) {
        std::perror(route_path.c_str());
        if (route_file) std::fclose(route_file);
        return 2;
    }
    std::fclose(route_file);
    std::vector<uint16_t> hX((uint64_t)s.N * s.H);
    for (uint64_t x = 0; x < hX.size(); x++) {
        uint32_t r = mix32((uint32_t)x ^ (uint32_t)(x >> 32) ^ seed);
        hX[x] = bf16_raw(float(int(r & 2047u) - 1024) * (1.0f / 1024.0f));
    }

    uint64_t I64 = (uint64_t)s.I >> 6;
    uint64_t H64 = (uint64_t)s.H >> 6;
    uint64_t W13_u32 = (uint64_t)s.E * H64 * ((uint64_t)s.I << 4);
    uint64_t S13_u32 = (uint64_t)s.E * H64 * ((uint64_t)s.I << 2);
    uint64_t W2_u32 = (uint64_t)s.E * I64 * ((uint64_t)s.H << 3);
    uint64_t S2_u32 = (uint64_t)s.E * I64 * ((uint64_t)s.H << 1);
    uint64_t X4_u32 = (uint64_t)s.E * s.NP * ((uint64_t)s.H >> 3);
    uint64_t Sx_u32 = (uint64_t)s.E * s.NP * H64;
    uint64_t Y4_u32 = (uint64_t)s.E * s.NP * I64 * 8u;
    uint64_t SY_final_u32 = (uint64_t)s.E * s.NP * I64;
    uint64_t SY_alloc_u32 = SY_final_u32 * 3u;

    uint64_t requested = (W13_u32 + S13_u32 + W2_u32 + S2_u32
                         + X4_u32 + Sx_u32 + Y4_u32 + SY_alloc_u32) * 4u
                       + ((uint64_t)s.N * s.H + (uint64_t)s.H * s.NP) * 2u
                       + (uint64_t)s.E * (3u * sizeof(uint32_t)
                           + 3u * sizeof(uint16_t))
                       + (uint64_t)s.E * (s.NP + 1u) * sizeof(uint16_t)
                       + (uint64_t)s.E * s.NP * sizeof(uint16_t)
                       + (uint64_t)s.N * s.TOPK
                           * (sizeof(int32_t) + 2u * sizeof(uint16_t))
                       + (uint64_t)s.abs_blocks * sizeof(uint32_t);
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (requested > (uint64_t)free_bytes * 9u / 10u) {
        std::fprintf(stderr,
            "device allocation needs %.2f GiB, only %.2f GiB free\n",
            requested / double(1ull << 30), free_bytes / double(1ull << 30));
        return 2;
    }

    Buffers b;
    CUDA_CHECK(cudaMalloc(&b.W13, W13_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.S13, S13_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.W13GS, (uint64_t)s.E * 2u * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.W2, W2_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.S2, S2_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.W2GS, (uint64_t)s.E * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.X, hX.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.topk_idx, hTopkIdx.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&b.topk_W, hTopkW.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.X4, X4_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.Sx, Sx_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.topk_off,
                          hTopkOffRef.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.expert_topk_W,
                          hExpertTopkRef.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.expert_token_idx,
                          hExpertTokenRef.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.Y4, Y4_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.SY, SY_alloc_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.YGSINV, (uint64_t)s.E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.O, (uint64_t)s.H * s.NP * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&b.partial, s.abs_blocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&b.act_scale, sizeof(MoeActScale)));

    CUDA_CHECK(cudaMemcpy(b.X, hX.data(), hX.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.topk_idx, hTopkIdx.data(),
                          hTopkIdx.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b.topk_W, hTopkW.data(),
                          hTopkW.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(b.Y4, 0, Y4_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(b.SY, 0, SY_alloc_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(b.YGSINV, 0, (uint64_t)s.E * sizeof(uint32_t)));
    init_W13_preflipped<<<4096, 256>>>(b.W13, W13_u32);
    init_u32<<<4096, 256>>>(b.S13, S13_u32, 0x38383838u);
    init_u16<<<256, 256>>>(b.W13GS, (uint64_t)s.E * 2u, 0x3c80u);
    init_W2_native<<<4096, 256>>>(b.W2, W2_u32);
    init_u32<<<4096, 256>>>(b.S2, S2_u32, 0x38383838u);
    init_u16<<<256, 256>>>(b.W2GS, (uint64_t)s.E, 0x3c80u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFuncSetAttribute(
        xR57F1_contiguous_graph, cudaFuncAttributeMaxDynamicSharedMemorySize,
        F1Y4_SMEM_BYTES));
    CUDA_CHECK(cudaFuncSetAttribute(
        xR57F2_direct_reduce_graph, cudaFuncAttributeMaxDynamicSharedMemorySize,
        F2_CTA * 8u * sizeof(uint32_t)));

    GraphExec act = build_act_graph(s, b);
    GraphExec route = build_route_graph(s, b);
    GraphExec expert = build_expert_graph(s, b);
    GraphExec e2e = build_e2e_graph(act, route, expert);
    cudaStream_t join_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&join_stream, cudaStreamNonBlocking));

    double e2e_ms = time_graph(e2e, join_stream, iters);

    CUDA_CHECK(cudaGraphLaunch(e2e.exec, join_stream));
    CUDA_CHECK(cudaStreamSynchronize(join_stream));

    std::vector<uint16_t> hTopkOff(hTopkOffRef.size());
    std::vector<uint16_t> hExpertTopk(hExpertTopkRef.size());
    std::vector<uint16_t> hExpertToken(hExpertTokenRef.size());
    std::vector<uint32_t> hYGSINV(s.E);
    std::vector<uint16_t> hO((uint64_t)s.H * s.NP);
    CUDA_CHECK(cudaMemcpy(hTopkOff.data(), b.topk_off,
                          hTopkOff.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hExpertTopk.data(), b.expert_topk_W,
                          hExpertTopk.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hExpertToken.data(), b.expert_token_idx,
                          hExpertToken.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hYGSINV.data(), b.YGSINV,
                          hYGSINV.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hO.data(), b.O, hO.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    if (std::getenv("XCALIBER_DUMP_OUTPUT")) {
        std::string output_path = std::string(csv_path) + ".out.bf16";
        FILE* output_file = std::fopen(output_path.c_str(), "wb");
        if (!output_file
            || std::fwrite(hO.data(), sizeof(uint16_t), hO.size(), output_file)
                != hO.size()) {
            std::perror(output_path.c_str());
            if (output_file) std::fclose(output_file);
            return 2;
        }
        std::fclose(output_file);
    }

    uint64_t route_errors = 0;
    for (uint64_t x = 0; x < hTopkOff.size(); x++)
        route_errors += hTopkOff[x] != hTopkOffRef[x];
    for (uint64_t x = 0; x < hExpertTopk.size(); x++)
        route_errors += hExpertTopk[x] != hExpertTopkRef[x];
    for (uint64_t x = 0; x < hExpertToken.size(); x++)
        route_errors += hExpertToken[x] != hExpertTokenRef[x];
    uint64_t scale_errors = 0;
    for (int e = 0; e < s.E; e++) {
        bool live = hExpertTokenRef[(uint64_t)e * (s.NP + 1u)] != 0u;
        uint16_t lo = (uint16_t)hYGSINV[e];
        uint16_t hi = (uint16_t)(hYGSINV[e] >> 16);
        scale_errors += live
            ? (!lo || lo != hi || (lo & 0x7f80u) == 0x7f80u)
            : hYGSINV[e] != 0u;
    }
    uint64_t nonfinite = 0, padded_nonzero = 0, useful_nonzero = 0;
    uint64_t checksum = 1469598103934665603ull;
    for (int h = 0; h < s.H; h++) {
        for (int n = 0; n < s.NP; n++) {
            uint16_t v = hO[(uint64_t)n * s.H + h];
            nonfinite += (v & 0x7f80u) == 0x7f80u;
            padded_nonzero += n >= s.N && v != 0u;
            useful_nonzero += n < s.N && v != 0u;
            checksum = (checksum ^ v) * 1099511628211ull;
        }
    }
    int f1_blocks = 0, f2_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &f1_blocks, xR57F1_contiguous_graph, CTA, F1Y4_SMEM_BYTES));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &f2_blocks, xR57F2_direct_reduce_graph, F2_CTA,
        F2_CTA * 8u * sizeof(uint32_t)));
    bool pass = !route_errors && !scale_errors && !nonfinite
             && !padded_nonzero && useful_nonzero && f1_blocks == 1
             && f2_blocks >= 1;
    double useful_tflops = 6.0 * double(rs.assignments) * s.I * s.H
                          / e2e_ms / 1.0e9;

    FILE* csv = std::fopen(csv_path, "w");
    if (!csv) {
        std::perror(csv_path);
        return 2;
    }
    std::fprintf(csv,
        "gpu,sms,mode,routing,seed,iters,E,N,NP,I,H,TOPK,live_experts,"
        "assignments,active_n8,active_n16,paired_n16,max_tokens_per_expert,"
        "max_active_n8_per_expert,max_active_n16_per_expert,"
        "max_paired_n16_per_expert,end_to_end_ms,useful_tflops,"
        "route_errors,scale_errors,"
        "nonfinite,padded_nonzero,useful_nonzero,f1_blocks_per_sm,"
        "f2_blocks_per_sm,checksum,validation_scope,status\n");
    std::fprintf(csv,
        "%s,%d,contiguous_direct_ff1_ff2_cp_reduce_w4a4,%s,0x%08x,%d,%d,%d,%d,%d,%d,%d,%d,"
        "%llu,%llu,%llu,%llu,%u,%u,%u,%u,%.6f,%.6f,"
        "%llu,%llu,%llu,%llu,%llu,%d,%d,0x%016llx,route_scale_finite,%s\n",
        prop.name, prop.multiProcessorCount, routing.c_str(), seed, iters,
        s.E, s.N, s.NP, s.I, s.H, s.TOPK, rs.live_experts,
        (unsigned long long)rs.assignments,
        (unsigned long long)rs.active_n8,
        (unsigned long long)rs.active_n16,
        (unsigned long long)rs.paired_n16,
        rs.max_tokens_per_expert,
        rs.max_active_n8_per_expert,
        rs.max_active_n16_per_expert,
        rs.max_paired_n16_per_expert,
        e2e_ms, useful_tflops,
        (unsigned long long)route_errors,
        (unsigned long long)scale_errors,
        (unsigned long long)nonfinite,
        (unsigned long long)padded_nonzero,
        (unsigned long long)useful_nonzero,
        f1_blocks, f2_blocks, (unsigned long long)checksum,
        pass ? "PASS" : "FAIL");
    std::fclose(csv);

    std::printf("+----------------------+------------+\n");
    std::printf("| path                 | median ms  |\n");
    std::printf("+----------------------+------------+\n");
    std::printf("| END TO END ONLY     | %10.6f |\n", e2e_ms);
    std::printf("+----------------------+------------+\n");
    std::printf("status=%s useful_tflops=%.3f route_errors=%llu "
                "scale_errors=%llu nonfinite=%llu csv=%s\n",
                pass ? "PASS" : "FAIL", useful_tflops,
                (unsigned long long)route_errors,
                (unsigned long long)scale_errors,
                (unsigned long long)nonfinite, csv_path);

    destroy_graph(e2e);
    destroy_graph(act);
    destroy_graph(route);
    destroy_graph(expert);
    CUDA_CHECK(cudaStreamDestroy(join_stream));
    cudaFree(b.W13); cudaFree(b.S13); cudaFree(b.W13GS);
    cudaFree(b.W2); cudaFree(b.S2); cudaFree(b.W2GS);
    cudaFree(b.X); cudaFree(b.topk_idx); cudaFree(b.topk_W);
    cudaFree(b.X4); cudaFree(b.Sx); cudaFree(b.topk_off);
    cudaFree(b.expert_topk_W); cudaFree(b.expert_token_idx);
    cudaFree(b.Y4); cudaFree(b.SY);
    cudaFree(b.YGSINV); cudaFree(b.O); cudaFree(b.partial);
    cudaFree(b.act_scale);
    return pass ? 0 : 3;
}
