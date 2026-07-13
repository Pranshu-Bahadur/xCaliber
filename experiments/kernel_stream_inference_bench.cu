#include "kernel_stream.cu"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

#ifndef STREAM_REGCAP
#define STREAM_REGCAP 0
#endif

#define CUDA_CHECK(call) do {                                                \
    cudaError_t e__ = (call);                                                \
    if (e__ != cudaSuccess) {                                                \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,            \
                     cudaGetErrorString(e__));                               \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

struct RouteStats {
    uint64_t assignments = 0;
    uint64_t active_n8 = 0;
    uint64_t active_n256 = 0;
    int live = 0;
    int min_tokens = 0;
    int p50_tokens = 0;
    int p95_tokens = 0;
    int max_tokens = 0;
    double mean_tokens = 0.0;
    double cv = 0.0;
    double gini = 0.0;
    double sm_tail_ratio = 0.0;
};

__global__ void init_nvfp4(
    uint32_t* p, uint64_t n, uint32_t seed, uint32_t flipped
) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t v = (uint32_t)x ^ (uint32_t)(x >> 32) ^ seed;
        v ^= v >> 16;
        v *= 0x7feb352du;
        v ^= v >> 15;
        p[x] = (flipped ? 0x88888888u : 0u) | (v & 0x33333333u);
    }
}

__global__ void init_ue4m3(uint32_t* p, uint64_t n, uint32_t seed) {
    uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
    for (; x < n; x += stride) {
        uint32_t v = (uint32_t)x ^ (uint32_t)(x >> 32) ^ seed;
        v ^= v >> 16;
        v *= 0x846ca68bu;
        v ^= v >> 15;
        p[x] = 0x30303030u | (v & 0x07070707u);
    }
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static std::vector<int> parse_ints(const char* raw) {
    std::vector<int> out;
    const char* p = raw;
    while (*p) {
        char* end = nullptr;
        long v = std::strtol(p, &end, 0);
        if (end == p || v <= 0) {
            std::fprintf(stderr, "bad integer list: %s\n", raw);
            std::exit(2);
        }
        out.push_back((int)v);
        p = end;
        if (*p == ',') p++;
        else if (*p) {
            std::fprintf(stderr, "bad integer list: %s\n", raw);
            std::exit(2);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static std::vector<std::string> parse_strings(const char* raw) {
    std::vector<std::string> out;
    std::string s(raw), item;
    size_t begin = 0;
    while (begin <= s.size()) {
        size_t end = s.find(',', begin);
        item = s.substr(begin, end == std::string::npos ? end : end - begin);
        if (!item.empty()) out.push_back(item);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return out;
}

static void add_unique(std::vector<int>& row, int e) {
    if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
}

static std::vector<std::vector<int>> make_routes(
    int E, int N, int topk, const std::string& mode, uint32_t seed
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
    int local_span = std::max(topk * 3, std::max(8, E / 16));
    for (int n = 0; n < N; n++) {
        if (mode == "burst" && !(n & 15)) burst_center = zipf(rng);
        while ((int)routes[n].size() < topk) {
            int e;
            if (mode == "uniform") {
                e = uniform(rng);
            } else if (mode == "zipf") {
                e = zipf(rng);
            } else if (mode == "burst") {
                int local_target = std::max(1, (topk * 3) / 4);
                if ((int)routes[n].size() < local_target) {
                    int delta = int(rng() % (uint32_t)local_span) - local_span / 2;
                    e = (burst_center + delta + E) % E;
                } else {
                    e = zipf(rng);
                }
            } else {
                std::fprintf(stderr, "unknown routing mode: %s\n", mode.c_str());
                std::exit(2);
            }
            add_unique(routes[n], e);
        }
    }
    return routes;
}

static RouteStats encode_routes(
    const std::vector<std::vector<int>>& routes,
    int E,
    int N,
    int NP,
    int topk,
    int Xb_stride,
    std::vector<uint32_t>& Xb,
    std::vector<__nv_bfloat16>& topk_W
) {
    Xb.assign((uint64_t)E * Xb_stride, 0u);
    topk_W.assign((uint64_t)E * NP, __float2bfloat16(0.0f));
    std::vector<int> counts(E, 0);
    std::vector<unsigned char> n8((uint64_t)E * ((NP + 7) >> 3), 0);
    std::vector<unsigned char> n256((uint64_t)E * ((NP + 255) >> 8), 0);
    __nv_bfloat16 tw = __float2bfloat16(1.0f / float(topk));

    for (int n = 0; n < N; n++) {
        for (int e : routes[n]) {
            counts[e]++;
            Xb[(uint64_t)e * Xb_stride] = 1u;
            Xb[(uint64_t)e * Xb_stride + 1u + ((uint32_t)n >> 5)]
                |= 1u << (n & 31);
            topk_W[(uint64_t)e * NP + n] = tw;
            n8[(uint64_t)e * ((NP + 7) >> 3) + (n >> 3)] = 1;
            n256[(uint64_t)e * ((NP + 255) >> 8) + (n >> 8)] = 1;
        }
    }

    RouteStats st;
    st.assignments = (uint64_t)N * topk;
    st.active_n8 = std::accumulate(n8.begin(), n8.end(), uint64_t(0));
    st.active_n256 = std::accumulate(n256.begin(), n256.end(), uint64_t(0));
    std::vector<int> live_counts;
    for (int c : counts) if (c) live_counts.push_back(c);
    st.live = (int)live_counts.size();
    std::sort(live_counts.begin(), live_counts.end());
    if (!live_counts.empty()) {
        st.min_tokens = live_counts.front();
        st.p50_tokens = live_counts[(live_counts.size() - 1) / 2];
        st.p95_tokens = live_counts[(live_counts.size() - 1) * 95 / 100];
        st.max_tokens = live_counts.back();
        st.mean_tokens = double(st.assignments) / st.live;
        double var = 0.0;
        for (int c : live_counts) {
            double d = c - st.mean_tokens;
            var += d * d;
        }
        st.cv = std::sqrt(var / st.live) / st.mean_tokens;
        double weighted = 0.0;
        for (size_t i = 0; i < live_counts.size(); i++)
            weighted += double(i + 1) * live_counts[i];
        st.gini = (2.0 * weighted) / (st.live * double(st.assignments))
                - (double(st.live) + 1.0) / st.live;
    }

    // Greedy persistent-scheduler proxy: each next expert takes the least-loaded SM.
    std::priority_queue<double, std::vector<double>, std::greater<double>> sm;
    for (int i = 0; i < 188; i++) sm.push(0.0);
    for (int e = 0; e < E; e++) {
        double load = sm.top(); sm.pop();
        sm.push(load + counts[e]);
    }
    double sum = 0.0, tail = 0.0;
    while (!sm.empty()) {
        sum += sm.top();
        tail = std::max(tail, sm.top());
        sm.pop();
    }
    st.sm_tail_ratio = sum ? tail / (sum / 188.0) : 0.0;
    return st;
}

static uint64_t checksum(const std::vector<__nv_bfloat16>& values) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const auto& v : values) {
        uint16_t raw;
        std::memcpy(&raw, &v, sizeof(raw));
        h = (h ^ raw) * 0x100000001b3ull;
    }
    return h;
}

static const char* mode_name(int) {
    return "n256_outer_stream";
}

static int mode_smem(int) {
    return (int)STREAM_SMEM_BYTES;
}

static void set_attributes() {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel_stream, cudaFuncAttributeMaxDynamicSharedMemorySize,
        STREAM_SMEM_BYTES));
}

static void launch(
    int mode,
    int E,
    const uint32_t* W13,
    const uint32_t* S13,
    const __nv_bfloat16* W13GS,
    const uint32_t* X4,
    const uint32_t* SX,
    __nv_bfloat16 XGSINV,
    const uint32_t* Xb,
    const __nv_bfloat16* topk_W,
    __nv_bfloat16* Y,
    int Xb_stride,
    int NP,
    int IP,
    int I,
    int H
) {
    int smem = mode_smem(mode);
    kernel_stream<<<E, STREAM_CTA, smem>>>(
        W13, S13, W13GS, X4, SX, XGSINV, Xb, topk_W, Y,
        Xb_stride, NP, IP, I, H);
}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::fprintf(stderr,
            "usage: %s model E I H TOPK N[,N] repeats seed routes csv\n",
            argv[0]);
        return 2;
    }

    std::string model = argv[1];
    int E = std::atoi(argv[2]);
    int I = std::atoi(argv[3]);
    int H = std::atoi(argv[4]);
    int topk = std::atoi(argv[5]);
    std::vector<int> Ns = parse_ints(argv[6]);
    int repeats = std::atoi(argv[7]);
    uint32_t seed = (uint32_t)std::strtoul(argv[8], nullptr, 0);
    std::vector<std::string> routes = parse_strings(argv[9]);
    const char* csv_path = argv[10];
    int IP = (I + 63) & ~63;
    int maxN = *std::max_element(Ns.begin(), Ns.end());
    int maxNP = (maxN + 7) & ~7;
    int H64 = H >> 6;

    if (E <= 0 || I <= 0 || (H & 63) || topk <= 0 || topk > E
        || repeats <= 0 || routes.empty()) {
        std::fprintf(stderr,
            "requires E,I,TOPK,repeats>0, TOPK<=E, H%%64=0\n");
        return 2;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major < 12) {
        std::fprintf(stderr, "requires SM120+, got %d.%d\n", prop.major, prop.minor);
        return 2;
    }

    uint64_t W_u32 = (uint64_t)E * H64 * (IP << 4);
    uint64_t S_u32 = (uint64_t)E * H64 * (IP << 2);
    uint64_t X_u32 = (uint64_t)maxNP * H64 * 8u;
    uint64_t SX_u32 = (uint64_t)maxNP * H64;
    uint64_t Y_bf16 = (uint64_t)IP * maxNP;
    int max_Xb_stride = 1 + ((maxNP + 31) >> 5);

    uint32_t *dW = nullptr, *dS = nullptr, *dX4 = nullptr, *dSX = nullptr;
    uint32_t* dXb = nullptr;
    __nv_bfloat16 *dWGS = nullptr, *dTopk = nullptr, *dY = nullptr;
    CUDA_CHECK(cudaMalloc(&dW, W_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dS, S_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dWGS, (uint64_t)E * 2u * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dX4, X_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dSX, SX_u32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dXb,
        (uint64_t)E * max_Xb_stride * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&dTopk,
        (uint64_t)E * maxNP * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&dY, Y_bf16 * sizeof(__nv_bfloat16)));

    // Index-varying format-valid packets expose resident/direct address drift.
    // W is already E2M1 sign-flipped: positive 0..1.5 -> nibbles 0x8..0xb.
    init_nvfp4<<<4096, 256>>>(dW, W_u32, 0x13a57e01u, 1u);
    init_ue4m3<<<4096, 256>>>(dS, S_u32, 0x51ca1e02u);
    init_nvfp4<<<4096, 256>>>(dX4, X_u32, 0xa47c1e03u, 0u);
    init_ue4m3<<<4096, 256>>>(dSX, SX_u32, 0x5ca1e004u);
    CUDA_CHECK(cudaGetLastError());
    std::vector<__nv_bfloat16> WGS((uint64_t)E * 2u);
    __nv_bfloat16 gs = __float2bfloat16(1.0f / std::sqrt((float)H));
    std::fill(WGS.begin(), WGS.end(), gs);
    CUDA_CHECK(cudaMemcpy(dWGS, WGS.data(),
        WGS.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    set_attributes();

    FILE* csv = std::fopen(csv_path, "w");
    if (!csv) {
        std::perror(csv_path);
        return 2;
    }
    std::fprintf(csv,
        "gpu,sms,model,routing,route_seed,E,N,NP,I,IP,H,H64,TOPK,mode,"
        "resident,live_pf,reg_cap,cta_waves,live_experts,empty_experts,"
        "assignments,tokens_min,tokens_p50,tokens_p95,tokens_max,tokens_mean,"
        "tokens_cv,gini,active_n8,active_n256,sm_tail_ratio,smem_bytes,"
        "blocks_per_sm,H_resident_pct,W_requested_bytes,S_requested_bytes,"
        "X_unique_bytes,X_ld_requested_bytes,pf_requested_bytes,reduce_bytes,"
        "mma_inst,useful_tflops,issued_tflops,mma_useful_pct,kernel_ms,"
        "kernel_min_ms,kernel_p25_ms,kernel_p75_ms,kernel_max_ms,timing_rel_iqr,"
        "speedup_vs_stream,validation_scope,max_abs_err,max_rel_err,checksum,status\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    __nv_bfloat16 XGSINV = __float2bfloat16(1.0f);

    for (int N : Ns) {
        int NP = (N + 7) & ~7;
        int Xb_stride = 1 + ((N + 31) >> 5);
        for (size_t route_i = 0; route_i < routes.size(); route_i++) {
            uint32_t route_seed = mix32(seed ^ (uint32_t)N
                                      ^ (uint32_t)(route_i * 0x9e3779b9u));
            auto route = make_routes(E, N, topk, routes[route_i], route_seed);
            std::vector<uint32_t> Xb;
            std::vector<__nv_bfloat16> topk_W;
            RouteStats rs = encode_routes(
                route, E, N, NP, topk, Xb_stride, Xb, topk_W);
            CUDA_CHECK(cudaMemcpy(dXb, Xb.data(),
                Xb.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dTopk, topk_W.data(),
                topk_W.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

            double stream_ms = 0.0;
            for (int mode = 0; mode < 1; mode++) {
                CUDA_CHECK(cudaMemset(dY, 0,
                    (uint64_t)IP * NP * sizeof(__nv_bfloat16)));
                launch(mode, E, dW, dS, dWGS, dX4, dSX, XGSINV,
                       dXb, dTopk, dY, Xb_stride, NP, IP, I, H);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<float> times(repeats);
                for (int rep = 0; rep < repeats; rep++) {
                    CUDA_CHECK(cudaMemset(dY, 0,
                        (uint64_t)IP * NP * sizeof(__nv_bfloat16)));
                    CUDA_CHECK(cudaEventRecord(start));
                    launch(mode, E, dW, dS, dWGS, dX4, dSX, XGSINV,
                           dXb, dTopk, dY, Xb_stride, NP, IP, I, H);
                    CUDA_CHECK(cudaEventRecord(stop));
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    CUDA_CHECK(cudaEventElapsedTime(&times[rep], start, stop));
                }
                std::sort(times.begin(), times.end());
                double kernel_ms = times[times.size() / 2];
                double kernel_min_ms = times.front();
                double kernel_p25_ms = times[(times.size() - 1) / 4];
                double kernel_p75_ms = times[(times.size() - 1) * 3 / 4];
                double kernel_max_ms = times.back();
                double timing_rel_iqr = kernel_ms
                    ? (kernel_p75_ms - kernel_p25_ms) / kernel_ms : 0.0;
                if (!mode) stream_ms = kernel_ms;

                std::vector<__nv_bfloat16> output((uint64_t)IP * NP);
                CUDA_CHECK(cudaMemcpy(output.data(), dY,
                    output.size() * sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToHost));
                double max_abs = NAN, max_rel = NAN;
                uint64_t sum = checksum(output);

                int blocks_per_sm = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, kernel_stream, STREAM_CTA, mode_smem(mode)));

                uint64_t i64 = (uint64_t)IP >> 6;
                uint64_t W_bytes = rs.active_n8 * i64 * (uint64_t)H64 * 4096u;
                uint64_t S_bytes = rs.active_n8 * i64 * (uint64_t)H64 * 512u;
                uint64_t X_unique = rs.assignments * (uint64_t)H / 2u;
                uint64_t X_requested = rs.active_n8 * i64 * (uint64_t)H64 * 8192u;
                uint64_t pf_bytes = X_unique;
                uint64_t reduce_bytes = rs.active_n8 * i64 * (uint64_t)I * 16u;
                uint64_t mma_inst = rs.active_n8 * i64 * (uint64_t)H64 * 8u;
                double useful_flops = 4.0 * rs.assignments * (double)I * H;
                double issued_flops = mma_inst * (16.0 * 8.0 * 64.0 * 2.0);
                double useful_tf = useful_flops / (kernel_ms * 1.0e9);
                double issued_tf = issued_flops / (kernel_ms * 1.0e9);
                double useful_pct = issued_flops
                    ? 100.0 * useful_flops / issued_flops : 0.0;
                bool pass = blocks_per_sm == 1;

                std::fprintf(csv,
                    "%s,%d,%s,%s,0x%08x,%d,%d,%d,%d,%d,%d,%d,%d,%s,"
                    "%d,%d,%d,%d,%d,%d,%llu,%d,%d,%d,%d,%.6f,%.6f,%.6f,"
                    "%llu,%llu,%.6f,%d,%d,%.4f,%llu,%llu,%llu,%llu,%llu,"
                    "%llu,%llu,%.6f,%.6f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                    "%.6f,%.6f,%s,%.9g,%.9g,"
                    "0x%016llx,%s\n",
                    prop.name, prop.multiProcessorCount, model.c_str(),
                    routes[route_i].c_str(), route_seed, E, N, NP, I, IP, H,
                    H64, topk, mode_name(mode), 0, 1,
                    STREAM_REGCAP, (E + prop.multiProcessorCount - 1)
                        / prop.multiProcessorCount,
                    rs.live, E - rs.live,
                    (unsigned long long)rs.assignments,
                    rs.min_tokens, rs.p50_tokens, rs.p95_tokens, rs.max_tokens,
                    rs.mean_tokens, rs.cv, rs.gini,
                    (unsigned long long)rs.active_n8,
                    (unsigned long long)rs.active_n256,
                    rs.sm_tail_ratio, mode_smem(mode), blocks_per_sm,
                    0.0,
                    (unsigned long long)W_bytes,
                    (unsigned long long)S_bytes,
                    (unsigned long long)X_unique,
                    (unsigned long long)X_requested,
                    (unsigned long long)pf_bytes,
                    (unsigned long long)reduce_bytes,
                    (unsigned long long)mma_inst,
                    useful_tf, issued_tf, useful_pct, kernel_ms,
                    kernel_min_ms, kernel_p25_ms, kernel_p75_ms, kernel_max_ms,
                    timing_rel_iqr,
                    kernel_ms ? stream_ms / kernel_ms : 0.0, "timing_only",
                    max_abs, max_rel, (unsigned long long)sum,
                    pass ? "PASS" : "FAIL");
                std::fflush(csv);

                std::printf(
                    "%-20s %-7s N=%-4d live=%-4d mode=%-19s %9.4f ms "
                    "%7.3fx iqr=%6.3f%% validation=timing_only %s\n",
                    model.c_str(), routes[route_i].c_str(), N, rs.live,
                    mode_name(mode), kernel_ms,
                    kernel_ms ? stream_ms / kernel_ms : 0.0,
                    timing_rel_iqr * 100.0,
                    pass ? "PASS" : "FAIL");
            }
        }
    }

    std::fclose(csv);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dW));
    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dWGS));
    CUDA_CHECK(cudaFree(dX4));
    CUDA_CHECK(cudaFree(dSX));
    CUDA_CHECK(cudaFree(dXb));
    CUDA_CHECK(cudaFree(dTopk));
    CUDA_CHECK(cudaFree(dY));
    return 0;
}
