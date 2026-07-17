#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>

#include <cstdint>

#include "xR64ABSRouter.cu"
#include "xR64ABSPreamble.cu"
#include "xR64ABSFF1.cu"
#include "xR64ABSFF2.cu"
#include "xR64ABSRepack.cu"

static void xr64_check(
    const torch::Tensor& t,
    const torch::Tensor& x,
    const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.get_device() == x.get_device(),
                name, " must be on x.device");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__global__ void xR64ABSF32ToBF16(
    const float* __restrict__ x,
    uint16_t* __restrict__ y,
    uint64_t n) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t i = p << 1;
    if (i + 1u < n) {
        reinterpret_cast<uint32_t*>(y)[p]
            = (uint32_t)xr64_to_bf16(x[i])
            | ((uint32_t)xr64_to_bf16(x[i + 1u]) << 16);
    } else if (i < n) {
        y[i] = xr64_to_bf16(x[i]);
    }
}

__global__ void xR64ABSF16ToBF16(
    const uint16_t* __restrict__ x,
    uint16_t* __restrict__ y,
    uint64_t n) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n)
        y[p] = xr64_to_bf16(
            __half2float(reinterpret_cast<const __half*>(x)[p]));
}

__global__ void xR64ABSF32ToF16(
    const float* __restrict__ x,
    uint16_t* __restrict__ y,
    uint64_t n) {
    uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n)
        reinterpret_cast<__half*>(y)[p] = __float2half_rn(x[p]);
}

torch::Tensor xr64_abs_repack_gptq(torch::Tensor src) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous(),
                "qweight must be contiguous CUDA int32");
    TORCH_CHECK(src.scalar_type() == at::kInt && src.dim() == 3,
                "qweight must be int32 [E,K/8,O]");
    int E = src.size(0);
    int K = src.size(1) * 8;
    int O = src.size(2);
    TORCH_CHECK(!(K & 31) && !(O & 7),
                "repack requires K %% 32 == 0 and O %% 8 == 0");
    c10::cuda::CUDAGuard guard(src.device());
    auto dst = torch::empty({E, O, K >> 3}, src.options());
    launch_xr64_abs_repack_gptq(
        reinterpret_cast<const uint32_t*>(src.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(dst.data_ptr<int32_t>()),
        E, K, O, at::cuda::getCurrentCUDAStream(src.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
}

torch::Tensor xr64_abs_forward(
    torch::Tensor x,
    torch::Tensor router_logits,
    torch::Tensor per_expert_scale,
    torch::Tensor w13,
    torch::Tensor s13,
    torch::Tensor w2,
    torch::Tensor s2,
    bool gelu_tanh) {

    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2,
                "x must be contiguous CUDA BF16/FP16 [N,H]");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16
                || x.scalar_type() == at::kHalf,
                "x must be BF16 or FP16");
    c10::cuda::CUDAGuard guard(x.device());
    xr64_check(router_logits, x, "router_logits");
    xr64_check(per_expert_scale, x, "per_expert_scale");
    xr64_check(w13, x, "w13");
    xr64_check(s13, x, "s13");
    xr64_check(w2, x, "w2");
    xr64_check(s2, x, "s2");

    TORCH_CHECK(router_logits.scalar_type() == at::kBFloat16
                && router_logits.dim() == 2,
                "router_logits must be BF16 [N,E]");
    TORCH_CHECK(per_expert_scale.scalar_type() == at::kBFloat16
                && per_expert_scale.dim() == 1,
                "per_expert_scale must be BF16 [E]");
    TORCH_CHECK(w13.scalar_type() == at::kInt
                && w2.scalar_type() == at::kInt,
                "W13/W2 must be repacked int32 bit carriers");
    TORCH_CHECK(s13.scalar_type() == at::kFloat
                && s2.scalar_type() == at::kFloat,
                "S13/S2 must be FP32 K128 scales");

    int64_t N = x.size(0);
    int64_t H = x.size(1);
    if (!N) return torch::empty_like(x);
    TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3
                && s13.dim() == 3 && s2.dim() == 3,
                "weights and scales must be rank 3");
    int64_t E = w13.size(0);
    int64_t I = w13.size(1) >> 1;
    int64_t NS = (N + 15) & ~15ll;
    auto bf16 = x.options().dtype(at::kBFloat16);
    torch::Tensor xb = x;
    if (x.scalar_type() == at::kHalf) {
        xb = torch::empty(x.sizes(), bf16);
        uint64_t values = (uint64_t)N * (uint64_t)H;
        xR64ABSF16ToBF16<<<(values + 255u) / 256u, 256, 0,
                            at::cuda::getCurrentCUDAStream(x.get_device())>>>(
            reinterpret_cast<const uint16_t*>(x.data_ptr<at::Half>()),
            reinterpret_cast<uint16_t*>(xb.data_ptr<at::BFloat16>()),
            values);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    TORCH_CHECK(E > 0 && E <= 1024 && !(E & 1),
                "xR64ABS requires even E <= 1024");
    TORCH_CHECK(!(H & 31) && !(I & 31),
                "xR64ABS requires H and I divisible by 32");
    TORCH_CHECK(router_logits.size(0) == N
                && router_logits.size(1) == E,
                "router_logits must be [N,E]");
    TORCH_CHECK(per_expert_scale.numel() == E,
                "per_expert_scale must be [E]");
    TORCH_CHECK(w13.size(1) == 2 * I && w13.size(2) == H / 8,
                "W13 must be [E,2I,H/8]");
    TORCH_CHECK(w2.size(0) == E && w2.size(1) == H
                && w2.size(2) == I / 8,
                "W2 must be [E,H,I/8]");
    TORCH_CHECK(s13.size(0) == E && s13.size(1) == 2 * I
                && s13.size(2) == (H + 127) / 128,
                "S13 must be [E,2I,ceil(H/128)]");
    TORCH_CHECK(s2.size(0) == E && s2.size(1) == H
                && s2.size(2) == (I + 127) / 128,
                "S2 must be [E,H,ceil(I/128)]");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto i32 = x.options().dtype(at::kInt);
    auto f32 = x.options().dtype(at::kFloat);
    auto topk_idx = torch::empty({N, XR64_ABS_TOPK}, i32);
    auto topk_weight = torch::empty({N, XR64_ABS_TOPK}, bf16);
    auto topk_off = torch::empty({N, XR64_ABS_TOPK}, i32);
    auto route_weight = torch::empty({E, NS}, bf16);
    auto route_token = torch::empty({E, NS}, i32);
    auto count = torch::empty({E}, i32);
    auto x4 = torch::empty({E, NS, H / 8}, i32);
    auto sx = torch::empty({E, NS, H / 32}, f32);
    auto y4 = torch::empty({E, NS, I / 8}, i32);
    auto sy = torch::empty({E, NS, I / 32}, f32);
    auto y32 = torch::empty({N, H}, f32);
    auto y = torch::empty_like(x);
    C10_CUDA_CHECK(cudaMemsetAsync(
        y32.data_ptr(), 0, y32.nbytes(), stream));

    launch_xr64_abs_topk(
        reinterpret_cast<const uint16_t*>(
            router_logits.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(
            per_expert_scale.data_ptr<at::BFloat16>()),
        topk_idx.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(
            topk_weight.data_ptr<at::BFloat16>()),
        N, E, stream);
    launch_xr64_abs_route_pack(
        topk_idx.data_ptr<int32_t>(),
        reinterpret_cast<const uint16_t*>(
            topk_weight.data_ptr<at::BFloat16>()),
        topk_off.data_ptr<int32_t>(),
        reinterpret_cast<uint16_t*>(
            route_weight.data_ptr<at::BFloat16>()),
        route_token.data_ptr<int32_t>(), count.data_ptr<int32_t>(),
        N, NS, E, stream);
    launch_xr64_abs_preamble(
        reinterpret_cast<const uint16_t*>(xb.data_ptr<at::BFloat16>()),
        topk_idx.data_ptr<int32_t>(), topk_off.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(x4.data_ptr<int32_t>()),
        sx.data_ptr<float>(), N, NS, H, XR64_ABS_TOPK, stream);
    launch_xr64_abs_ff1(
        reinterpret_cast<const uint32_t*>(w13.data_ptr<int32_t>()),
        s13.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(x4.data_ptr<int32_t>()),
        sx.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(
            route_weight.data_ptr<at::BFloat16>()),
        count.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(y4.data_ptr<int32_t>()),
        sy.data_ptr<float>(), E, NS, I, H, gelu_tanh, stream);
    launch_xr64_abs_ff2(
        reinterpret_cast<const uint32_t*>(w2.data_ptr<int32_t>()),
        s2.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(y4.data_ptr<int32_t>()),
        sy.data_ptr<float>(), route_token.data_ptr<int32_t>(),
        count.data_ptr<int32_t>(), y32.data_ptr<float>(),
        E, NS, I, H, stream);

    uint64_t pairs = ((uint64_t)N * (uint64_t)H + 1u) >> 1;
    if (x.scalar_type() == at::kBFloat16) {
        xR64ABSF32ToBF16<<<(pairs + 255u) / 256u, 256, 0, stream>>>(
            y32.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(y.data_ptr<at::BFloat16>()),
            (uint64_t)N * (uint64_t)H);
    } else {
        xR64ABSF32ToF16<<<(pairs * 2u + 255u) / 256u, 256, 0, stream>>>(
            y32.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(y.data_ptr<at::Half>()),
            (uint64_t)N * (uint64_t)H);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &xr64_abs_forward,
          "xCaliber xR64ABS symmetric INT4 routed MoE");
    m.def("repack_gptq", &xr64_abs_repack_gptq,
          "AutoGPTQ uint4b8 -> xR64 MMA-native s4");
    m.def("build_info", [] {
        return "xR64ABS SM75/SM120 symmetric-W4G128 CTA1024 bridge";
    });
}
