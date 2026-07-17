#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "xR64ABS.cuh"

void xmoe_abs_operator(
    torch::Tensor X,
    torch::Tensor W13,
    torch::Tensor S13,
    torch::Tensor W2,
    torch::Tensor S2,
    torch::Tensor topk_idx,
    torch::Tensor topk_off,
    torch::Tensor route_token,
    torch::Tensor route_weight,
    torch::Tensor count,
    torch::Tensor X4,
    torch::Tensor SX,
    torch::Tensor Y4,
    torch::Tensor SY,
    torch::Tensor Y,
    bool gelu_tanh) {

    TORCH_CHECK(X.is_cuda() && W13.is_cuda() && W2.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(X.scalar_type() == at::kBFloat16,
                "X must be contiguous BF16 [N,H]");
    TORCH_CHECK(W13.scalar_type() == at::kInt
                && W2.scalar_type() == at::kInt,
                "W13/W2 must be packed int32 bit carriers");
    TORCH_CHECK(S13.scalar_type() == at::kFloat
                && S2.scalar_type() == at::kFloat,
                "S13/S2 must be FP32 K128 scales");
    TORCH_CHECK(topk_idx.scalar_type() == at::kInt
                && topk_off.scalar_type() == at::kInt
                && route_token.scalar_type() == at::kInt
                && count.scalar_type() == at::kInt,
                "routing tensors must be int32");
    TORCH_CHECK(route_weight.scalar_type() == at::kBFloat16,
                "route_weight must be BF16");
    TORCH_CHECK(X4.scalar_type() == at::kInt
                && Y4.scalar_type() == at::kInt
                && SX.scalar_type() == at::kFloat
                && SY.scalar_type() == at::kFloat
                && Y.scalar_type() == at::kFloat,
                "output buffer dtype mismatch");
    TORCH_CHECK(X.is_contiguous() && W13.is_contiguous()
                && S13.is_contiguous() && W2.is_contiguous()
                && S2.is_contiguous() && topk_idx.is_contiguous()
                && topk_off.is_contiguous() && route_token.is_contiguous()
                && route_weight.is_contiguous() && count.is_contiguous()
                && X4.is_contiguous() && SX.is_contiguous()
                && Y4.is_contiguous() && SY.is_contiguous()
                && Y.is_contiguous(), "all tensors must be contiguous");

    int N = X.size(0);
    int H = X.size(1);
    int E = count.numel();
    int NS = route_token.size(1);
    int TOPK = topk_idx.size(1);
    int I = Y4.size(2) * 8;
    int H128 = (H + 127) >> 7;
    int I128 = (I + 127) >> 7;

    TORCH_CHECK(X.dim() == 2 && !(H & 31) && !(I & 31),
                "H and I must be divisible by 32");
    TORCH_CHECK(topk_idx.dim() == 2 && topk_idx.size(0) == N
                && topk_off.sizes() == topk_idx.sizes(),
                "topk_idx/topk_off must be [N,TOPK]");
    TORCH_CHECK(route_token.dim() == 2 && route_token.size(0) == E
                && route_weight.sizes() == route_token.sizes(),
                "route_token/route_weight must be [E,NS]");
    TORCH_CHECK(W13.numel() == (int64_t)E * (I << 1) * (H >> 3)
                && S13.numel() == (int64_t)E * (I << 1) * H128,
                "W13/S13 layout must be [E,2I,H/8] / [E,2I,H128]");
    TORCH_CHECK(W2.numel() == (int64_t)E * H * (I >> 3)
                && S2.numel() == (int64_t)E * H * I128,
                "W2/S2 layout must be [E,H,I/8] / [E,H,I128]");
    TORCH_CHECK(X4.numel() == (int64_t)E * NS * (H >> 3)
                && SX.numel() == (int64_t)E * NS * (H >> 5),
                "X4/SX layout must be [E,NS,H/8] / [E,NS,H32]");
    TORCH_CHECK(Y4.numel() == (int64_t)E * NS * (I >> 3)
                && SY.numel() == (int64_t)E * NS * (I >> 5),
                "Y4/SY layout must be [E,NS,I/8] / [E,NS,I32]");
    TORCH_CHECK(Y.dim() == 2 && Y.size(0) == N && Y.size(1) == H,
                "Y must be FP32 [N,H]");

    c10::cuda::CUDAGuard guard(X.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemsetAsync(Y.data_ptr(), 0, Y.nbytes(), stream));

    launch_xr64_abs_preamble(
        reinterpret_cast<const uint16_t*>(X.data_ptr<at::BFloat16>()),
        topk_idx.data_ptr<int32_t>(), topk_off.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(X4.data_ptr<int32_t>()),
        SX.data_ptr<float>(), N, NS, H, TOPK, stream);
    launch_xr64_abs_ff1(
        reinterpret_cast<const uint32_t*>(W13.data_ptr<int32_t>()),
        S13.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(X4.data_ptr<int32_t>()),
        SX.data_ptr<float>(),
        reinterpret_cast<const uint16_t*>(route_weight.data_ptr<at::BFloat16>()),
        count.data_ptr<int32_t>(),
        reinterpret_cast<uint32_t*>(Y4.data_ptr<int32_t>()),
        SY.data_ptr<float>(), E, NS, I, H, gelu_tanh, stream);
    launch_xr64_abs_ff2(
        reinterpret_cast<const uint32_t*>(W2.data_ptr<int32_t>()),
        S2.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(Y4.data_ptr<int32_t>()),
        SY.data_ptr<float>(), route_token.data_ptr<int32_t>(),
        count.data_ptr<int32_t>(), Y.data_ptr<float>(),
        E, NS, I, H, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("operator", &xmoe_abs_operator,
          "xR64-shaped SM75 INT4 W4A4 MoE with ABS activation compression");
}
