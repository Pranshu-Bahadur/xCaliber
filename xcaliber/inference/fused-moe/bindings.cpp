#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <cfloat>


void topk(
    at::Tensor router_logits,
    at::Tensor topk_idx,
    at::Tensor topk_weights,
    const int64_t K,
    bool softmax
);

TORCH_LIBRARY(forge, m) {
    m.def("topk", &topk);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk", &topk);
}