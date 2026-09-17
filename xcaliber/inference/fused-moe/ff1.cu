#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <cfloat>
#include <cooperative_groups.h>
#include <type_traits>


__global__ void ff1(
    const __nv_bfloat16* W13,
    const __nv_bfloat16* X,
    __nv_bfloat16* Y,
    const int topk_idx,
    const int topk_weights,
    const int K,
    const int N,
    const int E,
    const int H,
    const int I
){
    
}