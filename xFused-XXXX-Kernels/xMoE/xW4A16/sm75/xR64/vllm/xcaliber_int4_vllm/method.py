from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from . import _C


logger = init_logger(__name__)


class XCaliberR64ABSAutoRoundMoE(GPTQMarlinMoEMethod):
    """GPTQ loader plus xR64ABS monolithic routed executor."""

    def __init__(self, quant_config, moe_config) -> None:
        super().__init__(quant_config, moe_config)
        self._xr64_ready = False
        self._xr64_gelu_tanh = False
        self._xr64_per_expert_scale = None

    @property
    def supports_internal_mk(self) -> bool:
        return True

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.input_dtype = self.input_dtype
        self.is_k_full = True
        group = self.quant_config.group_size
        if group != 128:
            raise RuntimeError(f"xR64ABS requires K128 scales, got {group}")
        g13 = (hidden_size + group - 1) // group
        g2 = (intermediate_size_per_partition + group - 1) // group
        layer.num_groups_w13 = g13
        layer.num_groups_w2 = g2
        extra_weight_attrs.update({
            "quant_method": "group",
            "is_transposed": True,
        })

        def register(name, shape, dtype) -> None:
            value = torch.nn.Parameter(
                torch.empty(shape, dtype=dtype), requires_grad=False
            )
            layer.register_parameter(name, value)
            set_weight_attrs(value, extra_weight_attrs)

        register(
            "w13_qweight",
            (num_experts, hidden_size // self.quant_config.pack_factor,
             2 * intermediate_size_per_partition),
            torch.int32,
        )
        register(
            "w2_qweight",
            (num_experts,
             intermediate_size_per_partition // self.quant_config.pack_factor,
             hidden_size),
            torch.int32,
        )
        register(
            "w13_scales",
            (num_experts, g13, 2 * intermediate_size_per_partition),
            params_dtype,
        )
        register("w2_scales", (num_experts, g2, hidden_size), params_dtype)
        register(
            "w13_qzeros",
            (num_experts, g13,
             2 * intermediate_size_per_partition
             // self.quant_config.pack_factor),
            params_dtype,
        )
        register(
            "w2_qzeros",
            (num_experts, g2,
             hidden_size // self.quant_config.pack_factor),
            params_dtype,
        )
        set_weight_attrs(layer.w2_scales, {"load_full_w2": False})
        set_weight_attrs(layer.w2_qzeros, {"load_full_w2": False})
        register("w13_g_idx", (num_experts, hidden_size), torch.int32)
        register(
            "w2_g_idx",
            (num_experts, intermediate_size_per_partition),
            torch.int32,
        )
        register(
            "w13_g_idx_sort_indices",
            (num_experts, hidden_size),
            torch.int32,
        )
        register(
            "w2_g_idx_sort_indices",
            (num_experts, intermediate_size_per_partition),
            torch.int32,
        )
        layer.workspace = torch.empty(0, dtype=torch.int32)

    def process_weights_after_loading(self, layer) -> None:
        parallel = self.moe.moe_parallel_config
        if (
            parallel.tp_size != 1
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.pcp_size != 1
            or parallel.sp_size != 1
        ):
            raise RuntimeError("xR64ABS v0 requires TP1/EP1/DP1/PCP1/SP1")
        if self.quant_config.weight_bits != 4:
            raise RuntimeError("xR64ABS requires 4-bit expert weights")
        if self.quant_config.group_size != 128:
            raise RuntimeError("xR64ABS requires K128 weight scales")
        if not self.quant_config.is_sym or self.quant_config.desc_act:
            raise RuntimeError("xR64ABS requires symmetric RTN/GPTQ ordering")
        if not self.moe.is_act_and_mul:
            raise RuntimeError("xR64ABS requires gated FF1")
        if layer.shared_experts is not None:
            raise RuntimeError("xR64ABS v0 does not support shared experts")
        if layer.top_k != 8:
            raise RuntimeError(f"xR64ABS requires top_k=8, got {layer.top_k}")

        activation = os.environ.get(
            "XCALIBER_XR64_ACTIVATION", str(layer.activation)
        ).lower().rsplit(".", 1)[-1]
        if activation in {"gelu", "gelu_tanh", "gelu_pytorch_tanh"}:
            self._xr64_gelu_tanh = True
        elif activation in {"silu", "swiglu", "swish"}:
            self._xr64_gelu_tanh = False
        else:
            raise RuntimeError(
                f"xR64ABS does not support activation={activation!r}"
            )

        E = layer.w13_qweight.shape[0]
        H = layer.w13_qweight.shape[1] * self.quant_config.pack_factor
        I = layer.w13_qweight.shape[2] // 2
        if H % 32 or I % 32:
            raise RuntimeError(
                f"xR64ABS requires H/I divisible by 32, got H={H}, I={I}"
            )
        if tuple(layer.w2_qweight.shape) != (
            E, I // self.quant_config.pack_factor, H
        ):
            raise RuntimeError("xR64ABS raw W2 layout mismatch")
        if tuple(layer.w13_scales.shape) != (
            E, (H + 127) // 128, 2 * I
        ):
            raise RuntimeError("xR64ABS raw S13 layout mismatch")
        if tuple(layer.w2_scales.shape) != (
            E, (I + 127) // 128, H
        ):
            raise RuntimeError("xR64ABS raw S2 tail-group layout mismatch")

        per_expert_scale = getattr(layer, "_xcaliber_per_expert_scale", None)
        if per_expert_scale is None or per_expert_scale.numel() != E:
            raise RuntimeError("xR64ABS requires Gemma per_expert_scale [E]")

        replace_parameter(
            layer, "w13_qweight", _C.repack_gptq(layer.w13_qweight.detach())
        )
        replace_parameter(
            layer, "w2_qweight", _C.repack_gptq(layer.w2_qweight.detach())
        )
        replace_parameter(
            layer,
            "w13_scales",
            layer.w13_scales.detach().permute(0, 2, 1).float().contiguous(),
        )
        replace_parameter(
            layer,
            "w2_scales",
            layer.w2_scales.detach().permute(0, 2, 1).float().contiguous(),
        )
        layer.w13_weight = layer.w13_qweight
        layer.w2_weight = layer.w2_qweight

        for name in (
            "w13_qzeros",
            "w2_qzeros",
            "w13_g_idx",
            "w2_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
        ):
            old = getattr(layer, name, None)
            if old is not None:
                replace_parameter(
                    layer,
                    name,
                    torch.empty(0, dtype=old.dtype, device=old.device),
                )
        if getattr(layer, "workspace", None) is not None:
            layer.workspace = torch.empty(
                0, dtype=layer.workspace.dtype, device=layer.workspace.device
            )

        self._xr64_per_expert_scale = (
            per_expert_scale.detach().to(dtype=torch.bfloat16).contiguous()
        )
        self._xr64_ready = True
        logger.info(
            "xR64ABS active: E=%d H=%d I=%d S2_groups=%d activation=%s",
            E,
            H,
            I,
            (I + 127) // 128,
            "gelu_tanh" if self._xr64_gelu_tanh else "swiglu",
        )

    def apply_monolithic(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._xr64_ready:
            raise RuntimeError("xR64ABS weights were not post-processed")
        if getattr(layer, "expert_map", None) is not None:
            raise RuntimeError("xR64ABS v0 does not support expert remapping")
        if getattr(layer, "apply_router_weight_on_input", False):
            raise RuntimeError(
                "xR64ABS folds router weights after gated FF1"
            )
        return _C.forward(
            x.contiguous(),
            router_logits.to(dtype=torch.bfloat16).contiguous(),
            self._xr64_per_expert_scale,
            layer.w13_qweight,
            layer.w13_scales,
            layer.w2_qweight,
            layer.w2_scales,
            self._xr64_gelu_tanh,
        )


class XCaliberR64ABSCompressedTensorsMoE(
    CompressedTensorsWNA16MoEMethod
):
    """Compressed-tensors loader plus xR64ABS routed executor."""

    def __init__(
        self,
        weight_quant,
        input_quant,
        moe_config,
        layer_name: str | None = None,
    ) -> None:
        super().__init__(weight_quant, input_quant, moe_config, layer_name)
        self._xr64_ready = False
        self._xr64_gelu_tanh = False
        self._xr64_per_expert_scale = None

    @property
    def supports_internal_mk(self) -> bool:
        return True

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer) -> None:
        parallel = self.moe.moe_parallel_config
        if (
            parallel.tp_size != 1
            or parallel.ep_size != 1
            or parallel.dp_size != 1
            or parallel.pcp_size != 1
            or parallel.sp_size != 1
        ):
            raise RuntimeError("xR64ABS v0 requires TP1/EP1/DP1/PCP1/SP1")
        if self.num_bits != 4 or self.group_size != 128:
            raise RuntimeError("xR64ABS requires symmetric W4G128 weights")
        if not self.weight_quant.symmetric or self.input_quant is not None:
            raise RuntimeError("xR64ABS requires symmetric W4A16 weights")
        actorder = getattr(
            self.weight_quant.actorder,
            "value",
            self.weight_quant.actorder,
        )
        if str(actorder).lower() in {"group", "dynamic"}:
            raise RuntimeError(
                f"xR64ABS cannot consume runtime actorder={actorder}"
            )
        if not self.moe.is_act_and_mul:
            raise RuntimeError("xR64ABS requires gated FF1")
        if getattr(layer, "shared_experts", None) is not None:
            raise RuntimeError("xR64ABS v0 does not support shared experts")
        if layer.top_k != 8:
            raise RuntimeError(f"xR64ABS requires top_k=8, got {layer.top_k}")

        activation = os.environ.get(
            "XCALIBER_XR64_ACTIVATION", str(layer.activation)
        ).lower().rsplit(".", 1)[-1]
        if activation in {"gelu", "gelu_tanh", "gelu_pytorch_tanh"}:
            self._xr64_gelu_tanh = True
        elif activation in {"silu", "swiglu", "swish"}:
            self._xr64_gelu_tanh = False
        else:
            raise RuntimeError(
                f"xR64ABS does not support activation={activation!r}"
            )

        E = layer.w13_weight_packed.shape[0]
        H = layer.w13_weight_packed.shape[1] * self.packed_factor
        I = layer.w13_weight_packed.shape[2] // 2
        if H % 32 or I % 32:
            raise RuntimeError(
                f"xR64ABS requires H/I divisible by 32, got H={H}, I={I}"
            )
        if tuple(layer.w2_weight_packed.shape) != (
            E, I // self.packed_factor, H
        ):
            raise RuntimeError("xR64ABS raw W2 layout mismatch")
        if tuple(layer.w13_weight_scale.shape) != (
            E, H // 128, 2 * I
        ):
            raise RuntimeError("xR64ABS raw S13 layout mismatch")
        if tuple(layer.w2_weight_scale.shape) != (
            E, I // 128, H
        ):
            raise RuntimeError("xR64ABS raw S2 layout mismatch")

        replace_parameter(
            layer,
            "w13_weight_packed",
            _C.repack_gptq(layer.w13_weight_packed.detach()),
        )
        replace_parameter(
            layer,
            "w2_weight_packed",
            _C.repack_gptq(layer.w2_weight_packed.detach()),
        )
        replace_parameter(
            layer,
            "w13_weight_scale",
            layer.w13_weight_scale.detach().permute(0, 2, 1).float().contiguous(),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            layer.w2_weight_scale.detach().permute(0, 2, 1).float().contiguous(),
        )

        for name in (
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
            "w13_weight_shape",
            "w2_weight_shape",
        ):
            old = getattr(layer, name, None)
            if old is not None:
                replace_parameter(
                    layer,
                    name,
                    torch.empty(0, dtype=old.dtype, device=old.device),
                )

        self._xr64_per_expert_scale = torch.ones(
            E,
            dtype=torch.bfloat16,
            device=layer.w13_weight_packed.device,
        )
        self._xr64_ready = True
        logger.info(
            "xR64ABS compressed-tensors active: E=%d H=%d I=%d "
            "activation=%s actorder=%s",
            E,
            H,
            I,
            "gelu_tanh" if self._xr64_gelu_tanh else "swiglu",
            actorder,
        )

    def apply_monolithic(
        self,
        layer,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._xr64_ready:
            raise RuntimeError("xR64ABS weights were not post-processed")
        if getattr(layer, "expert_map", None) is not None:
            raise RuntimeError("xR64ABS v0 does not support expert remapping")
        if getattr(layer, "apply_router_weight_on_input", False):
            raise RuntimeError(
                "xR64ABS folds router weights after gated FF1"
            )
        return _C.forward(
            x.contiguous(),
            router_logits.to(dtype=torch.bfloat16).contiguous(),
            self._xr64_per_expert_scale,
            layer.w13_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_packed,
            layer.w2_weight_scale,
            self._xr64_gelu_tanh,
        )


# vLLM 0.19.1 gates packed checkpoint transposition on this exact class name.
XCaliberR64ABSCompressedTensorsMoE.__name__ = (
    "CompressedTensorsWNA16MoEMethod"
)
