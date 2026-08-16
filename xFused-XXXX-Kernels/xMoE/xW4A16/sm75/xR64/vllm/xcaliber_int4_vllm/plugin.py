from __future__ import annotations

import torch

from vllm.logger import init_logger


logger = init_logger(__name__)


def _bind_gemma_expert_scale() -> None:
    try:
        from vllm.model_executor.models.gemma4 import Gemma4MoE
    except (AttributeError, ImportError):
        return

    if getattr(Gemma4MoE, "_xcaliber_abs_scale_bound", False):
        return
    original_init = Gemma4MoE.__init__

    def init(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        object.__setattr__(
            self.experts,
            "_xcaliber_per_expert_scale",
            self.per_expert_scale,
        )

    Gemma4MoE.__init__ = init
    Gemma4MoE._xcaliber_abs_scale_bound = True


def register() -> None:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsMoEMethod,
    )
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinConfig,
        GPTQMarlinMoEMethod,
    )
    from vllm.model_executor.layers.quantization.inc import INCConfig

    from .method import (
        XCaliberR64ABSAutoRoundMoE,
        XCaliberR64ABSCompressedTensorsMoE,
    )

    if not getattr(INCConfig, "_xcaliber_abs_registered", False):
        original = INCConfig.apply_gptq_quant_layer

        def apply_gptq_quant_layer(
            self,
            layer,
            prefix: str,
            backend: str = "auto",
        ):
            if isinstance(layer, FusedMoE):
                bits, group_size, sym = self.get_layer_config(layer, prefix)
                if self.check_quantized(bits) and (
                    bits == 4 and group_size == 128 and sym
                ):
                    config = GPTQMarlinConfig(
                        weight_bits=bits,
                        group_size=group_size,
                        is_sym=sym,
                        lm_head_quantized=False,
                        desc_act=False,
                        dynamic={},
                        full_config={},
                    )
                    return XCaliberR64ABSAutoRoundMoE(
                        config, layer.moe_config
                    )
            return original(self, layer, prefix, backend)

        INCConfig.apply_gptq_quant_layer = apply_gptq_quant_layer
        INCConfig._xcaliber_abs_registered = True

    # Plain-GPTQ checkpoints (quant_method "gptq", e.g. palmfuture Qwen3.6)
    # reach FusedMoE via GPTQMarlinConfig.get_quant_method -> GPTQMarlinMoEMethod
    # directly, bypassing the INC and compressed-tensors hooks above. Wrap
    # get_quant_method so symmetric W4G128 (desc_act false) FusedMoE layers get
    # our monolithic executor while everything else (Linear layers, dynamic-
    # excluded modules that come back Unquantized, marlin-unsupported shapes
    # that fall back to MoeWNA16) is left untouched.
    if not getattr(GPTQMarlinConfig, "_xcaliber_abs_gptq_registered", False):
        original_get_quant_method = GPTQMarlinConfig.get_quant_method

        def gptq_get_quant_method(self, layer, prefix: str):
            method = original_get_quant_method(self, layer, prefix)
            if (
                isinstance(layer, FusedMoE)
                and type(method) is GPTQMarlinMoEMethod
                and getattr(method, "quant_config", None) is not None
            ):
                qc = method.quant_config
                if (
                    qc.weight_bits == 4
                    and qc.group_size == 128
                    and qc.is_sym
                    and not qc.desc_act
                ):
                    xmethod = XCaliberR64ABSAutoRoundMoE(qc, layer.moe_config)
                    xmethod.input_dtype = getattr(method, "input_dtype", None)
                    return xmethod
            return method

        GPTQMarlinConfig.get_quant_method = gptq_get_quant_method
        GPTQMarlinConfig._xcaliber_abs_gptq_registered = True

    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinLinearMethod,
    )

    if not getattr(GPTQMarlinLinearMethod, "_xcaliber_ceil_scales", False):
        # vLLM 0.19.1 sizes GPTQ scale/zero rows with FLOOR division
        # (input_size // group_size). Checkpoints whose K is not divisible
        # by the group size (Gemma-4 dense down_proj: K=2112, g=128 ->
        # 16.5 groups) carry a CEIL-sized tail group (17 rows), so the
        # floor allocation silently truncates one scale row at load time
        # and ConchLinearKernel (which indexes ceil groups) reads one row
        # past the tensor: NaN corruption or illegal memory access.
        # Marlin-eligible shapes have floor == ceil, so this is a no-op
        # for them.
        original_linear_create_weights = GPTQMarlinLinearMethod.create_weights

        def linear_create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            original_linear_create_weights(
                self,
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            group = self.quant_config.group_size
            if group is None or group <= 0:
                return
            for name in ("scales", "qzeros"):
                param = getattr(layer, name, None)
                if param is None or param.data.ndim < 2:
                    continue
                rows = param.data.shape[0]
                for k in (input_size, input_size_per_partition):
                    if k % group and rows == k // group:
                        fixed = torch.empty(
                            (k // group + 1,) + tuple(param.data.shape[1:]),
                            dtype=param.data.dtype,
                            device=param.data.device,
                        )
                        param.data = fixed
                        logger.info(
                            "xCaliber ceil-scales fix: %s %s -> %s "
                            "(K=%d, group=%d)",
                            name, rows, k // group + 1, k, group,
                        )
                        break

        GPTQMarlinLinearMethod.create_weights = linear_create_weights
        GPTQMarlinLinearMethod._xcaliber_ceil_scales = True

    if not getattr(
        CompressedTensorsMoEMethod,
        "_xcaliber_abs_registered",
        False,
    ):
        original_get_moe_method = CompressedTensorsMoEMethod.get_moe_method

        def get_moe_method(quant_config, layer, layer_name: str):
            quant_config._add_fused_moe_to_target_scheme_map()
            scheme_dicts = [
                quant_config.get_scheme_dict(
                    layer,
                    layer_name + projection,
                )
                for projection in (
                    ".0.gate_proj",
                    ".0.up_proj",
                    ".0.down_proj",
                )
            ]
            scheme = scheme_dicts[0]
            if scheme is not None and all(
                current == scheme for current in scheme_dicts[1:]
            ):
                weight_quant = scheme.get("weights")
                input_quant = scheme.get("input_activations")
                strategy = getattr(
                    getattr(weight_quant, "strategy", None),
                    "value",
                    getattr(weight_quant, "strategy", None),
                )
                actorder = getattr(
                    getattr(weight_quant, "actorder", None),
                    "value",
                    getattr(weight_quant, "actorder", None),
                )
                if (
                    weight_quant is not None
                    and weight_quant.num_bits == 4
                    and weight_quant.group_size == 128
                    and weight_quant.symmetric
                    and input_quant is None
                    and str(strategy).lower() == "group"
                    and str(actorder).lower() == "static"
                    and scheme.get("format") == "pack-quantized"
                ):
                    return XCaliberR64ABSCompressedTensorsMoE(
                        weight_quant,
                        input_quant,
                        layer.moe_config,
                        layer_name,
                    )
            return original_get_moe_method(quant_config, layer, layer_name)

        CompressedTensorsMoEMethod.get_moe_method = staticmethod(
            get_moe_method
        )
        CompressedTensorsMoEMethod._xcaliber_abs_registered = True

    _bind_gemma_expert_scale()
    logger.info(
        "xCaliber xR64ABS registered for symmetric W4G128 MoE"
    )
