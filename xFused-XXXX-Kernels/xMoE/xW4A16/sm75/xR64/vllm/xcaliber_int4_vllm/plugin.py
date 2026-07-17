from __future__ import annotations

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
