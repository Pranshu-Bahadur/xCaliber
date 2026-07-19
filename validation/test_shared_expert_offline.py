"""Offline shared-expert smoke test for the xR64ABS vLLM plugin.

Loads a plain-GPTQ Qwen3.6 MoE checkpoint (which carries a shared expert),
greedy-generates a handful of fixed prompts, and reports whether the xR64ABS
monolithic executor actually activated for the routed experts.

GPU is required -- this script is NOT run as part of the code review; it is the
artifact to run later once a GPU is free. Point CUDA_VISIBLE_DEVICES at an idle
device before running.

    CUDA_VISIBLE_DEVICES=0 python validation/test_shared_expert_offline.py
"""

from __future__ import annotations

import logging
import os
import sys


# --- environment (must be set before importing vllm) -----------------------
os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU", "0"))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_PLUGINS", "xcaliber_xr64_abs")

MODEL_ID = os.environ.get(
    "MODEL_ID", "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4"
)
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))


PROMPTS = (
    "Explain, in two sentences, what a mixture-of-experts shared expert does.",
    "Write a haiku about coalesced GPU memory loads.",
    "What is 248 * 17? Show the arithmetic briefly.",
    "Summarize the plot of Romeo and Juliet in one sentence.",
)


class CaptureHandler(logging.Handler):
    """AA2-style handler: mirror plugin logs to stdout and remember them so we
    can assert that the xR64ABS executor announced itself."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[str] = []
        self.setFormatter(
            logging.Formatter("[%(name)s %(levelname)s] %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.records.append(msg)
        print(msg, file=sys.stderr, flush=True)

    def saw_xr64_active(self) -> bool:
        return any(
            "xR64ABS" in r and "active" in r for r in self.records
        )


def install_capture() -> CaptureHandler:
    handler = CaptureHandler()
    logger = logging.getLogger("xcaliber_int4_vllm")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = True
    return handler


def main() -> int:
    capture = install_capture()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=0)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    print("\n" + "=" * 72)
    for i, (prompt, output) in enumerate(zip(PROMPTS, outputs)):
        response = output.outputs[0]
        print(f"----- PROMPT {i}: {prompt}")
        print(f"FINISH_REASON: {response.finish_reason}")
        print(f"OUTPUT_TOKENS: {len(response.token_ids)}")
        print(response.text.strip())
        print()

    active = capture.saw_xr64_active()
    print("=" * 72)
    print(f"xR64ABS active log observed: {active}")
    if not active:
        print(
            "WARNING: xR64ABS did not activate -- experts likely ran on the "
            "stock marlin path (check quant config / plugin registration)."
        )
    return 0 if active else 2


if __name__ == "__main__":
    raise SystemExit(main())
